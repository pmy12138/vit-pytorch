import os
import math
import argparse
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets

from vit_model import VisionTransformer
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 创建实验目录
    exp_name = f"baseline_patch{args.patch_size}_dim{args.embed_dim}_depth{args.depth}_heads{args.num_heads}"
    exp_dir = os.path.join("./experiments", exp_name)
    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))

    # 数据集划分
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
               # transforms.Resize(224),
               transforms.RandomCrop(32, padding=4),  # 适应CIFAR-10的32x32尺寸
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色增强，增加数据多样性
               transforms.ToTensor(),
               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])]),  # CIFAR-10 标准归一化参数
        "val": transforms.Compose([
               # transforms.Resize(224),
               # transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])}

    # 使用CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=data_transform["train"])
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                   transform=data_transform["val"])

    batch_size = args.batch_size  # 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               # collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             # collate_fn=val_dataset.collate_fn
                                             )

    model = VisionTransformer(
        img_size=32,  # CIFAR-10图像尺寸 32
        patch_size=16,  # 基线模型patch大小
        in_c=3,
        num_classes=10,  # CIFAR-10类别数
        embed_dim=768,  # 基线模型嵌入维度  256？
        depth=12,  # 基线模型Block层数
        num_heads=12,  # 基线模型注意力头数
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_ratio=0.1,  # 基线Dropout比例0.1  增加到0.2
        attn_drop_ratio=0.1,  # 增加到0.2
        representation_size=None,
        distilled=False
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结参数
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)  # 从0.05增加到0.1
    #
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练记录
    # train_history = {
    #     "train_loss": [],
    #     "train_acc": [],
    #     "val_loss": [],
    #     "val_acc": [],
    #     "learning_rate": []
    # }

    best_val_acc = 0.0
    best_epoch = 0

    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            scheduler.step()

            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            # # 记录指标
            # train_history["train_loss"].append(train_loss)
            # train_history["train_acc"].append(train_acc)
            # train_history["val_loss"].append(val_loss)
            # train_history["val_acc"].append(val_acc)
            # train_history["learning_rate"].append(optimizer.param_groups[0]["lr"])

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))

            # 保存检查点
            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

            # 保存训练历史
    # with open(os.path.join(exp_dir, "train_history.json"), 'w') as f:
    #     json.dump(train_history, f, indent=4)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # 保存配置信息
    config = {
        "model": "ViT-Base",
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "initial_lr": args.lr,
        "final_lr_factor": args.lrf,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-5,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch
    }

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)         #0.0005
    parser.add_argument('--lrf', type=float, default=0.01)

    # 模型配置
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)       # 256？
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default='./cifar10')
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='头训练，不加载预训练权重')
    parser.add_argument('--freeze-layers', type=bool, default=False)  # 不冻结层
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
