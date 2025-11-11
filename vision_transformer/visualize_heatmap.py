# visualize_heatmap_hybrid.py
# ---------------------------------------------------------------
# 适配 HybridViT（ResNet + Transformer）混合架构的热力图可视化脚本
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from hybrid_model import HybridViT  # ✅ 混合模型
from vit_model import VisionTransformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ===============================================================
# 🔹 特征提取类：用于提取指定层的输出
# ===============================================================
class FeatureExtractor:
    """用于注册forward hook并获取中间层特征"""
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []

        # 注册前向钩子  hybrid
        # for name, module in model.named_modules():
        #     # 兼容 HybridViT: cnn.layer1、blocks.7
        #     if any(name.endswith(t) for t in target_layers):
        #         hook = module.register_forward_hook(self._get_features(name))
        #         self.hooks.append(hook)

        #  vit
        for name, module in model.named_modules():
            # ViT 模型中 block 层的名字形如 blocks.0, blocks.1, ...
            if any(name.endswith(t) for t in target_layers):
                hook = module.register_forward_hook(self._hook(name))
                self.hooks.append(hook)

    def _get_features(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def _hook(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach()

        return fn

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


# ===============================================================
# 🔹 将特征图生成热力图
# ===============================================================
def generate_heatmap(feature_map):
    """
    Args:
        feature_map: (C, H, W)
    Returns:
        heatmap: (H, W)
    """
    heatmap = torch.mean(feature_map, dim=0).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """叠加原图与热力图"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    overlayed = alpha * heatmap_colored + (1 - alpha) * image
    return overlayed


# ===============================================================
# 🔹 主函数：生成HybridViT模型的浅层/深层注意力热力图
# ===============================================================
def visualize_model_attention(model_path, data_root='./data', num_samples=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------- 加载模型 --------------------
    # net = HybridViT(num_classes=10)
    # state_dict = torch.load(model_path, map_location=device)
    # net.load_state_dict(state_dict, strict=False)  # 防止 pos_embed 报错
    # net.to(device)
    # net.eval()

    net = VisionTransformer(
        img_size=32, patch_size=8, in_c=3, num_classes=10,
        embed_dim=384, depth=8, num_heads=8, mlp_ratio=4.0,
        qkv_bias=True, drop_ratio=0.2, attn_drop_ratio=0.2, drop_path_ratio=0.1
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    # -------------------- 数据加载 --------------------
    transform_vis = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_vis)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # -------------------- 注册Hook层 --------------------
    # hybrid 浅层: cnn.layer1, 深层: 最后一个 Transformer block (blocks.7)
    # target_layers = ['cnn.layer1', 'blocks.7']

    # vit 浅层: 第1个Transformer Block；深层: 最后一个Block
    target_layers = ['blocks.0', 'blocks.7']

    extractor = FeatureExtractor(net, target_layers)

    # -------------------- 可视化多个样本 --------------------
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for idx in range(num_samples):
        image, label = test_dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()  # (C,H,W)->(H,W,C)

        # 归一化（与训练一致）
        image_normalized = transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010]
        )(image)

        with torch.no_grad():
            output = net(image_normalized.unsqueeze(0).to(device))
            pred_class = torch.argmax(output, dim=1).item()

        # -------------------- 特征提取 --------------------
        # shallow_features = extractor.features['cnn.layer1'][0]  # (C,H,W)

        # # Transformer 输出 [B, N, D]
        # deep_feat = extractor.features['blocks.7']  # (B,N,D)
        # deep_feat = deep_feat[:, 1:, :]  # 去掉 cls_token
        # num_patches = deep_feat.shape[1]
        # side = int(num_patches ** 0.5)  # 假设为正方形
        # deep_feat_map = deep_feat[0].transpose(0, 1).reshape(-1, side, side)  # (C,H,W)

        shallow_feat = extractor.features['blocks.0']  # (B, N, D)
        deep_feat = extractor.features['blocks.7']  # (B, N, D)

        # 去掉 cls_token, 还原为空间结构
        shallow_feat = shallow_feat[:, 1:, :].reshape(-1, 4, 4, 384).permute(0, 3, 1, 2)  # (B,C,H,W)
        deep_feat = deep_feat[:, 1:, :].reshape(-1, 4, 4, 384).permute(0, 3, 1, 2)

        shallow_heatmap = generate_heatmap(shallow_feat[0])
        deep_heatmap = generate_heatmap(deep_feat[0])

        shallow_overlay = overlay_heatmap(image_np, shallow_heatmap)
        deep_overlay = overlay_heatmap(image_np, deep_heatmap)

        # -------------------- hybrid 生成热力图 --------------------
        # shallow_heatmap = generate_heatmap(shallow_features)
        # deep_heatmap = generate_heatmap(deep_feat_map)
        #
        # shallow_overlay = overlay_heatmap(image_np, shallow_heatmap)
        # deep_overlay = overlay_heatmap(image_np, deep_heatmap)

        # -------------------- 绘制结果 --------------------
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f'Original\nTrue: {class_names[label]}\nPred: {class_names[pred_class]}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(shallow_heatmap, cmap='jet')
        axes[idx, 1].set_title('CNN Layer1 Heatmap')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(shallow_overlay)
        axes[idx, 2].set_title('CNN Layer1 Overlay')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(deep_overlay)
        axes[idx, 3].set_title('Transformer Block7 Overlay')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    # plt.savefig('hybrid_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.savefig('vit3.3_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()

    extractor.remove_hooks()
    # print("✅ 热力图已保存为 hybrid_heatmaps.png")
    print("✅ 热力图已保存为 vit3.3_heatmaps.png")


# ===============================================================
# 🔹 主入口
# ===============================================================
if __name__ == '__main__':
    visualize_model_attention(
        # model_path='experiments/model_hybrid_patch8_dim128_depth8_heads8/weights/best_model.pth',
        model_path='experiments/model3.3_patch8_dim384_depth8_heads8/weights/best_model.pth',
        data_root='./data',
        num_samples=5
    )
