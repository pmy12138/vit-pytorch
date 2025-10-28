import os
import pickle
import numpy as np
from PIL import Image
import urllib.request
import tarfile


def download_cifar10(data_dir='./data'):
    """下载 CIFAR-10 数据集"""
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(filepath):
        print(f"Downloading CIFAR-10 from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete!")

        # 解压
    print("Extracting files...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete!")

    return os.path.join(data_dir, 'cifar-10-batches-py')


def organize_cifar10(cifar_dir, output_dir='./cifar10'):
    """将 CIFAR-10 组织成文件夹结构"""
    # CIFAR-10 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 创建输出目录
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

            # 处理训练集
    print("Processing training data...")
    for i in range(1, 6):
        batch_file = os.path.join(cifar_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')

        images = batch_data[b'data']
        labels = batch_data[b'labels']
        filenames = batch_data[b'filenames']

        for idx, (img_data, label, filename) in enumerate(zip(images, labels, filenames)):
            # 重塑图像数据 (3072,) -> (3, 32, 32) -> (32, 32, 3)
            img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img)

            # 保存图像
            class_name = class_names[label]
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')
            save_path = os.path.join(output_dir, class_name, f'train_{i}_{filename}')
            img.save(save_path)

            # 处理测试集
    print("Processing test data...")
    test_file = os.path.join(cifar_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')

    images = test_data[b'data']
    labels = test_data[b'labels']
    filenames = test_data[b'filenames']

    for idx, (img_data, label, filename) in enumerate(zip(images, labels, filenames)):
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)

        class_name = class_names[label]
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        save_path = os.path.join(output_dir, class_name, f'test_{filename}')
        img.save(save_path)

    print(f"Dataset organized successfully in {output_dir}")
    print(f"Total classes: {len(class_names)}")
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg'))])
        print(f"  {class_name}: {num_images} images")


if __name__ == '__main__':
    # 下载数据集
    cifar_dir = download_cifar10()

    # 组织数据集
    organize_cifar10(cifar_dir, output_dir='./cifar10')

    print("\nDone! You can now use the dataset with:")
    print("python train_baseline.py --data-path ./cifar10 --num_classes 10 --batch-size 128 --epochs 100")