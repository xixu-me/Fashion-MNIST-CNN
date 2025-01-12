from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    """
    获取 Fashion-MNIST 数据集的数据加载器

    参数:
        batch_size: 批次大小

    返回:
        训练数据加载器和测试数据加载器
    """
    # 定义数据转换
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # 加载训练集
    train_dataset = datasets.FashionMNIST(
        "data", train=True, download=True, transform=transform
    )

    # 加载测试集
    test_dataset = datasets.FashionMNIST(
        "data", train=False, download=True, transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 类别标签
class_labels = [
    "T-shirt/top",  # T 恤/上衣
    "Trouser",  # 裤子
    "Pullover",  # 套头衫
    "Dress",  # 连衣裙
    "Coat",  # 外套
    "Sandal",  # 凉鞋
    "Shirt",  # 衬衫
    "Sneaker",  # 运动鞋
    "Bag",  # 包
    "Ankle boot",  # 短靴
]
