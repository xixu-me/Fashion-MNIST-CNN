import torch

from src.dataset import class_labels, get_data_loaders
from src.model import FashionCNN
from src.train import train_model
from src.visualize import plot_predictions, plot_training_history


def main():
    """
    主函数
    执行模型训练和结果可视化的完整流程
    """
    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size=64)

    # 创建并训练模型
    model = FashionCNN().to(device)
    train_losses, test_losses, train_acc, test_acc = train_model(
        model, train_loader, test_loader, device
    )

    # 可视化训练结果
    plot_training_history(train_losses, test_losses, train_acc, test_acc)
    plot_predictions(model, test_loader, class_labels)


if __name__ == "__main__":
    main()
