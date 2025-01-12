import os

import matplotlib.pyplot as plt
import torch


def plot_training_history(train_losses, val_losses, train_acc, val_acc):
    """
    绘制训练历史图表

    参数:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        train_acc: 训练准确率历史
        val_acc: 验证准确率历史
    """
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 保存图表
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_history.png")
    plt.close()


def plot_predictions(model, test_loader, class_labels, num_samples=10):
    """
    绘制模型预测结果

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        class_labels: 类别标签
        num_samples: 显示样本数量
    """
    model.eval()
    fig = plt.figure(figsize=(15, 3))
    device = next(model.parameters()).device

    # 获取预测结果
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 绘制每个样本
            ax = fig.add_subplot(1, num_samples, i + 1)
            img = images[0].cpu().numpy().squeeze()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(
                f"Pred: {class_labels[predicted[0].item()]}\nTrue: {class_labels[labels[0]]}",
                color=("green" if predicted[0].item() == labels[0] else "red"),
            )

    # 保存图表
    plt.tight_layout()
    plt.savefig("results/predictions.png")
    plt.close()
