import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import EarlyStopping, save_checkpoint


def train_model(model, train_loader, test_loader, device, num_epochs=50):
    """
    模型训练函数

    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备（CPU/GPU）
        num_epochs: 训练轮数

    返回:
        训练损失、测试损失、训练准确率和测试准确率的历史记录
    """
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )  # 定义优化器
    early_stopping = EarlyStopping(patience=5)  # 早停机制
    train_losses, test_losses = [], []  # 记录损失
    train_acc, test_acc = [], []  # 记录准确率

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print("-" * 50)
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # 训练循环
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 计算训练指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练集上的平均损失和准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # 测试阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # 测试循环
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算测试集上的平均损失和准确率
        test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct / total

        # 记录训练历史
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)

        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # 早停检查
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # 保存检查点
        save_checkpoint(model, optimizer, epoch, test_loss, test_accuracy)

    return train_losses, test_losses, train_acc, test_acc
