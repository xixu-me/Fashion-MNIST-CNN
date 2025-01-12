import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Fashion-MNIST 数据集的 CNN 模型
    包含 3 个卷积层和 2 个全连接层
    使用批归一化和 Dropout 来防止过拟合
    """

    def __init__(self, dropout_rate=0.3):
        super(FashionCNN, self).__init__()
        # 卷积层序列：包含卷积、ReLU 激活、批归一化、Dropout 和最大池化
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 第一个卷积层
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 第二个卷积层
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 第三个卷积层
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
        )
        # 全连接层序列：展平操作、两个线性层和 Dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),  # 第一个全连接层
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.2),
            nn.Linear(512, 10),  # 输出层，10 个类别
        )

    def forward(self, x):
        """前向传播函数"""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
