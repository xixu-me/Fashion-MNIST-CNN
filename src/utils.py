import os

import torch


def save_checkpoint(
    model, optimizer, epoch, loss, accuracy, path="models/checkpoint.pth"
):
    """
    保存模型检查点

    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失值
        accuracy: 当前准确率
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    torch.save(checkpoint, path)


class EarlyStopping:
    """
    早停机制类
    用于监控验证损失，防止过拟合

    参数:
        patience: 容忍验证损失不下降的轮数
        min_delta: 最小变化阈值
    """

    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # 计数器
        self.best_loss = None  # 最佳损失值
        self.early_stop = False  # 早停标志

    def __call__(self, val_loss):
        """
        检查是否应该早停

        参数:
            val_loss: 当前验证损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
