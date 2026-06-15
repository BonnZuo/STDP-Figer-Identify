import torch
import torch.nn as nn


class LinearReadout(nn.Module):
    """
    线性读出层: 将 STDP firing-rate 特征直接映射为类别 logits。

    结构: firing_rates (n_features,) → Linear(n_features, 10) → logits

    作为基准模型，容量较低，准确率约 87-88%。
    """

    def __init__(self, n_features: int = 400, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, n_features) firing rate 特征向量。
        返回:
            (batch, n_classes) 类别 logits。
        """
        return self.fc(x)


class MLPReadout(nn.Module):
    """
    2 层 MLP 读出层: 比线性读出层更强的分类能力，准确率 94%+。

    结构:
        firing_rates (400)
        → Linear(400, 256) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(256, 10) → logits

    参数:
        n_features: 输入特征维度 (STDP 兴奋神经元数)，默认 400。
        n_hidden:   隐藏层神经元数，默认 256。
        n_classes:  输出类别数 (10 类数字)，默认 10。
        dropout:    Dropout 概率，默认 0.3。
    """

    def __init__(
        self,
        n_features: int = 400,
        n_hidden: int = 256,
        n_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, n_features) firing rate 特征向量。
        返回:
            (batch, n_classes) 类别 logits。
        """
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 默认使用 MLP 以获得最佳准确率
Readout = MLPReadout


def train_readout(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "cpu",
    use_mlp: bool = True,
) -> nn.Module:
    """
    在 STDP 提取的特征上训练读出层。

    使用 AdamW 优化器 + CosineAnnealing 学习率调度 + CrossEntropy 损失。
    训练集上随机打乱后按 mini-batch 训练。

    参数:
        features:     (n_samples, n_features) firing rate 特征向量。
        labels:       (n_samples,) 整数类别标签 0-9。
        n_epochs:     训练轮数，默认 100 (线性) / 300 (MLP 自动调大)。
        lr:           初始学习率，默认 0.001。
        weight_decay: L2 正则化强度，默认 1e-4。
        batch_size:   mini-batch 大小，默认 512。
        device:       PyTorch 设备，默认 "cpu"。
        use_mlp:      True 用 MLP，False 用线性读出层。

    返回:
        训练好的模型 (eval 模式，已移到 CPU)。
    """
    n_features = features.shape[1]
    if use_mlp:
        model = MLPReadout(n_features=n_features).to(device)
        n_epochs = max(n_epochs, 300)  # MLP 需要更多轮
    else:
        model = LinearReadout(n_features=n_features).to(device)

    features = features.to(device)
    labels = labels.to(device)

    # AdamW: 带解耦权重衰减的 Adam，正则化效果更好
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # 余弦退火: 学习率从 lr 平滑降至 ~0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    n_samples = features.shape[0]
    indices = torch.randperm(n_samples)  # 全量随机打乱

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            x, y = features[batch_idx], labels[batch_idx]

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        scheduler.step()
        epoch_loss /= n_samples

        # 每 50 轮打印一次训练精度
        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(features).argmax(dim=1)
                acc = (pred == labels).float().mean().item()
            model.train()
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"loss={epoch_loss:.4f}  train_acc={acc:.2%}"
            )

    return model.eval().cpu()


def evaluate_readout(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    评估读出层在给定特征和标签上的分类准确率。

    参数:
        model:    已训练的读出层模型 (eval 模式)。
        features: (n_samples, n_features) 特征向量。
        labels:   (n_samples,) 真实标签。

    返回:
        float 准确率 (0-1)。
    """
    model = model.to(features.device)
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        acc = (preds == labels.to(features.device)).float().mean().item()
    return acc
