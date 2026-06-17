import torch
import torch.nn as nn


class LinearReadout(nn.Module):
    """
    线性读出层: 将 STDP firing-rate 特征直接映射为类别 logits。没什么用。。。

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
        dropout: float = 0.3,#随机失活，抑制过拟合
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    #向前传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, n_features) firing rate 特征向量。
        返回:
            (batch, n_classes) 类别 logits。
        """
        x = self.fc1(x)#经过第一层全连接，
        x = self.bn(x)#批量归一化
        x = torch.relu(x)#relu激活
        x = self.dropout(x)#随机屏蔽神经元
        x = self.fc2(x)#映射到10维


# 调用时只 MLP from readout import Readout
Readout = MLPReadout


def train_readout(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 0.001,#初始化cpu
    weight_decay: float = 1e-4,
    batch_size: int = 512,#batch——size大小
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
    if use_mlp:#使用mlp方案
        model = MLPReadout(n_features=n_features).to(device)
        n_epochs = max(n_epochs, 300)  # MLP 需要更多轮
    else:
        model = LinearReadout(n_features=n_features).to(device)

    #数据迁移到计算设备
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
    criterion = nn.CrossEntropyLoss()#交叉熵损失

    #样本打乱，构造随机索引
    n_samples = features.shape[0]
    indices = torch.randperm(n_samples)  # 全量随机打乱

    #完整的epoch训练循环
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):#按batch_size分批训练
            batch_idx = indices[start : start + batch_size]
            x, y = features[batch_idx], labels[batch_idx]

            logits = model(x)#前向传播
            loss = criterion(logits, y)#计算分类损失

            optimizer.zero_grad()#清空上一轮梯度
            loss.backward()#反向传播，计算梯度
            optimizer.step()#admaw更新训练权重

            epoch_loss += loss.item() * len(batch_idx)

        scheduler.step()#更新学习率
        epoch_loss /= n_samples#平均每个样本的损失

        # 每 50 轮打印一次训练精度
        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()#评估模式
            with torch.no_grad():#关闭梯度计算
                pred = model(features).argmax(dim=1)
                acc = (pred == labels).float().mean().item()
            model.train()
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}  "
                f"loss={epoch_loss:.4f}  train_acc={acc:.2%}"
            )

    return model.eval().cpu()#返回训练完成的模型


#输入训练好的读出模型，特征，标签，返回 0~1 之间的分类准确率
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
    with torch.no_grad():#无梯度推理
        logits = model(features)#特征前向传播得到logits
        preds = logits.argmax(dim=1)
        acc = (preds == labels.to(features.device)).float().mean().item()
    return acc
