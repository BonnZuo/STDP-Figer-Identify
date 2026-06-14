import torch
import torch.nn as nn


class LinearReadout(nn.Module):
    """Linear readout layer: maps STDP firing-rate features to class logits."""

    def __init__(self, n_features: int = 400, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPReadout(nn.Module):
    """2-layer MLP readout: 400 → 256 → ReLU → Dropout → 10."""

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
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Default: use MLP for best accuracy
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
    Train a readout layer on extracted STDP features.

    Args:
        features: (n_samples, n_features) firing rate vectors.
        labels: (n_samples,) integer class labels.
        n_epochs: number of training epochs.
        lr: learning rate for AdamW.
        weight_decay: L2 regularization strength.
        batch_size: mini-batch size.
        device: torch device.
        use_mlp: if True, use 2-layer MLP; else linear readout.

    Returns:
        Trained model (MLPReadout or LinearReadout).
    """
    n_features = features.shape[1]
    if use_mlp:
        model = MLPReadout(n_features=n_features).to(device)
    else:
        model = LinearReadout(n_features=n_features).to(device)

    features = features.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()

    n_samples = features.shape[0]
    indices = torch.randperm(n_samples)

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
    """Evaluate readout accuracy on given features."""
    model = model.to(features.device)
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        acc = (preds == labels.to(features.device)).float().mean().item()
    return acc
