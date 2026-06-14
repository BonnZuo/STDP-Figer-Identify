"""Full training pipeline: NumPy STDP + PyTorch readout for >90% accuracy."""
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

from network import SNN
from readout import Readout, train_readout, evaluate_readout
from utils import load_mnist, poisson_encode


def train_stdp(
    net: SNN,
    train_set,
    n_train: int,
    duration: int,
) -> SNN:
    """Phase 1: unsupervised STDP training (NumPy, CPU-optimized)."""
    indices = np.random.choice(len(train_set), n_train, replace=False)

    for idx in tqdm(indices, desc="Phase 1: STDP"):
        img, _ = train_set[idx]
        spikes = poisson_encode(img, duration=duration)
        net.reset_state()
        net.run(spikes, train=True)

    return net


def extract_features_numpy(
    net: SNN,
    dataset,
    indices: np.ndarray,
    duration: int,
    desc: str = "Extracting",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract STDP firing-rate features using NumPy network."""
    features_list = []
    labels_list = []

    for idx in tqdm(indices, desc=desc):
        img, label = dataset[idx]
        spikes = poisson_encode(img, duration=duration)
        net.reset_state()
        record = net.run(spikes, train=False)
        firing_rates = record.sum(axis=0).astype(np.float32)
        features_list.append(torch.from_numpy(firing_rates))
        labels_list.append(label)

    features = torch.stack(features_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels


def train(
    n_train_stdp: int = 60000,
    n_exc: int = 400,
    duration: int = 350,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading MNIST...")
    train_set, test_set = load_mnist()

    # ── Phase 1: STDP unsupervised training ──
    print(f"\n{'='*50}")
    print(f"Phase 1: STDP training ({n_train_stdp} samples, {duration}ms, {n_exc} neurons)")
    print(f"{'='*50}")

    net = SNN(n_exc=n_exc)
    t0 = time.time()
    net = train_stdp(net, train_set, n_train=n_train_stdp, duration=duration)
    t1 = time.time()
    print(f"STDP training time: {t1-t0:.1f}s ({n_train_stdp/(t1-t0):.1f} samples/s)")

    W = net.synapses.W.copy()
    np.save("outputs/weights_full.npy", W)
    print(f"Weights saved to outputs/weights_full.npy ({W.shape})")

    # ── Phase 2: Feature extraction ──
    print(f"\n{'='*50}")
    print("Phase 2: Extracting features for readout...")
    print(f"{'='*50}")

    # Use all training data for readout
    n_features_train = min(n_train_stdp, len(train_set))
    train_indices = np.random.choice(len(train_set), n_features_train, replace=False)

    t0 = time.time()
    features_train, labels_train = extract_features_numpy(
        net, train_set, train_indices, duration=duration, desc="Train features"
    )
    t1 = time.time()
    print(f"Feature extraction (train): {t1-t0:.1f}s for {n_features_train} samples")

    # Use all test data for evaluation
    test_indices = np.arange(len(test_set))
    t0 = time.time()
    features_test, labels_test = extract_features_numpy(
        net, test_set, test_indices, duration=duration, desc="Test features"
    )
    t1 = time.time()
    print(f"Feature extraction (test): {t1-t0:.1f}s for {len(test_set)} samples")

    torch.save({"features": features_train, "labels": labels_train}, "outputs/features_train.pt")
    torch.save({"features": features_test, "labels": labels_test}, "outputs/features_test.pt")
    print(f"Features saved. Train: {features_train.shape}, Test: {features_test.shape}")

    # ── Phase 3: Readout training ──
    print(f"\n{'='*50}")
    print("Phase 3: Training readout layer (400 → 10 softmax)")
    print(f"{'='*50}")

    readout = train_readout(
        features_train, labels_train,
        n_epochs=200, lr=0.001, weight_decay=1e-4,
    )
    torch.save(readout.state_dict(), "outputs/readout_full.pth")
    print("Readout saved to outputs/readout_full.pth")

    # ── Evaluation ──
    print(f"\n{'='*50}")
    print("Evaluation")
    print(f"{'='*50}")

    train_acc = evaluate_readout(readout, features_train, labels_train)
    test_acc = evaluate_readout(readout, features_test, labels_test)

    print(f"\nFinal Results:")
    print(f"  Train accuracy (readout): {train_acc:.2%}")
    print(f"  Test accuracy  (readout): {test_acc:.2%}")

    # Class-wise accuracy
    with torch.no_grad():
        logits = readout(features_test)
        preds = logits.argmax(dim=1)

    print("\nClass-wise test accuracy:")
    for c in range(10):
        mask = labels_test == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  Digit {c}: {class_acc:.1%}")

    return net, readout


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    n_exc = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    train(n_train_stdp=n_train, n_exc=n_exc)
