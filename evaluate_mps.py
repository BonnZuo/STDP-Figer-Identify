import sys
import numpy as np
import torch
from tqdm import tqdm

from network import SNN
from readout import MLPReadout, evaluate_readout
from utils import load_mnist, poisson_encode, plot_weights


def evaluate(
    weights_path: str = "outputs/weights_full.npy",
    readout_path: str = "outputs/readout_full.pth",
    n_test: int = 10000,
    duration: int = 350,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading MNIST...")
    _, test_set = load_mnist()

    print(f"Loading weights from {weights_path}...")
    W = np.load(weights_path)
    print(f"Loading readout from {readout_path}...")
    n_exc = W.shape[1]
    readout = MLPReadout(n_features=n_exc)
    readout.load_state_dict(torch.load(readout_path, map_location="cpu"))
    readout.eval()

    net = SNN(n_exc=n_exc)
    net.synapses.W = W.copy()

    print(f"\nEvaluating on {n_test} test samples ({duration}ms each)...")
    indices = np.random.choice(len(test_set), n_test, replace=False)

    features_list = []
    labels_list = []

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        img, label = test_set[idx]
        labels_list.append(label)

        spikes = poisson_encode(img, duration=duration)
        net.reset_state()
        record = net.run(spikes, train=False)

        spike_counts = record.sum(axis=0).astype(np.float32)
        features_list.append(torch.from_numpy(spike_counts))

    features = torch.stack(features_list)
    labels = torch.tensor(labels_list, dtype=torch.long)

    readout_acc = evaluate_readout(readout, features, labels)

    print(f"\nResults on {n_test} test samples:")
    print(f"  STDP + Readout accuracy: {readout_acc:.2%}")

    with torch.no_grad():
        logits = readout(features)
        preds = logits.argmax(dim=1)

    print("\nClass-wise accuracy:")
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  Digit {c}: {class_acc:.1%}")

    print("\nGenerating weight visualization...")
    plot_weights(W, save_path="outputs/weights_full_visualization.png")
    print("Saved to outputs/weights_full_visualization.png")


if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/weights_full.npy"
    readout_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/readout_full.pth"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    evaluate(weights_path, readout_path, n_test)
