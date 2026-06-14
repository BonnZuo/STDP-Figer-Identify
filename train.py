import sys
import numpy as np
from tqdm import tqdm
from network import SNN
from utils import load_mnist, poisson_encode


def assign_labels(net: SNN, train_set, n_samples: int = 10000) -> np.ndarray:
    """
    Assign a digit label to each excitatory neuron based on its firing preference.

    Returns:
        (n_exc,) int array. -1 means unassigned.
    """
    spike_counts = np.zeros((net.n_exc, 10))
    indices = np.random.choice(len(train_set), n_samples, replace=False)

    for idx in tqdm(indices, desc="Assigning labels"):
        img, label = train_set[idx]
        img_spikes = poisson_encode(img, duration=350)

        net.reset_state()
        record = net.run(img_spikes, train=False)
        spike_counts[:, label] += record.sum(axis=0)

    assigned = np.argmax(spike_counts, axis=1)
    total = spike_counts.sum(axis=1)
    assigned[total == 0] = -1

    print(f"Neurons assigned: {(assigned != -1).sum()} / {net.n_exc}")
    return assigned


def train(
    n_exc: int = 400,
    n_train: int = 10000,
    duration: int = 350,
    seed: int = 42,
):
    np.random.seed(seed)

    print("Loading MNIST...")
    train_set, test_set = load_mnist()

    print(f"Initializing SNN with {n_exc} excitatory neurons...")
    net = SNN(n_exc=n_exc)

    indices = np.random.choice(len(train_set), n_train, replace=False)

    print(f"Training on {n_train} samples ({duration}ms each)...")
    for idx in tqdm(indices, desc="Training"):
        img, _ = train_set[idx]
        img_spikes = poisson_encode(img, duration=duration)

        net.reset_state()
        net.run(img_spikes, train=True)

    print("Training complete. Saving weights...")
    np.save("outputs/weights.npy", net.synapses.W)
    print(f"Weights saved to outputs/weights.npy (shape: {net.synapses.W.shape})")

    print("Assigning labels to neurons...")
    labels = assign_labels(net, train_set, n_samples=min(n_train, 10000))
    np.save("outputs/neuron_labels.npy", labels)
    print(f"Labels saved to outputs/neuron_labels.npy")

    # Quick evaluation on a subset of test data
    print("\nQuick evaluation on test set (1000 samples)...")
    test_indices = np.random.choice(len(test_set), 1000, replace=False)
    correct = 0
    total = 0

    for idx in tqdm(test_indices, desc="Evaluating"):
        img, label = test_set[idx]
        img_spikes = poisson_encode(img, duration=duration)

        net.reset_state()
        record = net.run(img_spikes, train=False)

        spike_counts = record.sum(axis=0)
        winner = np.argmax(spike_counts)

        if spike_counts[winner] > 0 and labels[winner] != -1:
            if labels[winner] == label:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")

    return net


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    n_exc = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    train(n_exc=n_exc, n_train=n_train)
