import sys
import numpy as np
from tqdm import tqdm
from network import SNN
from utils import load_mnist, poisson_encode, plot_weights, compute_accuracy


def evaluate(
    weights_path: str = "outputs/weights.npy",
    labels_path: str = "neuron_labels.npy",
    n_test: int = 10000,
    duration: int = 350,
    seed: int = 42,
):
    np.random.seed(seed)

    print("Loading MNIST...")
    _, test_set = load_mnist()

    print(f"Loading weights from {weights_path}...")
    W = np.load(weights_path)
    neuron_labels = np.load(labels_path)

    net = SNN(n_exc=W.shape[1])
    net.synapses.W = W.copy()

    print(f"Evaluating on {n_test} test samples ({duration}ms each)...")
    indices = np.random.choice(len(test_set), n_test, replace=False)

    predictions = np.zeros(n_test, dtype=int)
    ground_truth = np.zeros(n_test, dtype=int)

    for i, idx in enumerate(tqdm(indices, desc="Evaluating")):
        img, label = test_set[idx]
        ground_truth[i] = label

        img_spikes = poisson_encode(img, duration=duration)
        net.reset_state()
        record = net.run(img_spikes, train=False)

        spike_counts = record.sum(axis=0)
        winner = np.argmax(spike_counts)

        if spike_counts[winner] > 0 and neuron_labels[winner] != -1:
            predictions[i] = neuron_labels[winner]
        else:
            predictions[i] = -1

    acc = compute_accuracy(predictions, ground_truth, n_test)
    valid_mask = predictions != -1
    classified = valid_mask.sum()
    correct = np.sum((predictions == ground_truth) & valid_mask)

    print(f"\nResults on {n_test} test samples:")
    print(f"  Classified: {classified}/{n_test} ({classified/n_test:.1%})")
    print(f"  Correct: {correct}/{n_test} ({correct/n_test:.1%})")
    print(f"  Accuracy (of classified): {correct/classified:.2%}" if classified > 0 else "")

    # Class-wise accuracy
    print("\nClass-wise accuracy:")
    for c in range(10):
        class_mask = ground_truth == c
        class_total = class_mask.sum()
        class_correct = np.sum((predictions == c) & class_mask)
        print(f"  Digit {c}: {class_correct}/{class_total} ({class_correct/class_total:.1%})")

    # Visualize learned weights
    print("\nGenerating weight visualization...")
    plot_weights(W, save_path="weights_visualization.png")
    print("Saved to weights_visualization.png")

    return predictions, ground_truth


if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/weights.npy"
    labels_path = sys.argv[2] if len(sys.argv) > 2 else "neuron_labels.npy"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    evaluate(weights_path, labels_path, n_test)
