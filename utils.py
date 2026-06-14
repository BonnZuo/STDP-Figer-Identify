import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _download_mnist(data_dir: str):
    """Download MNIST raw files if they don't exist."""
    import ssl
    import urllib.request

    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for f in files:
        dest = os.path.join(raw_dir, f)
        if not os.path.exists(dest):
            print(f"Downloading {f}...")
            with urllib.request.urlopen(base_url + f, context=ctx) as resp:
                with open(dest, "wb") as out:
                    out.write(resp.read())


def _read_idx(filename: str):
    """Read IDX file format (used by MNIST)."""
    with gzip.open(filename, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        shape = struct.unpack(f">{'I' * (magic % 256 - 1)}", f.read(4 * (magic % 256 - 1)))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1)
    return data


class _MNISTPair:
    """A simple MNIST dataset pair that mimics torchvision Dataset interface."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].astype(np.float32) / 255.0, int(self.labels[idx])


def load_mnist(data_dir: str = "./data"):
    """Load MNIST dataset. Downloads raw files on first call, then parses them."""
    raw_dir = os.path.join(data_dir, "MNIST", "raw")

    # Download if needed
    _download_mnist(data_dir)

    train_images = _read_idx(os.path.join(raw_dir, "train-images-idx3-ubyte.gz"))
    train_labels = _read_idx(os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"))
    test_images = _read_idx(os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = _read_idx(os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"))

    train_set = _MNISTPair(train_images, train_labels.flatten())
    test_set = _MNISTPair(test_images, test_labels.flatten())

    return train_set, test_set


def poisson_encode(
    image: np.ndarray, duration: int = 350, max_rate: float = 0.2, dt: float = 1.0
) -> np.ndarray:
    """
    Convert a normalized image to Poisson spike trains.

    Args:
        image: (784,) float array, values in [0, 1] (pixels scaled).
        duration: number of timesteps.
        max_rate: maximum firing rate per timestep for value=1.0.
        dt: timestep in ms.

    Returns:
        (duration, 784) boolean array.
    """
    rates = image * max_rate
    spikes = np.random.rand(duration, len(image)) < rates[np.newaxis, :]
    return spikes


def plot_weights(W: np.ndarray, save_path: str = None):
    """
    Visualize learned synaptic weights as 28x28 receptive fields.

    Args:
        W: (784, n_exc) weight matrix.
        save_path: if provided, save figure to this path.
    """
    n_exc = W.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_exc)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(n_exc):
        w = W[:, i].reshape(28, 28)
        axes[i].imshow(w, cmap="hot", interpolation="nearest")
        axes[i].axis("off")

    for i in range(n_exc, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def compute_accuracy(
    predictions: np.ndarray, labels: np.ndarray, n_total: int
) -> float:
    """Compute classification accuracy, treating -1 predictions as wrong."""
    valid = predictions != -1
    correct = np.sum((predictions == labels) & valid)
    return correct / n_total
