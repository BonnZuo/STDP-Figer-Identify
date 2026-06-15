import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# MNIST 数据加载
# =============================================================================

def _download_mnist(data_dir: str):
    """
    下载 MNIST 原始 IDX 文件到 data/MNIST/raw/。

    通过 HTTPS 从 AWS 官方镜像下载四个 .gz 文件。
    跳过已存在的文件，避免重复下载。
    """
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
            print(f"下载 {f}...")
            with urllib.request.urlopen(base_url + f, context=ctx) as resp:
                with open(dest, "wb") as out:
                    out.write(resp.read())


def _read_idx(filename: str) -> np.ndarray:
    """
    解析 IDX 格式文件 (MNIST 使用的二进制格式)。

    文件头:
        - 4 bytes magic number (标识数据类型和维度)
        - 4×N bytes 各维度大小
        - 剩余字节为数据

    返回: (n_samples, n_features) uint8 numpy 数组。
    """
    with gzip.open(filename, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        shape = struct.unpack(
            f">{'I' * (magic % 256 - 1)}", f.read(4 * (magic % 256 - 1))
        )
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1)
    return data


class _MNISTPair:
    """
    MNIST 数据集封装，提供类似 torchvision Dataset 的接口。

    属性:
        images: (n, 784) float32 数组，像素值归一化到 [0, 1]。
        labels: (n,) int 数组，类别标签 0-9。
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 返回归一化后的像素和标签
        return self.images[idx].astype(np.float32) / 255.0, int(self.labels[idx])


def load_mnist(data_dir: str = "./data"):
    """
    加载 MNIST 数据集。

    首次调用时自动从网络下载原始文件 (~11MB)，
    后续调用直接读取本地缓存。

    返回:
        train_set: _MNISTPair，60000 训练样本。
        test_set:  _MNISTPair，10000 测试样本。
    """
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    _download_mnist(data_dir)

    train_images = _read_idx(os.path.join(raw_dir, "train-images-idx3-ubyte.gz"))
    train_labels = _read_idx(os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"))
    test_images = _read_idx(os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = _read_idx(os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"))

    train_set = _MNISTPair(train_images, train_labels.flatten())
    test_set = _MNISTPair(test_images, test_labels.flatten())
    return train_set, test_set


# =============================================================================
# 泊松脉冲编码
# =============================================================================

def poisson_encode(
    image: np.ndarray,
    duration: int = 350,
    max_rate: float = 0.2,
    dt: float = 1.0,
) -> np.ndarray:
    """
    将归一化图像转换为泊松脉冲序列。

    每个像素值作为神经元发放率 (Hz): rate = pixel_value * max_rate * 1000。
    每时间步以 rate * dt 的概率独立生成脉冲。

    例如 max_rate=0.2，白色像素 (值=1.0) 在每个 1ms 时间步
    有 20% 概率发放，在 350ms 内约发放 70 次。

    参数:
        image:    (784,) float32 数组，像素值 ∈ [0, 1]。
        duration: 脉冲序列长度 (时间步数)，默认 350。
        max_rate: 最大发放概率/时间步，默认 0.2。
        dt:       时间步长 (ms)，默认 1.0。

    返回:
        (duration, 784) 布尔数组，True 表示该时间步该像素发放脉冲。
    """
    rates = image * max_rate
    spikes = np.random.rand(duration, len(image)) < rates[np.newaxis, :]
    return spikes


# =============================================================================
# 可视化
# =============================================================================

def plot_weights(W: np.ndarray, save_path: str = None):
    """
    将学习到的 STDP 权重可视化为 28×28 感受野。

    每个兴奋神经元对应一个 28×28 的小图，
    展示了它从 784 个输入像素学习的权重模式。
    理想情况下会呈现类似数字笔画的模式。

    参数:
        W:         (784, n_exc) 权重矩阵。
        save_path: 如果提供，保存图片到此路径。
    """
    n_exc = W.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_exc)))  # 排列成正方形网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(n_exc):
        w = W[:, i].reshape(28, 28)
        axes[i].imshow(w, cmap="hot", interpolation="nearest")
        axes[i].axis("off")

    # 隐藏多余的空白子图
    for i in range(n_exc, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def compute_accuracy(
    predictions: np.ndarray, labels: np.ndarray, n_total: int
) -> float:
    """
    计算分类准确率 (无读出层时使用)。

    参数:
        predictions: (n,) int 数组，预测标签。-1 表示无法分类。
        labels:      (n,) int 数组，真实标签。
        n_total:     样本总数。

    返回:
        准确率 float，预测为 -1 的样本视为错误。
    """
    valid = predictions != -1
    correct = np.sum((predictions == labels) & valid)
    return correct / n_total
