"""
STDP 手写数字识别 — 完整训练流程。

三阶段训练:
    Phase 1 — STDP 无监督学习: 脉冲神经网络通过 STDP 规则学习输入特征
    Phase 2 — 特征提取:        提取兴奋层 firing rate 作为特征向量
    Phase 3 — 读出层训练:      在特征上训练 MLP 分类器

用法:
    python3 train_full.py [训练样本数] [兴奋神经元数]
    python3 train_full.py 60000 400    # 默认
"""

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
    """
    Phase 1: 无监督 STDP 训练。

    对 n_train 个样本逐一:
        1. 泊松编码图像 → 脉冲序列
        2. 网络运行 duration 时间步
        3. STDP 在线更新权重 + WTA 竞争

    参数:
        net:       初始化的 SNN 网络。
        train_set: MNIST 训练集。
        n_train:   训练样本数。
        duration:  每样本的模拟时长 (ms)。

    返回:
        训练后的 SNN 网络。
    """
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
    """
    Phase 2: 从已训练的 SNN 中提取 firing rate 特征。

    对每个样本:
        1. 关闭 STDP (train=False)，运行网络
        2. 统计各兴奋神经元的脉冲发放次数
        3. 以 400 维向量作为该样本的特征表示

    参数:
        net:      已训练的 SNN。
        dataset:  MNIST 数据集。
        indices:  要提取的样本索引。
        duration: 每样本模拟时长 (ms)。
        desc:     进度条描述文字。

    返回:
        (features, labels) — PyTorch 张量，供读出层训练使用。
    """
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
    """
    执行完整三阶段训练流程。

    参数:
        n_train_stdp: STDP 训练样本数，默认 60000。
        n_exc:        兴奋层神经元数，默认 400。
        duration:     每样本模拟时长 (ms)，默认 350。
        seed:         随机种子。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("加载 MNIST 数据集...")
    train_set, test_set = load_mnist()

    # =========================================================================
    # Phase 1: STDP 无监督训练
    # =========================================================================
    print(f"\n{'='*50}")
    print(f"Phase 1: STDP 无监督训练 ({n_train_stdp} 样本, {duration}ms, {n_exc} 神经元)")
    print(f"{'='*50}")

    net = SNN(n_exc=n_exc)
    t0 = time.time()
    net = train_stdp(net, train_set, n_train=n_train_stdp, duration=duration)
    t1 = time.time()
    print(f"训练耗时: {t1-t0:.0f}s ({n_train_stdp/(t1-t0):.0f} 样本/秒)")

    # 保存 STDP 权重
    W = net.synapses.W.copy()
    np.save("outputs/weights_full.npy", W)
    print(f"权重已保存到 outputs/weights_full.npy ({W.shape})")

    # =========================================================================
    # Phase 2: 特征提取
    # =========================================================================
    print(f"\n{'='*50}")
    print("Phase 2: 提取 firing rate 特征...")
    print(f"{'='*50}")

    n_features_train = min(n_train_stdp, len(train_set))
    train_indices = np.random.choice(len(train_set), n_features_train, replace=False)

    t0 = time.time()
    features_train, labels_train = extract_features_numpy(
        net, train_set, train_indices, duration=duration, desc="训练集特征"
    )
    t1 = time.time()
    print(f"训练集特征提取: {t1-t0:.0f}s ({n_features_train} 样本)")

    # 提取全部 10000 测试样本的特征
    test_indices = np.arange(len(test_set))
    t0 = time.time()
    features_test, labels_test = extract_features_numpy(
        net, test_set, test_indices, duration=duration, desc="测试集特征"
    )
    t1 = time.time()
    print(f"测试集特征提取: {t1-t0:.0f}s ({len(test_set)} 样本)")

    # 缓存特征文件，方便后续调试和重训练读出层
    torch.save(
        {"features": features_train, "labels": labels_train},
        "outputs/features_train.pt",
    )
    torch.save(
        {"features": features_test, "labels": labels_test},
        "outputs/features_test.pt",
    )
    print(
        f"特征已缓存: 训练集 {features_train.shape}, 测试集 {features_test.shape}"
    )

    # =========================================================================
    # Phase 3: 读出层训练
    # =========================================================================
    print(f"\n{'='*50}")
    print("Phase 3: 训练读出层 (MLP: 400 → 256 → ReLU → Dropout → 10)")
    print(f"{'='*50}")

    readout = train_readout(
        features_train,
        labels_train,
        n_epochs=300,
        lr=0.001,
        weight_decay=1e-4,
        use_mlp=True,
    )
    torch.save(readout.state_dict(), "outputs/readout_full.pth")
    print("读出层已保存到 outputs/readout_full.pth")

    # =========================================================================
    # 最终评估
    # =========================================================================
    print(f"\n{'='*50}")
    print("最终评估")
    print(f"{'='*50}")

    train_acc = evaluate_readout(readout, features_train, labels_train)
    test_acc = evaluate_readout(readout, features_test, labels_test)

    print(f"\n结果:")
    print(f"  训练集准确率: {train_acc:.2%}")
    print(f"  测试集准确率: {test_acc:.2%}")

    # 各类别准确率
    with torch.no_grad():
        logits = readout(features_test)
        preds = logits.argmax(dim=1)

    print("\n测试集各类别准确率:")
    for c in range(10):
        mask = labels_test == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  数字 {c}: {class_acc:.1%}")

    return net, readout


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    n_exc = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    train(n_train_stdp=n_train, n_exc=n_exc)
