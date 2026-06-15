"""
STDP 手写数字识别 — 评估脚本。

加载已训练的 STDP 权重和读出层模型，在 MNIST 测试集上评估分类准确率。

用法:
    python3 evaluate.py                                     # 使用默认路径
    python3 evaluate.py outputs/weights_full.npy outputs/readout_full.pth
"""

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
    """
    加载模型并在测试集上评估。

    流程:
        1. 加载 STDP 权重矩阵 (784×400)
        2. 加载 MLP 读出层参数
        3. 对每个测试样本: 泊松编码 → SNN 推理 → 提取特征 → 读出层分类
        4. 输出总体及各类别准确率
        5. 生成权重可视化图

    参数:
        weights_path: STDP 权重 .npy 文件路径。
        readout_path: 读出层 .pth 文件路径。
        n_test:       测试样本数，默认 10000。
        duration:     每样本模拟时长 (ms)，默认 350。
        seed:         随机种子。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("加载 MNIST 数据集...")
    _, test_set = load_mnist()

    # 加载模型
    print(f"加载 STDP 权重: {weights_path}")
    W = np.load(weights_path)
    n_exc = W.shape[1]

    print(f"加载读出层: {readout_path}")
    readout = MLPReadout(n_features=n_exc)
    readout.load_state_dict(torch.load(readout_path, map_location="cpu"))
    readout.eval()

    # 初始化 SNN 网络并加载权重
    net = SNN(n_exc=n_exc)
    net.synapses.W = W.copy()

    # 评估
    print(f"\n评估 {n_test} 个测试样本 ({duration}ms/样本)...")
    indices = np.random.choice(len(test_set), n_test, replace=False)

    features_list = []
    labels_list = []

    for i, idx in enumerate(tqdm(indices, desc="推理中")):
        img, label = test_set[idx]
        labels_list.append(label)

        # 泊松编码 → 网络推理
        spikes = poisson_encode(img, duration=duration)
        net.reset_state()
        record = net.run(spikes, train=False)

        # 统计各神经元的脉冲发放次数作为特征
        spike_counts = record.sum(axis=0).astype(np.float32)
        features_list.append(torch.from_numpy(spike_counts))

    features = torch.stack(features_list)
    labels = torch.tensor(labels_list, dtype=torch.long)

    # 读出层预测
    readout_acc = evaluate_readout(readout, features, labels)
    print(f"\n测试准确率 (STDP + Readout): {readout_acc:.2%}")

    # 各类别准确率
    with torch.no_grad():
        logits = readout(features)
        preds = logits.argmax(dim=1)

    print("\n各类别准确率:")
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            class_acc = (preds[mask] == c).float().mean().item()
            print(f"  数字 {c}: {class_acc:.1%}")

    # 权重可视化
    print("\n生成权重可视化...")
    plot_weights(W, save_path="outputs/weights_full_visualization.png")
    print("已保存到 outputs/weights_full_visualization.png")


if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/weights_full.npy"
    readout_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/readout_full.pth"
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    evaluate(weights_path, readout_path, n_test)
