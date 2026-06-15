# STDP 手写数字识别

基于脉冲时序依赖可塑性 (Spike-Timing-Dependent Plasticity) 的无监督特征学习 + 监督读出层的 MNIST 手写数字识别。**测试准确率 94.37%。**

## 架构

```
784 输入像素 → 泊松编码 → STDP (784×400) → 400 LIF 神经元 + WTA 抑制
                                              ↓
                                        Firing Rates (400维)
                                              ↓
                                    MLP: 400 → 256 → ReLU → Dropout → 10
```

- **输入编码**：像素值 → 泊松脉冲序列 (max rate = 200 Hz)
- **神经元模型**：电导型 Leaky Integrate-and-Fire (LIF)，自适应阈值，不应期 5ms
- **突触可塑性**：经典 STDP 规则 (pre-before-post → LTP, post-before-pre → LTD)
- **竞争机制**：全局 Winner-Take-All 横向抑制
- **读出层**：2层 MLP，将 STDP 特征映射为类别概率

## 文件结构

```
├── neuron.py          # LIF 脉冲神经元模型 (NumPy)
├── stdp.py            # STDP 学习规则 + 权重归一化 (NumPy)
├── network.py         # 完整 SNN 网络 + 训练/推理 (NumPy)
├── readout.py         # Linear / MLP 读出层 (PyTorch)
├── utils.py           # MNIST 加载、泊松编码、权重可视化
├── train_full.py      # 三阶段训练主脚本 (STDP + 特征提取 + 读出层)
├── evaluate.py    # 评估脚本 (STDP + Readout)
├── requirements.txt   # Python 依赖
├── .gitignore
├── data/              # MNIST 原始数据 (自动下载)
└── outputs/           # 训练产出 (权重 / 特征 / 模型 / 可视化)
```

## 快速开始

### 安装依赖

```bash
pip3 install -r requirements.txt
```

### 训练

```bash
# 完整训练 (60k 样本 STDP + 特征提取 + 读出层)
python3 train_full.py 60000 400
```

三阶段训练流程：
1. **Phase 1 - STDP 无监督训练**：60k 样本 × 350ms，约 14 分钟
2. **Phase 2 - 特征提取**：对全量训练/测试集提取 firing rate 特征，约 7 分钟
3. **Phase 3 - 读出层训练**：MLP 监督训练 300 epochs，约 10 秒

训练产出（均在 `outputs/` 目录下）：
- `weights_full.npy` — STDP 权重矩阵 (784×400)
- `features_train.pt` / `features_test.pt` — 提取的特征
- `readout_full.pth` — MLP 读出层参数
- `weights_full_visualization.png` — 权重可视化

### 评估

```bash
python3 evaluate.py
```

## 结果

| 指标 | 值 |
|------|-----|
| 测试准确率 | **94.37%** |
| 训练准确率 | 100.00% |

| 类别 | 准确率 |
|------|--------|
| 0 | 97.3% |
| 1 | 98.7% |
| 2 | 94.0% |
| 3 | 94.2% |
| 4 | 94.7% |
| 5 | 93.4% |
| 6 | 95.7% |
| 7 | 92.6% |
| 8 | 92.7% |
| 9 | 89.9% |

## 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 兴奋神经元数 | 400 | |
| 模拟时长 | 350ms | 每样本 |
| 时间步长 dt | 1ms | |
| V_rest | -65mV | 静息电位 |
| V_thresh | -52mV | 发放阈值 |
| tau_v | 100ms | 膜时间常数 |
| A_plus | 0.005 | LTP 学习率 |
| A_minus | 0.00525 | LTD 学习率 |
| w_max | 1.0 | 权重上限 |
| 不应期 | 5ms | |
| 泊松 max_rate | 0.2/ms | 200 Hz |

## 参考文献

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
