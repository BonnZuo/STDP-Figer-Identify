import numpy as np
from neuron import LIFNeurons
from stdp import STDPSynapses


class SNN:
    """
    基于 STDP 学习的脉冲神经网络 (SNN)。

    网络结构:
        输入层 (784 泊松脉冲) ──STDP突触──→ 兴奋层 (LIF 神经元 + WTA 抑制)

    前向流程 (每个时间步):
        1. 输入脉冲通过权重矩阵产生兴奋性电导
        2. LIF 神经元更新膜电位并检测发放
        3. STDP 在线更新权重 (训练模式)
        4. Winner-Take-All 横向抑制

    参数:
        n_input:    输入神经元数 (MNIST: 28×28=784)
        n_exc:      兴奋层神经元数，默认 400
        v_rest:     静息电位 (mV)，默认 -65
        v_thresh:   初始发放阈值 (mV)，默认 -52
        tau_v:      膜时间常数 (ms)，默认 100
        t_refrac:   不应期 (ms)，默认 5
        theta_plus: 自适应阈值增量，默认 0.05
        w_max:      STDP 权重上限，默认 1.0
        A_plus:     LTP 学习率，默认 0.005
        A_minus:    LTD 学习率，默认 0.00525
    """

    def __init__(
        self,
        n_input: int = 784,
        n_exc: int = 400,
        #LIF超参
        v_rest: float = -65.0,
        v_thresh: float = -52.0,
        tau_v: float = 100.0,
        t_refrac: int = 5,
        theta_plus: float = 0.05,
        #stdp超参
        w_max: float = 1.0,
        A_plus: float = 0.005,
        A_minus: float = 0.00525,
    ):
        self.n_input = n_input
        self.n_exc = n_exc#保存网络维度，记录输入、隐藏神经元数量，后续分配数组、矩阵尺寸使用

        # 实例化兴奋层 LIF 神经元
        self.exc_neurons = LIFNeurons(
            n=n_exc,
            v_rest=v_rest,
            v_thresh=v_thresh,
            tau_v=tau_v,
            t_refrac=t_refrac,
            theta_plus=theta_plus,
        )#创建 400 个自适应阈值 LIF 神经元，内部维护：膜电位v、不应期计时器refrac、自适应阈值v_thresh、突触后脉冲迹spike_trace

        # 实例化STDP 突触连接，维护全连接权重
        self.synapses = STDPSynapses(
            n_pre=n_input,
            n_post=n_exc,
            w_max=w_max,
            A_plus=A_plus,
            A_minus=A_minus,
        )

        # 训练样本计数器，用于周期性权重归一化
        self.sample_count = 0

    #单张样本的完整时序仿真
    def run(
        self,
        input_spikes: np.ndarray,#泊松编码生成脉冲序列
        train: bool = True,#开启train，有权重更新
    ) -> np.ndarray:
        """
        在输入脉冲序列上运行网络。

        参数:
            input_spikes: (duration, n_input) 布尔数组，泊松编码的输入脉冲。
            train:        是否执行 STDP 学习 (推理时设为 False)。

        返回:
            (duration, n_exc) 布尔数组，兴奋层脉冲发放记录。
        """
        duration = input_spikes.shape[0]
        spike_record = np.zeros((duration, self.n_exc), dtype=bool)

        #外层循环，一个时间步仿真
        for t in range(duration):
            inp = input_spikes[t].astype(np.float64)#当前时刻 784 维输入脉冲 0/1 向量

            # 计算兴奋性电导: g_exc[j] = Σ_i inp[i] * W[i, j]
            g_exc = inp @ self.synapses.W

            # 一步 LIF 神经元更新，更新膜电位、判断是否发放脉冲、更新后脉冲迹，返回当前时刻 400 神经元发放标记。
            exc_spikes = self.exc_neurons.step(g_exc)
            spike_record[t] = exc_spikes

            # 训练模式下执行 STDP 权重更新，把当前输入脉冲、后层脉冲传入 STDP 突触，执行迹衰减、LTP 权重增加、LTD 权重减小、权重裁剪；推理阶段跳过此步，权重完全不变
            if train:
                self.synapses.step(inp.astype(bool), exc_spikes)

            # WTA 横向抑制 任一神经元发放时，将其余神经元膜电位复位
            # 确保不同神经元学习不同的输入模式
            if np.any(exc_spikes):
                self.exc_neurons.v[~exc_spikes] = self.exc_neurons.v_rest

        # 每 100 个样本归一化一次权重，防止部分神经元垄断
        if train:
            self.sample_count += 1
            if self.sample_count % 100 == 0:
                self.synapses.normalize()

        return spike_record#返回全程时序脉冲矩阵，后续统计发放率特征

    #封装推理流程，直接输出单样本分类特征，供extract_features_numpy批量调用
    def extract_firing_rates(
        self, input_spikes: np.ndarray
    ) -> np.ndarray:
        """
        提取 firing rate 特征向量 (供读出层使用)。

        参数:
            input_spikes: (duration, n_input) 泊松脉冲。

        返回:
            (n_exc,) float32 数组，各兴奋神经元的脉冲发放次数。
        """
        spike_record = self.run(input_spikes, train=False)
        return spike_record.sum(axis=0).astype(np.float32)

    def reset_state(self):
        """切换样本时重置所有神经元状态和突触迹。"""
        self.exc_neurons.v = np.full(#膜电位
            self.n_exc, self.exc_neurons.v_rest, dtype=np.float64
        )
        #清空不应期
        self.exc_neurons.refrac = np.zeros(self.n_exc, dtype=int)
        #自适应阈值初始化
        self.exc_neurons.v_thresh = np.full(
            self.n_exc, self.exc_neurons.v_thresh_base, dtype=np.float64
        )
        #清空神经元后脉冲迹
        self.exc_neurons.spike_trace = np.zeros(self.n_exc, dtype=np.float64)
        #stdp清空前后脉冲迹
        self.synapses.reset_traces()
