import numpy as np


class LIFNeurons:
    """
    LIF (Leaky Integrate-and-Fire) 脉冲神经元层。

    每个神经元维护膜电位 v，随时间按电导驱动的 LIF 动力学演化。
    当 v 达到阈值时发放脉冲，随后重置并进入不应期。
    采用自适应阈值机制防止单个神经元垄断响应。

    动力学方程:
        dv/dt = ((v_rest - v) + g_exc * (E_exc - v)) / tau_v

    参数:
        n:          神经元数量
        v_rest:     静息电位 (mV)，默认 -65
        v_thresh:   初始发放阈值 (mV)，默认 -52
        tau_v:      膜时间常数 (ms)，默认 100
        t_refrac:   不应期时长 (ms)，默认 5
        theta_plus: 每次发放后阈值的增量 (mV)，默认 0.05
        tau_theta:  阈值衰减时间常数 (ms)，默认 1e7 (几乎不衰减)
        dt:         仿真步长 (ms)，默认 1.0
    """

    def __init__(
        self,
        n: int,
        v_rest: float = -65.0,
        v_thresh: float = -52.0,
        tau_v: float = 100.0,
        t_refrac: int = 5,
        theta_plus: float = 0.05,
        tau_theta: float = 1e7,
        dt: float = 1.0,
    ):
        # 基本参数
        self.n = n
        self.dt = dt
        self.v_rest = v_rest
        self.tau_v = tau_v
        self.t_refrac = t_refrac
        self.tau_theta = tau_theta
        self.theta_plus = theta_plus
        self.v_thresh_base = v_thresh

        # 状态变量
        self.v = np.full(n, v_rest, dtype=np.float64)     # 膜电位 (mV)
        self.v_thresh = np.full(n, v_thresh, dtype=np.float64)  # 自适应阈值
        self.refrac = np.zeros(n, dtype=int)               # 不应期剩余计时器

        # STDP 辅助变量
        self.spike_trace = np.zeros(n, dtype=np.float64)   # 脉冲迹 (指数衰减)
        self.tau_trace = 20.0                              # 迹的时间常数 (ms)

    def step(self, conductance: np.ndarray) -> np.ndarray:
        """
        执行一个仿真时间步 (默认 1ms)。

        流程:
            1. 更新不应期计时器，确定当前可兴奋的神经元
            2. 按 LIF 电导模型更新膜电位
            3. 检测脉冲发放
            4. 发放后重置膜电位、启动不应期、提升阈值
            5. 更新脉冲迹 (供 STDP 使用)

        参数:
            conductance: (n,) 数组，每个神经元的兴奋性电导 g_exc。

        返回:
            (n,) 布尔数组，标记哪些神经元在本时间步发放了脉冲。
        """
        # 不应期内神经元不参与更新
        active = self.refrac == 0
        self.refrac = np.maximum(self.refrac - 1, 0)

        # 电导驱动 LIF: E_exc = 0mV，v_rest 和 v 为负值，正电导使膜电位上升
        dv = (self.dt / self.tau_v) * (
            (self.v_rest - self.v) + conductance * (0.0 - self.v)
        )
        self.v[active] += dv[active]

        # 检测发放: 膜电位超过阈值且不在不应期内
        spikes = (self.v >= self.v_thresh) & active

        # 发放后重置
        self.v[spikes] = self.v_rest        # 膜电位复位
        self.refrac[spikes] = self.t_refrac  # 进入不应期

        # 自适应阈值: 发放后阈值升高，避免该神经元频繁独占
        self.v_thresh[spikes] += self.theta_plus
        # 全局阈值以极慢速度向基准值衰减 (tau_theta = 1e7 时几乎不衰减)
        self.v_thresh = self.v_thresh_base + (
            self.v_thresh - self.v_thresh_base
        ) * np.exp(-self.dt / self.tau_theta)

        # 脉冲迹: 指数衰减后，发放的神经元迹+1。STDP 学习中追踪脉冲时间
        self.spike_trace *= np.exp(-self.dt / self.tau_trace)
        self.spike_trace[spikes] += 1.0

        return spikes
