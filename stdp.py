import numpy as np


class STDPSynapses:
    """
    STDP (Spike-Timing-Dependent Plasticity) 突触连接层。

    连接前突触层 (n_pre) 和后突触层 (n_post) 之间的一对全连接权重矩阵。
    使用迹 (trace) 机制实现经典的 STDP 规则:

        - 前突触发放后，后突触发放 → LTP (长时程增强，权重增加)
        - 后突触发放后，前突触发放 → LTD (长时程抑制，权重减少)

    权重更新公式:
        post 发放时:  W[i,j] += A_plus * pre_trace[i]   (LTP)
        pre  发放时:  W[i,j] -= A_minus * post_trace[j] (LTD)

    参数:
        n_pre:     前突触神经元数 (输入层 = 784)
        n_post:    后突触神经元数 (兴奋层 = 400)
        w_max:     权重上限，默认 1.0
        A_plus:    LTP 学习率，默认 0.005
        A_minus:   LTD 学习率，默认 0.00525 (略大于 LTP 以维持稳定)
        tau_pre:   前突触迹时间常数 (ms)，默认 20
        tau_post:  后突触迹时间常数 (ms)，默认 20
        dt:        仿真步长 (ms)，默认 1.0
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        w_max: float = 1.0,
        A_plus: float = 0.005,
        A_minus: float = 0.00525,
        tau_pre: float = 20.0,#前突出迹时间常数，用于脉冲迹衰减
        tau_post: float = 20.0,#后突出迹时间常数
        dt: float = 1.0,
    ):
        self.dt = dt
        self.w_max = w_max
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # 权重矩阵 (n_pre × n_post)，初始化为 [0, 0.1) 小随机值
        self.W = np.random.uniform(0.0, 0.1, (n_pre, n_post)).astype(np.float64)

        # 脉冲迹: 追踪最近脉冲时间，指数衰减
        self.pre_trace = np.zeros(n_pre, dtype=np.float64)
        self.post_trace = np.zeros(n_post, dtype=np.float64)

        # 每个后突触神经元的目标输入权重总和，实现神经元竞争wta，防止某一个神经元无限变大
        self.target_sum = w_max * n_pre * 0.1  # = 78.4 (当 n_pre=784)

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """
        执行一个时间步的 STDP 权重更新。

        流程:
            1. 脉冲迹指数衰减
            2. 发放的神经元迹 +1
            3. 若后突触发放，按 LTP 增强对应权重
            4. 若前突触发放，按 LTD 减弱对应权重
            5. 权重裁剪到 [0, w_max]

        参数:
            pre_spikes:  (n_pre,) 布尔数组，前突触脉冲。
            post_spikes: (n_post,) 布尔数组，后突触脉冲。
        """
        # 迹衰减 a[t+1] = a[t] * exp(-dt/tau)
        decay_pre = float(np.exp(-self.dt / self.tau_pre))
        decay_post = float(np.exp(-self.dt / self.tau_post))
        self.pre_trace *= decay_pre
        self.post_trace *= decay_post

        # 发放神经元迹 +1
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

        # LTP: 后突触发放 则 增强来自「近期发放的前突触」的权重
        if np.any(post_spikes):
            # 用外积计算权重增量: (n_pre,) ⊗ (n_post,) → (n_pre, n_post)
            dw_ltp = self.A_plus * np.outer(self.pre_trace, post_spikes.astype(np.float64))
            self.W += dw_ltp

        # LTD: 前突触发放  减弱通往「近期发放的后突触」的权重
        if np.any(pre_spikes):
            dw_ltd = -self.A_minus * np.outer(pre_spikes.astype(np.float64), self.post_trace)
            self.W += dw_ltd

        # 权重裁剪，防止权重过大
        np.clip(self.W, 0.0, self.w_max, out=self.W)

    def normalize(self):
        """
        权重归一化: 将每个后突触神经元的传入权重总和缩放到 target_sum。

        这确保不同神经元获得相等总量的突触输入，
        防止部分神经元累积全部权重而垄断响应。
        """
        col_sums = self.W.sum(axis=0)#按列求和：每一列对应一个后神经元 j，算出该神经元全部输入突触权重总和
        col_sums[col_sums == 0] = 1.0
        scale = self.target_sum / col_sums#计算缩放系数
        self.W *= scale
        np.clip(self.W, 0.0, self.w_max, out=self.W)

    def reset_traces(self):
        """在切换样本时清零脉冲迹，避免上一样本残留影响。"""
        self.pre_trace = np.zeros_like(self.pre_trace)
        self.post_trace = np.zeros_like(self.post_trace)
