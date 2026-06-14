import numpy as np


class STDPSynapses:
    """STDP-modifiable synapses connecting a pre-synaptic to a post-synaptic layer."""

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        w_max: float = 1.0,
        A_plus: float = 0.005,
        A_minus: float = 0.00525,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        dt: float = 1.0,
    ):
        self.W = np.random.uniform(0.0, 0.1, (n_pre, n_post)).astype(np.float64)
        self.w_max = w_max
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.dt = dt

        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)

        # Target sum of incoming weights per post neuron
        self.target_sum = w_max * n_pre * 0.1

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """
        Update synaptic weights for one timestep.

        Args:
            pre_spikes: (n_pre,) boolean array.
            post_spikes: (n_post,) boolean array.
        """
        # Decay traces
        self.pre_trace *= np.exp(-self.dt / self.tau_pre)
        self.post_trace *= np.exp(-self.dt / self.tau_post)

        # Update traces for spiking neurons
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

        # LTP: when post fires, strengthen synapses from recently active pre neurons
        if np.any(post_spikes):
            self.W[:, post_spikes] += self.A_plus * self.pre_trace[:, np.newaxis]

        # LTD: when pre fires, weaken synapses to recently active post neurons
        if np.any(pre_spikes):
            self.W[pre_spikes, :] -= (
                self.A_minus * self.post_trace[np.newaxis, :]
            )

        # Clip weights
        np.clip(self.W, 0.0, self.w_max, out=self.W)

    def normalize(self):
        """Scale each post neuron's incoming weight sum to target_sum."""
        col_sums = self.W.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        scale = self.target_sum / col_sums
        self.W *= scale
        np.clip(self.W, 0.0, self.w_max, out=self.W)
