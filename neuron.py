import numpy as np


class LIFNeurons:
    """Layer of Leaky Integrate-and-Fire neurons with adaptive threshold."""

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
        self.n = n
        self.v = np.full(n, v_rest)
        self.v_rest = v_rest
        self.v_thresh_base = v_thresh
        self.v_thresh = np.full(n, v_thresh)
        self.tau_v = tau_v
        self.t_refrac = t_refrac
        self.refrac = np.zeros(n, dtype=int)
        self.theta_plus = theta_plus
        self.tau_theta = tau_theta
        self.dt = dt

        self.spike_trace = np.zeros(n)
        self.tau_trace = 20.0

    def step(self, conductance: np.ndarray) -> np.ndarray:
        """
        One simulation timestep.

        Args:
            conductance: (n,) array of excitatory conductance for each neuron.

        Returns:
            (n,) boolean array of spikes.
        """
        active = self.refrac == 0
        self.refrac = np.maximum(self.refrac - 1, 0)

        # Conductance-based LIF: dv/dt = ((v_rest - v) + g * (E_exc - v)) / tau
        dv = (self.dt / self.tau_v) * (
            (self.v_rest - self.v) + conductance * (0.0 - self.v)
        )
        self.v[active] += dv[active]

        # Spike detection
        spikes = (self.v >= self.v_thresh) & active

        # Reset
        self.v[spikes] = self.v_rest
        self.refrac[spikes] = self.t_refrac

        # Adaptive threshold
        self.v_thresh[spikes] += self.theta_plus
        self.v_thresh = self.v_thresh_base + (
            self.v_thresh - self.v_thresh_base
        ) * np.exp(-self.dt / self.tau_theta)

        # Spike trace for STDP
        self.spike_trace *= np.exp(-self.dt / self.tau_trace)
        self.spike_trace[spikes] += 1.0

        return spikes
