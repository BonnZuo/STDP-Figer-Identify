import numpy as np
from neuron import LIFNeurons
from stdp import STDPSynapses


class SNN:
    """Spiking neural network with STDP learning for digit recognition."""

    def __init__(
        self,
        n_input: int = 784,
        n_exc: int = 400,
        v_rest: float = -65.0,
        v_thresh: float = -52.0,
        tau_v: float = 100.0,
        t_refrac: int = 5,
        theta_plus: float = 0.05,
        w_max: float = 1.0,
        A_plus: float = 0.005,
        A_minus: float = 0.00525,
    ):
        self.n_input = n_input
        self.n_exc = n_exc

        self.exc_neurons = LIFNeurons(
            n=n_exc,
            v_rest=v_rest,
            v_thresh=v_thresh,
            tau_v=tau_v,
            t_refrac=t_refrac,
            theta_plus=theta_plus,
        )

        self.synapses = STDPSynapses(
            n_pre=n_input,
            n_post=n_exc,
            w_max=w_max,
            A_plus=A_plus,
            A_minus=A_minus,
        )

        self.input_trace = np.zeros(n_input)
        self.tau_input_trace = 20.0
        self.dt = 1.0
        self.sample_count = 0

    def run(
        self,
        input_spikes: np.ndarray,
        train: bool = True,
    ) -> np.ndarray:
        """
        Run the network on a batch of input spikes.

        Args:
            input_spikes: (duration, n_input) boolean array of input spike trains.
            train: if True, apply STDP learning and weight normalization.

        Returns:
            (duration, n_exc) boolean array of excitatory neuron spikes.
        """
        duration = input_spikes.shape[0]
        spike_record = np.zeros((duration, self.n_exc), dtype=bool)

        for t in range(duration):
            inp = input_spikes[t].astype(np.float64)

            # Compute excitatory conductance: g_exc[j] = sum_i inp[i] * W[i, j]
            g_exc = inp @ self.synapses.W

            # Step excitatory neurons
            exc_spikes = self.exc_neurons.step(g_exc)
            spike_record[t] = exc_spikes

            if train:
                self.synapses.step(inp.astype(bool), exc_spikes)

            # Lateral inhibition: when any neuron fires, reset others to V_rest
            if np.any(exc_spikes):
                non_firing = ~exc_spikes
                self.exc_neurons.v[non_firing] = self.exc_neurons.v_rest

        if train:
            self.sample_count += 1
            if self.sample_count % 50 == 0:
                self.synapses.normalize()

        return spike_record

    def reset_state(self):
        """Reset neuron states between samples."""
        self.exc_neurons.v = np.full(self.n_exc, self.exc_neurons.v_rest)
        self.exc_neurons.refrac = np.zeros(self.n_exc, dtype=int)
        self.exc_neurons.v_thresh = np.full(
            self.n_exc, self.exc_neurons.v_thresh_base
        )
        self.exc_neurons.spike_trace = np.zeros(self.n_exc)
        self.synapses.pre_trace = np.zeros(self.n_input)
        self.synapses.post_trace = np.zeros(self.n_exc)
