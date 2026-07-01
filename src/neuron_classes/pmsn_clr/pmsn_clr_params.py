import torch

class PMSN_CLR_Params:
    def __init__(self, polynomial_coeffs, threshold, reset_coeffs, state_delta, n_states, degree,
                 surrogate_method="scaled_superspike", surrogate_alpha=1.0, resting_state=None,
                 spike_rate=0.0, detach_spikes=True, energy_idle=0.0, energy_spike=0.0):
        self.polynomial_coeffs = polynomial_coeffs
        self.threshold = threshold
        self.reset_coeffs = reset_coeffs
        self.state_delta = state_delta
        self.n_states = n_states
        self.degree = degree
        self.resting_state = torch.zeros(n_states) if resting_state is None else resting_state
        assert self.resting_state.shape == (n_states,), "restng_state should be a torch tensor of shape (n_states,) or None"
        self.spike_rate = spike_rate
        self.energy_idle = float(energy_idle)
        self.energy_spike = float(energy_spike)
        self.surrogate_method = surrogate_method
        self.surrogate_alpha = surrogate_alpha
        self.detach_spikes = detach_spikes

    def set_resting_state(self, resting_state):
        self.resting_state = resting_state

    def set_spike_rate(self, spike_rate):
        self.spike_rate = spike_rate
    
    def save(self, path):
        torch.save({
            'polynomial_coeffs': self.polynomial_coeffs,
            'threshold': self.threshold,
            'reset_coeffs': self.reset_coeffs,
            'state_delta': self.state_delta,
            'n_states': self.n_states,
            'degree': self.degree,
            'resting_state': self.resting_state,
            'spike_rate': self.spike_rate,
            'energy_idle': self.energy_idle,
            'energy_spike': self.energy_spike,
            'surrogate_method': self.surrogate_method,
            'surrogate_alpha': self.surrogate_alpha
        }, path)

    @staticmethod
    def load(path):
        state = torch.load(path, weights_only = False)
        return PMSN_CLR_Params(
            state['polynomial_coeffs'], 
            state['threshold'], 
            state['reset_coeffs'], 
            state['state_delta'], 
            state['n_states'], 
            state['degree'], 
            state['surrogate_method'],
            state['surrogate_alpha'],
            state['resting_state'],
            state['spike_rate'],
            state.get('detach_spikes', True),
            state.get('energy_idle', 0.0),
            state.get('energy_spike', 0.0),
        )

