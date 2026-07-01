import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N3D2(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 4.743375204478946,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0], 1.1928743124008179), ([2, 2], 0.9214622378349304)],
                [([3], 0.10607706010341644), ([0, 0], 0.4860191345214844), ([0, 2], 1.1291004419326782)],
                [([1, 3], -0.909032940864563)],
            ],
            n_polynomials=3,
            degree=2,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [],
                [],
                [],
            ],
            n_polynomials=3,
            degree=1,
        )

        params = PMSN_CLR_Params(
            polynomial_coeffs=polynomial_coeffs,
            threshold=threshold,
            reset_coeffs=reset_coeffs,
            state_delta=state_delta,
            n_states=3,
            degree=2,
            resting_state=torch.tensor([0.0, 0.0, 0.0]),
            spike_rate=0.05865002050995827,
            energy_idle=86.905,
            energy_spike=1010.62,
        )

        super().__init__(params)