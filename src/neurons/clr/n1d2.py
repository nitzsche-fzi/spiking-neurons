import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N1D2(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 8.927202837964593,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0], 1.494614601135254), ([1], 0.48830416798591614), ([0, 0], 1.1130210161209106)],
            ],
            n_polynomials=1,
            degree=2,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [],
            ],
            n_polynomials=1,
            degree=1,
        )

        params = PMSN_CLR_Params(
            polynomial_coeffs=polynomial_coeffs,
            threshold=threshold,
            reset_coeffs=reset_coeffs,
            state_delta=state_delta,
            n_states=1,
            degree=2,
            resting_state=torch.tensor([0.0]),
            spike_rate=0.05580001696944237,
            energy_idle=55.02000000000001,
            energy_spike=608.1,
        )

        super().__init__(params)
