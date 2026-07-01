import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N1D1(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 7.369365822647005,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0], 2.5485799312591553), ([1], 0.033389247953891754)],
            ],
            n_polynomials=1,
            degree=1,
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
            degree=1,
            resting_state=torch.tensor([0.0]),
            spike_rate=0.03260001912713051,
            energy_idle=22.24,
            energy_spike=307.68,
        )

        super().__init__(params)