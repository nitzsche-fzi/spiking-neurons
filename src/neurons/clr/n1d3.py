import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N1D3(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 2.719810386315708,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0, 0], 0.2349938154220581), ([0, 1], 1.01937735080719), ([1, 1, 1], 0.13486982882022858)],
            ],
            n_polynomials=1,
            degree=3,
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
            degree=3,
            resting_state=torch.tensor([0.0]),
            spike_rate=0.04835003241896629,
            energy_idle=93.44200000000001,
            energy_spike=1093.45,
        )

        super().__init__(params)