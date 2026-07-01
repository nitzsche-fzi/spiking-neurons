import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N2D1(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 10.48284409604818,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([1], -0.8585874438285828), ([2], -5.3628315925598145)],
                [([0], -6.157683372497559), ([1], 0.7055922150611877)],
            ],
            n_polynomials=2,
            degree=1,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [([2], 0.2777828872203827)],
                [([0], -0.08679541200399399)],
            ],
            n_polynomials=2,
            degree=1,
        )

        params = PMSN_CLR_Params(
            polynomial_coeffs=polynomial_coeffs,
            threshold=threshold,
            reset_coeffs=reset_coeffs,
            state_delta=state_delta,
            n_states=2,
            degree=1,
            resting_state=torch.tensor([-0.0011501925764605403, 0.00035746218054555357]),
            spike_rate=0.08180002868175507,
            energy_idle=96.91200000000002,
            energy_spike=1018.62,
        )

        super().__init__(params)