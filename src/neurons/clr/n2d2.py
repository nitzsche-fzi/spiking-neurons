import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N2D2(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 9.476533797583208,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0, 2], 2.2999775409698486), ([2, 2], 1.118208885192871)],
                [([0, 0], 0.7606891989707947), ([0, 2], 0.63163161277771)],
            ],
            n_polynomials=2,
            degree=2,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [([1], 0.098269984126091)],
                [],
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
            degree=2,
            resting_state=torch.tensor([0.9411954879760742, 0.0]),
            spike_rate=0.06190001592040062,
            energy_idle=103.852,
            energy_spike=1207.08,
        )

        super().__init__(params)