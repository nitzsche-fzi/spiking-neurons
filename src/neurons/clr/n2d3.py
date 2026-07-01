import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N2D3(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 10.32781538571575,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([2, 2], 1.9648507833480835)],
                [([0], 0.7767528891563416), ([0, 2], 1.1703076362609863), ([0, 0, 1], 1.018011212348938)],
            ],
            n_polynomials=2,
            degree=3,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [],
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
            degree=3,
            resting_state=torch.tensor([0.0, 0.0]),
            spike_rate=0.058300018310546875,
            energy_idle=116.59199999999998,
            energy_spike=1262.8,
        )

        super().__init__(params)