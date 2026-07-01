import torch
from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class N3D1(PMSN_CLR):
    def __init__(self,
        state_delta: float = 0.1,
        threshold: float = 11.645693058070743,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([1], -0.7234458923339844), ([3], 5.79391622543335)],
                [([0], 1.3879823684692383), ([2], -1.1296266317367554)],
                [([0], 6.293953895568848), ([1], -0.6118342280387878)],
            ],
            n_polynomials=3,
            degree=1,
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [],
                [],
                [([2], -0.038465943187475204)],
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
            degree=1,
            resting_state=torch.tensor([-0.6885635256767273, 2.5090960420470765e-09, -0.14992935955524445]),
            spike_rate=0.08065002411603928,
            energy_idle=102.67200000000001,
            energy_spike=1102.32,
        )

        super().__init__(params)