from ...neuron_classes import PMSN_CLR, PMSN_CLR_Params
from ..utils import get_coefficients

class LIFBox(PMSN_CLR):
    def __init__(self,
        state_decay: float = 0.7553391009569168,
        threshold: float = 9.996206283569336,
        input_scale: float = 2.462735652923584,
        reset_value: float = 2.914475679397583,
        surrogate_method: str = "scaled_superspike",
        surrogate_alpha: float = 1.0,
    ):
        polynomial_coeffs = get_coefficients(
            coeffs=[
                [([0], input_scale), ([1], state_decay-1.0)]
            ],
            n_polynomials=1,
            degree=1
        )

        reset_coeffs = get_coefficients(
            coeffs=[
                [([], reset_value)]
            ],
            n_polynomials=1,
            degree=1
        )

        params = PMSN_CLR_Params(
            polynomial_coeffs=polynomial_coeffs,
            threshold=threshold,
            reset_coeffs=reset_coeffs,
            state_delta=1.0,
            n_states=1,
            degree=1,
            surrogate_method=surrogate_method,
            surrogate_alpha=surrogate_alpha,
            spike_rate=0.132,
            energy_idle=24.104,
            energy_spike=335.91999999999996,
        )

        super().__init__(params)
