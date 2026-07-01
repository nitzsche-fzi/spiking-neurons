import torch
import torch.nn as nn

# ===========================
# Surrogate gradient families
# ===========================

def superspike(grad, x, alpha):
    return grad / (alpha * torch.abs(x) + 1.0).pow(2)

def scaled_superspike(grad, x, alpha):
    return 0.5 * alpha * grad / (alpha * torch.abs(x) + 1.0).pow(2)

def atan_derivative(grad, x, alpha):
    return grad * alpha / (torch.pi * (alpha**2 * x**2 + 1))

def straight_through(grad, x, alpha):
    return grad

def triangular(grad, x, alpha):
    mask = (torch.abs(x) <= (1.0 / alpha)).to(x.dtype)
    return grad * alpha * (1.0 - alpha * torch.abs(x)) * mask

def gaussian(grad, x, alpha):
    return grad * (alpha / torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(-0.5 * (alpha * x)**2)

def box(grad, x, alpha):
    mask = (torch.abs(x) <= (1.0 / alpha)).to(x.dtype)
    return grad * (alpha / 2.0) * mask

# =======================================
# Factory: INSTANT-SPREAD VERSION
# =======================================

def get_normalized_heaviside(surrogate_method: str,
                             alpha: float,
                             reduce_dims=(0,)):
    """
    A cudagraph-safe Heaviside with surrogate gradients.
    Normalizes alpha based only on the *current* input spread.
    No EMA, no persistent state, no mutation.

    alpha_dyn = alpha / (sqrt(mean(x^2 over reduce_dims)) + eps)
    """

    # pick surrogate
    if surrogate_method == "superspike":
        surrogate_func = superspike
    elif surrogate_method == "scaled_superspike":
        surrogate_func = scaled_superspike
    elif surrogate_method == "atan_derivative":
        surrogate_func = atan_derivative
    elif surrogate_method == "straight_through":
        surrogate_func = straight_through
    elif surrogate_method == "triangular":
        surrogate_func = triangular
    elif surrogate_method == "gaussian":
        surrogate_func = gaussian
    elif surrogate_method == "box":
        surrogate_func = box
    else:
        raise ValueError(f"Unknown surrogate method: {surrogate_method}")

    # custom autograd
    class HeavisideFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha_dyn):
            ctx.save_for_backward(x, alpha_dyn)
            return (x > 0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            x, alpha_dyn = ctx.saved_tensors
            return surrogate_func(grad_output, x, alpha_dyn), None

    # module wrapper
    class Heaviside(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = alpha
            if isinstance(reduce_dims, int):
                self.reduce_dims = (reduce_dims,)
            else:
                self.reduce_dims = tuple(sorted(reduce_dims))

        def _compute_instant_spread(self, x):
            mom2 = x * x
            for d in self.reduce_dims:
                mom2 = mom2.mean(dim=d, keepdim=True)
            return torch.sqrt(mom2 + 1e-12).detach()

        def forward(self, x):
            # compute spread on the fly (no state)
            with torch.no_grad():
                spread = self._compute_instant_spread(x)
                alpha_dyn = (self.alpha / (spread + 1e-12)).detach()

            return HeavisideFunction.apply(x, alpha_dyn)

    return Heaviside()
