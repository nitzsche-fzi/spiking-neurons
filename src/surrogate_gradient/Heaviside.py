import torch
import torch.nn as nn

# Define the surrogate gradient functions
def superspike(grad, x, alpha):
    return grad / (alpha * torch.abs(x) + 1.0).pow(2)

def scaled_superspike(grad, x, alpha):
    return 0.5 * alpha * grad / (alpha * torch.abs(x) + 1.0).pow(2)

def atan_derivative(grad, x, alpha):
    return grad * alpha / (torch.pi * (alpha**2 * x**2 + 1))

def straight_through(grad, x, alpha):
    return grad


def normalized_atan_derivative(grad, x, alpha, reduce_dim=0):
    # this function determines the alpha based on the distribution of x along reduce_dim
    # we take the unnormalized variance to get alpha. 
    mean_x = torch.mean(x, dim=reduce_dim, keepdim=True)
    var_x = torch.mean((x - mean_x)**2, dim=reduce_dim, keepdim=True)
    dynamic_alpha = 1.0 / torch.sqrt(var_x + 1e-12)  # add small constant to avoid division by zero
    return grad * dynamic_alpha / (torch.pi * (dynamic_alpha**2 * x**2 + 1))

def get_heaviside(surrogate_method:str, alpha:float) -> nn.Module:
    """
    Function to dynamically generate the Heaviside class with the specified surrogate method.
    Supported surrogate methods are: superspike, scaled_superspike, atan_derivative, straight_through.
    :param surrogate_method: The surrogate method to use.
    :param alpha: The alpha parameter for the surrogate function.
    :return: A Heaviside module with the specified surrogate method.
    """
    # Select the appropriate surrogate function
    if surrogate_method == "superspike":
        surrogate_func = superspike
    elif surrogate_method == "scaled_superspike":
        surrogate_func = scaled_superspike
    elif surrogate_method == "atan_derivative":
        surrogate_func = atan_derivative
    elif surrogate_method == "straight_through":
        surrogate_func = straight_through
    else:
        raise ValueError(f"Unknown surrogate method: {surrogate_method}")

    class HeavisideFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            ctx.save_for_backward(x)
            return torch.gt(x, 0.0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, = ctx.saved_tensors
            # Directly use alpha from the enclosing scope
            return surrogate_func(grad_output, x, alpha)
    
    class Heaviside(torch.nn.Module):
        def __init__(self):
            super(Heaviside, self).__init__()

        def forward(self, x: torch.Tensor):
            return HeavisideFunction.apply(x)

    return Heaviside()
