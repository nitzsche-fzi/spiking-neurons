import math
import torch
import torch.nn as nn

def neuromorphic_dense_layer(
    in_features, out_features,
    inp_mean=None,
    inp_var=None,
    spike_rate=None
):
    """
    Initializes weights ~ N(mu_w, sigma_w^2) such that <x, w> ~ N(1, 1),
    assuming x ~ N(inp_mean, inp_var)^n (i.i.d.)
    """
    layer = nn.Linear(in_features, out_features, bias=False)

    if spike_rate is not None:
        inp_mean = spike_rate
        inp_var = spike_rate * (1.0 - spike_rate)

    assert inp_mean is not None, "Input mean or spike rate must be provided."
    assert inp_var is not None, "Input variance or spike rate must be provided."

    n = in_features
    mu_x = inp_mean
    sigma_x_sq = inp_var

    mu_w = 1.0 / (n * mu_x)
    denom = sigma_x_sq + mu_x ** 2
    var_w = (1.0 / denom) * (1.0 / n + 1.0 / n**2) - 1.0 / (n**2 * mu_x**2)

    if var_w < 0:
        raise ValueError(f"Computed negative variance for weights: var_w = {var_w}. Check inp_mean and inp_var.")

    std_w = math.sqrt(var_w)

    with torch.no_grad():
        layer.weight.normal_(mean=mu_w, std=std_w)

    return layer
