import torch

from typing import Optional, Tuple, List

class Integrator(torch.nn.Module):
    def __init__(self):
        super(Integrator, self).__init__()

    def forward(self, x : torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the integrator neuron.
        :param x: Input tensor of shape [batch_size, n_neurons].
        :param state: Previous state of the integrator neuron.
        :return: Updated state of the integrator neuron.
        """
        state = state + x
        return state, state
    