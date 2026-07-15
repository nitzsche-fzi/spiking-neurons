import torch


class BaseNeuronClass(torch.nn.Module):
    """Base class for spiking neurons that provides hardware-energy estimation.

    Subclasses must call ``super().__init__(energy_idle, energy_spike)`` in
    their ``__init__`` so that energy attributes are available on the instance.
    """

    def __init__(self, energy_idle: float, energy_spike: float):
        super().__init__()
        self.energy_idle = energy_idle
        self.energy_spike = energy_spike

    def compute_energy(self, spike_rate: float, include_idle: bool = False) -> float:
        """Single-neuron energy for a given instantaneous spike rate.

        Parameters
        ----------
        spike_rate : float
            Firing rate in (0, 1), e.g. the mean normalised spike count per
            neuron per step.
        include_idle : bool, optional
            If ``True``, include idle energy for non-spiking updates. By default
            only spike energy is counted.

        Returns
        -------
        float
            Energy in the same units as ``energy_idle`` / ``energy_spike``.
        """
        if not include_idle:
            return spike_rate * self.energy_spike
        return (1.0 - spike_rate) * self.energy_idle + spike_rate * self.energy_spike

    def total_energy(self, spike_rates: torch.Tensor, include_idle: bool = False) -> float:
        """Network-level energy by summing over a population.

        Parameters
        ----------
        spike_rates : torch.Tensor
            Per-neuron spike rates (any shape). Each element is a rate in (0, 1).
        include_idle : bool, optional
            If ``True``, include idle energy for non-spiking updates. By default
            only spike energy is counted.

        Returns
        -------
        float
            Sum of ``compute_energy(r)`` over all elements of ``spike_rates``.
        """
        rates = torch.as_tensor(spike_rates, dtype=torch.float32)
        if not include_idle:
            return (rates * self.energy_spike).sum().item()
        return (
            (1.0 - rates) * self.energy_idle + rates * self.energy_spike
        ).sum().item()
