# Evolved Spiking Neurons
This repository provides several new spiking neuron models in PyTorch, built to integrate smoothly with Norse, snnTorch and other libraries in the ecosystem.
Additionally, it contains a weight initializer for spiking layers and data augmentation functions for event-based audio and video data.

---

## Repository Structure

```
repo/
├── esn/
│   ├── neuron_classes/
│   │   ├── pmsn_clr       # PMSN-CLR class definition
│   ├── neurons/
│   │   ├── clr/           # Instantiated PMSN-CLR models
|   |   |   ├── n1d1.py    # model with 1 internal state with polynomial degree 1
|   |   |   |── n2d2.py    # model with 2 internal states with polynomial degree 2
│   │   |   └── ...           
│   ├── initializers/
│   │   ├── linear.py      # Custom weight initializers for linear layers
│   ├── augmentations/
│   │   └── ...            # Data augmentation modules for tonic
│   └── ...
├── README.md
└── pyproject.toml
```

1. **`esn.neuron_classes`** houses the generalized neuron definitions.
2. **`esn.neurons`** contains instantiated/learned models.
3. **`esn.initializers`** provides custom weight initializers for sparse spiking inputs.
4. **`esn.augmentations`** offers extended data augmentation methods for spiking data.

---

## Installation
1. Clone this repository:

```
git clone https://anonymous.4open.science/r/evolved-spiking-neurons/
cd evolved-spiking-neurons
```

2. Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

3. Install the package:

```
pip install -e .
```

To run the examples, install the optional example dependencies as well:

```
pip install -e ".[examples]"
```

---

## Tutorial Notebooks
The example notebooks show basic training workflows and are the recommended starting point for learning how to use the provided neuron models:

- [Norse tutorial](examples/norse.ipynb): train an NMNIST classifier with `norse.torch.SequentialState` and an evolved `N2D2` neuron.
- [snnTorch tutorial](examples/snntorch.ipynb): train the same NMNIST classifier with an explicit snnTorch-style time loop.
- [Energy proxy tutorial](examples/energy.ipynb): estimate test-set energy proxy values from hidden-layer spikes to compare network efficiency.

---

## Usage
The provided neuron models are regular PyTorch modules that return spikes and an updated neuron state:

```
from esn.neurons.clr import N2D2

neuron = N2D2()
spikes, state = neuron(input_current, state=None)
```

### Norse usage
In Norse, the neurons can be placed directly inside `norse.torch.SequentialState` modules:

```
import torch.nn as nn
import norse.torch as snn
from esn.neurons.clr import N2D2

model = snn.SequentialState(
    nn.Linear(input_features, hidden_features, bias=False),
    N2D2(threshold=1.0),
    nn.Linear(hidden_features, n_classes),
)

state = None
logits_t, state = model(x_t, state)
```

See the [Norse tutorial](examples/norse.ipynb) for a complete NMNIST workflow.

### snnTorch usage
For snnTorch-style code, keep the explicit time loop and carry the neuron state between time steps:

```
import torch
import torch.nn as nn
from esn.neurons.clr import N2D2

fc1 = nn.Linear(input_features, hidden_features, bias=False)
neuron = N2D2(threshold=1.0)
fc2 = nn.Linear(hidden_features, n_classes)

state = None
logits_over_time = []
for x_t in input_spikes.transpose(0, 1):
    hidden_current = fc1(x_t)
    hidden_spikes, state = neuron(hidden_current, state)
    logits_over_time.append(fc2(hidden_spikes))

logits = torch.stack(logits_over_time).mean(dim=0)
```

See the [snnTorch tutorial](examples/snntorch.ipynb) for a complete NMNIST workflow.

### Energy proxy usage
Neuron classes include built-in energy proxy values that can be evaluated from observed spike rates. By default, idle energy is ignored:

```
spike_rates = hidden_spikes.float().mean(dim=0)
energy_proxy = neuron.total_energy(spike_rates, include_idle=False)
print(f"Energy proxy: {energy_proxy:.2f} pJ")
```

See the [Energy proxy tutorial](examples/energy.ipynb) for a full test-set calculation.

---

## API Reference

### Neuron Classes
`PMSN` stands for Polynomial Multi-State Neuron, and the suffix indicates the threshold and reset configurations:

- **`PMSN-CLR`**: Constant Threshold, Linear Reset

The neuron classes are each instantiable with parameters describing their internal state dynamics.
All neuron models in this repo are based on these classes.

### Neuron Models
Neuron models are instantiated versions of the generalized classes. These models can be used directly in norse, and are optimized.

They are located in `esn.neurons.clr` and `esn.neurons.nt` directories.

### Weight Initializers
Custom initializers designed for sparse spiking inputs, found in `esn.initializers`.

- **`linear.neuromorphic_input_intializer`**: Initializes weights for `nn.Linear` layers for inputs from neuromorphic datasets.
This should only be used for the first layer of the network. It requires knowledge about the mean and variance of the input data, which has to be determined beforehand. The weight matrix gets initialized with a normal distribution such that the output of the linear layer is normally distributed with mean 0 and variance 1.
- **`linear.neuromorphic_hidden_initializer`**: Initializes weights for `nn.Linear` layers for hidden layers. This initializer requires knowledge about the spike rate of the input, which is provided for the neurons provided with `esn` through `neuron.spike_rate`. The neurons get initialized with a normal distribution such that the output of the linear layer is normally distributed with mean 0 and variance 1.

### Data Augmentations
Modules compatible with tonic, available under `esn.augmentations`.

---

## References
- Norse: https://github.com/norse/norse
- snnTorch: https://github.com/jeshraghian/snntorch
- Tonic: https://github.com/tony-sicilia/tonic
