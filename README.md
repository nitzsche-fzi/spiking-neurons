# Evolved Spiking Neurons and Augmentations

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

## Usage

The provided neuron models can be used with both Norse and snnTorch-based workflows.
Basic usage is summarized below. See the [Norse](examples/norse.ipynb) and [snnTorch](examples/snntorch.ipynb) example notebooks for complete workflows.

Model construction:
```
import torch
from esn.neurons.clr import N2D2

neuron = N2D2()
print(neuron.get_forward_code())
```

### Norse usage

The neuron can be used as a drop-in replacement for a Norse neuron model. State
handling is explicit:

```
state = None
for t in range(100):
    input = torch.randn(1, 1)
    out, state = neuron(input, state)
```

The state defaults to `None`, so the following Norse-style usage is also valid:

```
input = torch.randn(1, 1)
out, state = neuron(input)
for t in range(100):
    input = torch.randn(1, 1)
    out, state = neuron(input, state)
```

### snnTorch usage

The neuron models also support snnTorch workflows. A short snnTorch example will
be added here once the recommended usage pattern is finalized.


---

## References

- Norse: https://github.com/norse/norse
- snnTorch: https://github.com/jeshraghian/snntorch
- Tonic: https://github.com/tony-sicilia/tonic
