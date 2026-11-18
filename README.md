# LuxIA - Photonic Neural Networks

## Overview

LuxIA is a Python package for simulating large-scale photonic neural networks.
Built on top of PyTorch, it provides a modular and extensible framework for defining,
training, and benchmarking photonic neural network architectures.

## Installation

Clone the repository:

```bash
git clone https://github.com/smilies-polito/Luxia.git
cd Luxia
```

Install the package:

```bash
pip install .
```

For editable/development mode:

```bash
pip install -e .
```

## Examples

The `examples/` directory contains ready-to-run Python scripts organised by topic:

| Directory | Description |
|-----------|-------------|
| `examples/python/basic_examples/` | One-file demos for every mesh architecture (Clements Bell, Fldzhyan Bell, Braided, …) |
| `examples/python/accelerators/` | Full accelerator pipelines with DAC/ADC and photodetectors |
| `examples/python/custom_mesh/` | Shows how to build your own photonic mesh on top of LuxIA primitives |
| `examples/python/trainable_accelerator/` | End-to-end training and evaluation loop |
| `examples/pnn-benchmarks/` | Multi-tool benchmarking suite comparing LuxIA against other PNN frameworks |

## Quick Start

Run a mesh forward-pass in one line:

```bash
python examples/python/basic_examples/clements_bell_example.py
```

Or kick off a training run on the Iris dataset:

```bash
cd examples/pnn-benchmarks/bash_train_scripts/train_luxia
bash lx_iris_clements.sh
```

## Citation

If you use LuxIA in your research, please cite:

```bibtex
@ARTICLE{Luxia2026,
  author={Melendez Carmona, Tzamn and Marchesin, Federico and Abrate, Marco P. and Bienstman, Peter and Di Carlo, Stefano and Savino, Alessandro},
  journal={IEEE Transactions on Computers}, 
  title={LuxIA: A Lightweight Unitary Matrix-Based Framework Built on an Iterative Algorithm for Photonic Neural Network Training}, 
  year={2026},
  volume={75},
  number={8},
  pages={2885-2899},
  keywords={Training;Modeling;Accuracy;Matrices;Cells (biology);Memory;Windows;Optimization;Timing;Equations;Edge-computing;photonics;simulation tools;neural networks;artificial intelligence},
  doi={10.1109/TC.2026.3698750}
}
```

## License

LuxIA is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
Any modifications to this software or derivative works must also be made available under AGPL-3.0.

Copyright (C) 2024–2026 Politecnico di Torino, Marco P. Abrate

## Documentation

Documentation and additional tutorials will be added soon.

