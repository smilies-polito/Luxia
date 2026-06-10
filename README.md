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

Inside the `examples/` directory you will find a complete benchmarking project
for comparing multiple photonic neural network tools, including LuxIA.

The benchmark project is located at:

```text
examples/pnn-benchmarks
```

It includes:

* benchmarking scenarios
* training scripts
* testing utilities
* comparisons across multiple architectures and tools

## Quick Start

To quickly try a LuxIA example:

```bash
cd examples/pnn-benchmarks/bash_train_scripts/train_luxia
bash lx_iris_clements.sh
```

## Citation

If you use LuxIA in your research, please cite:

```bibtex

@ARTICLE{11541186,
  author={Carmona, Tzamn Melendez and Marchesin, Federico and Abrate, Marco P. and Bienstman, Peter and Di Carlo, Stefano and Savino, Alessandro},
  journal={IEEE Transactions on Computers}, 
  title={LuxIA: A Lightweight Unitary matriX-based Framework Built on an Iterative Algorithm for Photonic Neural Network Training}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Training;Modeling;Accuracy;Matrices;Cells (biology);Memory;Windows;Optimization;Timing;Equations;Inference Pipeline;Real-time System;Scheduling;Cloud Computing},
  doi={10.1109/TC.2026.3698750}}

```

## Documentation

Documentation and additional tutorials will be added soon.

