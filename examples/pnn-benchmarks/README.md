# PNN Benchmarks

## Overview

This project contains the complete benchmarking environment used to evaluate LuxIA against other photonic neural network frameworks.

The benchmark suite was used to evaluate:

* training performance
* execution time
* CPU/GPU memory usage
* numerical accuracy
* scalability across large photonic meshes
* training behavior across multiple datasets and architectures

The experiments reproduce the evaluations presented in the LuxIA paper.

## Benchmarked Frameworks

The project includes implementations and benchmarking support for:

* LuxIA
* Neurophox
* Neuroptica
* Photontorch

## Supported Photonic Architectures

The benchmark suite includes the following PUM mesh architectures:

* Clements
* Clements Bell
* Fldzhyan
* Fldzhyan Bell

## Datasets

The following datasets are used throughout the experiments:

* Iris
* Digits
* MNIST
* Olivetti Faces

## Installation

Install the benchmark dependencies:

```bash
pip install -r util/requirements.txt
```

## Project Structure

```text
pnn-benchmarks/
├── bash_error_scripts/
├── bash_scripts/
├── bash_test_scripts/
├── bash_train_scripts/
├── datasets/
├── lx_models/
├── neu_models/
├── nx_models/
├── pt_models/
├── plot_pyscripts/
└── util/
```

## Folder Description

### bash_scripts

Utility scripts used to verify that the required software and dependencies are correctly installed on a Linux system.

These scripts also prepare isolated Python virtual environments for each framework used in the benchmarks.

### bash_train_scripts

Training launch scripts for all benchmark experiments.

These scripts configure the experiments and call the main training pipeline implemented in:

```text
util/train.py
```

The corresponding framework-specific models are loaded from:

* `lx_models/`
* `pt_models/`
* `neu_models/`
* `nx_models/`

### bash_test_scripts

Scripts used to evaluate previously trained models.

These scripts require trained checkpoints/models generated during the training phase and execute the testing pipeline.

### bash_error_scripts

Scripts used for numerical error analysis and validation experiments.

These experiments are used to evaluate numerical differences between methods and configurations.

### plot_pyscripts

Python plotting utilities used to generate benchmark figures and visualizations.

### util

Core utilities and benchmark infrastructure.

Important files include:

* `train.py` — main training pipeline
* `test.py` — evaluation pipeline
* `benchmark_window.py` — runtime and memory benchmark utilities
* `numerical_window_error.py` — numerical accuracy experiments
* `requirements.txt` — benchmark dependencies
* `neurophox_constrains.txt` — compatibility constraints for Neurophox

## Quick Start

Example LuxIA training run:

```bash
bash bash_train_scripts/train_luxia/lx_iris_clements.sh
```

After training, models can be evaluated using the testing scripts from:

```text
bash_test_scripts/
```

## Benchmark Goals

The benchmark suite is designed to evaluate:

* scalability of photonic neural network frameworks
* execution time for large photonic meshes
* CPU and GPU memory consumption
* numerical stability
* convergence behavior during training
* comparative performance across multiple datasets and architectures

## Reproducibility

The scripts included in this project were used to generate the experiments and benchmark results reported in the LuxIA paper.

## Status

This benchmark suite is under active development and may evolve alongside the LuxIA framework.

