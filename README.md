# PhotonNN - Photonic Neural Networks
## Overview
PhotonNN is a Python package for simulating photonic large-scale neural networks. The package is
designed to be modular and extensible, allowing users to easily define and simulate a wide range of
photonic neural network architectures. The package is built on top of the
[PyTorch](https://pytorch.org/) deep learning framework, and is designed to be compatible with
PyTorch's neural network modules.

## Installation
For now, the package is not available on PyPI, so you will need to install it from source. To do so,
clone the repository and run the following command in the root directory of the repository:
```bash
git clone https://gitlabtsgroup.polito.it/root/photonnn.git 
```
Enter your credentials and then move to the cloned repository:
```bash
cd photonnn
```
Then run the following command:
```bash
pip install .
```
if you want to install the package in editable mode, run the following command in the root directory
of the repository:
```bash
pip install -e .
```

## Getting Started
To get started with PhotonNN, you can check out the examples in the `examples` directory. And you
can try to run the run_train.sh script to train a simple photonic neural network. To do so, go to
photonnn/examples and run the following command:

```bash
./run_train.sh
```
If everything is set up correctly, you should see the training loss decreasing over time.
if you want to see the training loss plot and the accuracy plot, you will notice that under the
examples directory a new directory called experiment was created. You will find 2 images, one for
the training loss and the other for the accuracy. you can open them using any image viewer of your
preference.

## Examples
### Accelerators
On the examples/accelerators directory, you will find different examples for creating different
types of photonic neural networks accelerators. Such as the Clements, Clements Bell, Fldzhyan, and
Fldzhyan Bell. You can run those examples by commenting and uncommenting the desired example in the
run_train.sh
### Basic Examples
On the examples/basic_examples directory, you will find different examples for creating different
and instanciating different types of photonin meshes, creating different types of photonic blocks,
etc. To run those examples, you can directly run the desired example for example, for running the
clements_example.py, you can run the following command:
```bash
python clements_example.py
```

## Documentation
To be added soon.

