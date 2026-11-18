import torch
import luxia_lib as lib
from luxia.params import FldzhyanMeshParams, FldzhyanLayerParams

# Define the number of inputs and layers of the FldzhyanMesh
n_inputs = 4
n_layers = 4

# Build FldzhyanMeshParams — one FldzhyanLayerParams per physical layer
layer_params = [
    FldzhyanLayerParams(number_of_inputs=n_inputs, is_even_layer=(i % 2 == 0))
    for i in range(n_layers)
]
mesh_params = FldzhyanMeshParams(
    number_of_inputs=n_inputs, number_of_layers=n_layers, layer_params=layer_params
)

# Create the Fldzhyan mesh using the C++ backend directly (luxia_lib)
mesh = lib.FldzhyanMesh(params=mesh_params)

# Create a random input tensor [batch_size=1, n_inputs]
x = torch.rand(1, n_inputs)
print("Input:", x)

# Forward pass — the mesh returns a complex optical field
print("Performing Forward Pass ..")
y_complex = mesh(x)
print("Complex optical field:", y_complex)

# Optical power (photodetection: |E|^2)
y_power = y_complex.abs().pow(2)
print("Output power |E|^2:", y_power)
