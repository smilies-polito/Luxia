# Import luxia and from it import photonic_meshes and io_components
import torch
import luxia as lx
import luxia.io_components as io
import luxia.photonic_meshes as msh

# Define the number of inputs and layers of the Braided Mesh
n_in = 5
n_lay = 2
# Define the number of classes of the Photonic Accelerator
n_clss = 3

# Create a electrical to optical module
electrical_to_optical = io.ElectricalToOptical()
# Create a photonic mesh
ph_mesh = msh.BraidedMesh(n_inputs=n_in, n_layers=n_lay, is_matmul=False)
# Create a optical to electrical module
optical_to_electrical = io.OpticalToElectrical()

# Create the accelerator
my_accelerator = lx.LxAccelerator(
    n_inputs=n_in, n_classes=n_clss,
    lx_module_list=[electrical_to_optical, ph_mesh, optical_to_electrical]
)

# Create a random input tensor
x = torch.rand(1, n_in)
print("Input tensor:", x)
# Perform the forward pass
y = my_accelerator(x)
# Print the output
print(y)
