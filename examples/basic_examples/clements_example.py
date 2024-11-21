# Import photonnn and from it photonic_architectures and io_components
import torch
import photonnn as pnn
import photonnn.io_components as io
import photonnn.photonic_meshes as ph_arch

# Define the number of inputs and layers of the ClementsMesh
n_in = 8
n_lay = 1
# Define the number of classes of the Photonic Accelerator
n_clss = 3

# Create a electrical to optical module
electrical_to_optical = io.ElectricalToOptical()
# Create a photonic mesh
ph_mesh = ph_arch.ClementsMesh(n_inputs=n_in, n_layers=n_lay, is_matmul=False)
# Create a optical to electrical module
optical_to_electrical = io.OpticalToElectrical()

# Create the accelerator
my_accelerator = pnn.PnnAccelerator(
    n_inputs=n_in, n_classes=n_clss,
    pnn_module_list=[electrical_to_optical, ph_mesh, optical_to_electrical]
)

# Create a random input tensor
x = torch.rand(1, n_in)
# Perform the forward pass
y = my_accelerator(x)
