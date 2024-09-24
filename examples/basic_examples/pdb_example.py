import torch
import torch.nn as nn
from photonnn.io_components import OpticalToElectrical
from photonnn.photonic_components import Photodetector


class OE_PDB(OpticalToElectrical):
    def __init__(self, n_inputs: int):
        self.photodetectors = Photodetector()
        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex):
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias


# Create an instance of the OpticalToElectrical class with a 8 Photodetectors, 
# A DiagLayer, and Bias
pdb = OE_PDB(8)
