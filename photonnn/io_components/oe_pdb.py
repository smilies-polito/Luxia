import torch
from torch import nn
from photonnn.photonic_components import Photodetector
from .optical_to_electrical import OpticalToElectrical


Class OE_PDB(OpticalToElectrical):
    def __init__(self, n_inputs: int):
        self.photodetectors = Photodetector()

        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex):
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias
