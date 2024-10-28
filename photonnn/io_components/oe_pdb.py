import torch
from torch import nn
from photonnn.photonic_components import Photodetector
from photonnn.io_components import OpticalToElectrical


class OE_PDB(OpticalToElectrical):
    def __init__(self, n_inputs: int):
        super(OE_PDB, self).__init__()

        self.photodetectors = Photodetector()

        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex):
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias
