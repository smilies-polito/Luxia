import torch
from torch import nn
from luxia.photonic_components.ideal_photodetector import IdealPhotodetector
from luxia.io_components.optical_to_electrical import OpticalToElectrical


class OE_PDB(OpticalToElectrical):
    def __init__(self, n_inputs: int):
        super(OE_PDB, self).__init__()

        self.photodetectors = IdealPhotodetector()

        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex):
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias
