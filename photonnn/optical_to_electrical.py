import torch
import torch.nn as nn
from .photonic_components.photodetector import Photodetector


class OpticalToElectrical(nn.Module):
    def __init__(self, n_inputs: int):
        super(OpticalToElectrical, self).__init__()

        self.photodetectors = Photodetector()
        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex):
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias
