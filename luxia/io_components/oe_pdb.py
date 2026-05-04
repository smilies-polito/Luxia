import torch
from torch import nn
from luxia.photonic_components import Photodetector
from luxia.io_components import OpticalToElectrical


class OE_PDB(OpticalToElectrical):
    """Optical-to-electrical block with photodetection, diagonal gain, and bias."""

    def __init__(self, n_inputs: int):
        super(OE_PDB, self).__init__()

        self.photodetectors = Photodetector()

        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        """Convert optical inputs to electrical outputs and apply affine scaling."""
        x_reals = self.photodetectors(x_complex)
        return x_reals * self.diag_layer + self.bias
