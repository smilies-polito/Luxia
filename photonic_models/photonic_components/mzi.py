import torch
import torch.nn as nn

from .beamsplitter import Beamsplitter
from .waveguide import Waveguide


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mzi(nn.Module):
    """
    March Zender Interferometer, for the moment it implemented with 2
    beamsplitters beamsplitter and a wabeguide as a heater

        A -- --[w]-- --C
            X       X
        B -- ------- --D

    """

    def __init__(self, n_mzi: int):
        super(Mzi, self).__init__()
        self.mmi0 = Beamsplitter()
        self.heater = Waveguide(n_wg=n_mzi)
        self.mmi1 = Beamsplitter()

    def forward(self, portA, portB):
        """

        Args:
            portA (): Tensor of type [batch, features]
            portB (): Tensor of type [batch, features]

        Returns:
        """
        wave_0, wave_1 = self.mmi0(portA, portB)
        outPs = self.heater(wave_0)
        y_0, y_1 = self.mmi1(outPs, wave_1)
        return y_0, y_1
