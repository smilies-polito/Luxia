import torch
import torch.nn as nn

import beamsplitter as bs
import waveguide as wg


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class mzi(nn.Module):
    """
    March Zender Interferometer, for the moment it implments with 2 beamsplitters of the class
    beamsplitter

        A -- --[w]-- --C
            X       X
        B -- ------- --D

    """

    def __init__(self):
        super(mzi, self).__init__()
        self.mmi0 = bs.beamsplitter()
        self.mmi1 = bs.beamsplitter()
        self.heater = wg.waveguide()

    def forward(self, portA, portB):
        """

        Args:
            portA (): Tensor of type [batch, features]
            portB (): Tensor of type [batch, features]

        Returns:
        """
        wave = self.mmi0(portA, portB)
        y_0 = self.heater(wave[:, 0])
        y = self.mmi1(y_0, wave[:, 1])
        return y
