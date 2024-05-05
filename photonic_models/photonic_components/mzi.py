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

         _____ MZI _____
        |    wg1        |
        |--- --[w]-- ---|-C
        |   X       X   |
        |--- ------- ---|-D
        |  bs0     bs1  |
         ---------------
    """

    def __init__(self,
                 n_mzi: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(Mzi, self).__init__()
        i_loss_0 = None
        i_loss_1 = None
        imbalance_0 = None
        imbalance_1 = None
        if mmi_i_losses is not None:
            i_loss_0 = mmi_i_losses[0]
            i_loss_1 = mmi_i_losses[1]
        if mmi_imbalances is not None:
            imbalance_0 = mmi_imbalances[0]
            imbalance_1 = mmi_imbalances[1]

        self.mmi_0 = Beamsplitter(n_bs=n_mzi,
                                  i_loss=i_loss_0,
                                  imbalance=imbalance_0)
        self.heater = Waveguide(n_wg=n_mzi)
        self.mmi_1 = Beamsplitter(n_bs=n_mzi,
                                  i_loss=i_loss_1,
                                  imbalance=imbalance_1)

    def forward(self, portA, portB):
        """

        Args:
            portA (): Tensor of type [batch, features]
            portB (): Tensor of type [batch, features]

        Returns:
        """
        wave_0, wave_1 = self.mmi_0(portA, portB)
        outPs = self.heater(wave_0)
        y_0, y_1 = self.mmi_1(outPs, wave_1)
        return y_0, y_1
