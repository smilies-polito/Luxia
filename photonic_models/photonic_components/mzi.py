import torch
import torch.nn as nn

from .beamsplitters.mmi import Mmi
from .heater import Heater
from ..util.util_functions import diag_matrix_alternating_tensor_with_ones


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
        |  mmi0   mmi1  |
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

        self._mmi_0 = Mmi(n_mmis=n_mzi,
                          i_loss=i_loss_0,
                          imbalance=imbalance_0)
        self._heater = Heater(n_heaters=n_mzi)
        self._mmi_1 = Mmi(n_mmis=n_mzi,
                          i_loss=i_loss_1,
                          imbalance=imbalance_1)

    def forward(self, portA, portB):
        """
        Args:
            portA (): Tensor of type [batch, features]
            portB (): Tensor of type [batch, features]

        Returns:
        """
        wave_0, wave_1 = self._mmi_0(portA, portB)
        outPs = self._heater(wave_0)
        y_0, y_1 = self._mmi_1(outPs, wave_1)
        return y_0, y_1

    def get_tf_matrix(self):
        mmi_0_tf_matrices = self._mmi_0.get_tf_matrix()
        mmi_1_tf_matrices = self._mmi_1.get_tf_matrix()
        mmi_0_diag_matrix = torch.block_diag(*mmi_0_tf_matrices)
        mmi_1_diag_matrix = torch.block_diag(*mmi_1_tf_matrices)

        heater_phases = self._heater.getPhases()

        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)

        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, heater_diag_matrix, mmi_1_diag_matrix)
        )

        return tf_matrix
