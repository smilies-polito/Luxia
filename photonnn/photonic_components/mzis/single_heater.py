import torch
from photonnn.photonic_components.mzis.mzi import Mzi
from photonnn.photonic_components.heater import Heater
from photonnn.util.util_functions import diag_matrix_alternating_tensor_with_ones


class SingleHeaterMzi(Mzi):
    """
    Mach-Zehnder Interferometer with a single waveguide heater.

         _____ MZI _____
        |    wg1        |
        |--- --[w]-- ---|-C
        |   X       X   |
        |--- ------- ---|-D
        |  mmi0   mmi1  |
         ---------------

    Args:
        n_mzi (int): Number of MZIs.
        mmi_i_losses (torch.Tensor, optional): Tensor of MMI insertion losses.
        mmi_imbalances (torch.Tensor, optional): Tensor of MMI imbalances.
    """

    def __init__(self,
                 n_mzi: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(SingleHeaterMzi, self).__init__(n_mzi=n_mzi,
                                              mmi_i_losses=mmi_i_losses,
                                              mmi_imbalances=mmi_imbalances)
        self._heater = Heater(n_heaters=n_mzi)

    def forward(self, portA: torch.Tensor, portB: torch.Tensor):
        """
        Forward pass through the MZI with a single heater.

        Args:
            portA (torch.Tensor): Input tensor from port A.
            portB (torch.Tensor): Input tensor from port B.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors from the MZI.
        """
        wave_0, wave_1 = self._mmi_0(portA, portB)
        outPs = self._heater(wave_0)
        return self._mmi_1(outPs, wave_1)

    def get_tf_matrix(self) -> torch.Tensor:
        """
        Get the transfer function matrix for the MZI.

        Returns:
            torch.Tensor: Transfer function matrix.
        """
        mmi_0_diag_matrix = self._mmi_0.get_tf_matrix()
        mmi_1_diag_matrix = self._mmi_1.get_tf_matrix()
        heater_phases = self._heater.getPhases()
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)

        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, heater_diag_matrix, mmi_1_diag_matrix)
        )

        return tf_matrix
