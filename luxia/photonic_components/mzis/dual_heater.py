import torch
from luxia.photonic_components.mzis.mzi import Mzi
from luxia.photonic_components.heater import Heater
from luxia.util.util_functions import diag_matrix_alternating_tensors


class DualHeaterMzi(Mzi):
    """
    Mach-Zehnder Interferometer with 2 phase shifters implemented with 2
    beam splitters and 2 waveguides as heaters.

         _____ MZI _____
        |    wg1        |
        |--- --[w]-- ---|-C
        |   X       X   |
        |--- --[w]-- ---|-D
        |  bs0     bs1  |
         ---------------

    Args:
        n_mzi (int): Number of MZIs.
        mmi_i_losses (torch.Tensor, optional): Tensor of MMI insertion losses.
        mmi_imbalances (torch.Tensor, optional): Tensor of MMI imbalances.
    """

    def __init__(self, n_mzi: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        super(DualHeaterMzi, self).__init__(n_mzi=n_mzi,
                                            mmi_i_losses=mmi_i_losses,
                                            mmi_imbalances=mmi_imbalances)
        self._heater_0 = Heater(n_heaters=n_mzi)
        self._heater_1 = Heater(n_heaters=n_mzi)

    def forward(self, portA: torch.Tensor, portB: torch.Tensor):
        """
        Forward pass through the MZI with two heaters.

        Args:
            portA (torch.Tensor): Input tensor from port A.
            portB (torch.Tensor): Input tensor from port B.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors from the MZI.
        """
        wave_0, wave_1 = self._mmi_0(portA, portB)
        outPs_0 = self._heater_0(wave_0)
        outPs_1 = self._heater_1(wave_1)
        return self._mmi_1(outPs_0, outPs_1)

    def get_tf_matrix(self) -> torch.Tensor:
        """
        Get the transfer function matrix for the MZI.

        Returns:
            torch.Tensor: Transfer function matrix.
        """
        mmi_0_diag_matrix = self._mmi_0.get_tf_matrix()
        mmi_1_diag_matrix = self._mmi_1.get_tf_matrix()
        heater_phases_0 = self._heater_0.getPhases()
        heater_phases_1 = self._heater_1.getPhases()

        heater_diag_matrix = diag_matrix_alternating_tensors(
            heater_phases_0, heater_phases_1
        )

        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, heater_diag_matrix, mmi_1_diag_matrix)
        )
        return tf_matrix
