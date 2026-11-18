from luxia.photonic_components.beamsplitters.mmi import Mmi
from luxia.params import MmiParams
from typing import List, Tuple
from luxia import LxModule
import torch


class Mzi(LxModule):
    """
    Mach-Zehnder Interferometer implemented with 2 beam splitters.
         _____ MZI _____
        |               |
        |--- ------- ---|-C
        |   X       X   |
        |--- ------- ---|-D
        |  mmi0   mmi1  |
         ---------------
    """

    def __init__(self,
                 n_mzi: int,
                 mmis_params: List[MmiParams] = None,
                 ):
        """
        Constructor for the Mzi class.

        Args:
            n_mzi (int): Number of MZI units.
            mmis_params (List[MmiParams]): List of parameters for MMI components.
                If length is 1, the same parameters are used for both MMIs.
                If length is 2, different parameters are used for each MMI.

        Raises:
            ValueError: If more than 2 MMI parameter sets are provided.
        """
        super(Mzi, self).__init__()
        p = [MmiParams(number_of_splitters=n_mzi)] if mmis_params is None else mmis_params

        if len(p) > 2:
            raise ValueError(
                "Only 1 or 2 MMI parameter sets are allowed in the MZI")

        mmi_0_params = p[0]
        mmi_1_params = p[0]  # Default to same parameters

        if len(p) == 2:
            mmi_1_params = mmis_params[1]

        # Check if the number of splitters is the same as the number of MZIs in each MMI
        n0 = mmi_0_params.number_of_splitters
        n1 = mmi_1_params.number_of_splitters
        if n0 != n_mzi or n1 != n_mzi:
            raise ValueError(
                "Number of splitters in the MMI parameters must be equal to "
                f"the number of MZIs. Got {n0} and {n1} splitters for"
                " MMI 0 and 1, ")

        self._mmi_0 = Mmi(mmi_params=mmi_0_params)
        self._mmi_1 = Mmi(mmi_params=mmi_1_params)

    def forward(self, portA: torch.Tensor, portB: torch.Tensor):
        """Forward pass through the MZI.

        Args:
            portA (torch.Tensor): Input tensor from port A.
            portB (torch.Tensor): Input tensor from port B.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors from the MZI.
        """
        wave_0, wave_1 = self._mmi_0(portA, portB)
        return self._mmi_1(wave_0, wave_1)

    def get_tf_matrix(self) -> torch.Tensor:
        """Get the transfer function matrix for the MZI.

        Returns:
            torch.Tensor: Transfer function matrix.
        """
        mmi_0_tf_matrices = self._mmi_0.get_tf_matrix()
        mmi_1_tf_matrices = self._mmi_1.get_tf_matrix()
        mmi_0_diag_matrix = torch.block_diag(*mmi_0_tf_matrices)
        mmi_1_diag_matrix = torch.block_diag(*mmi_1_tf_matrices)

        # NOTE: Is this the correct way to multiply the matrices?
        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, mmi_1_diag_matrix)
        )
        return tf_matrix
