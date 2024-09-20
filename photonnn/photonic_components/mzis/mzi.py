import torch
from photonnn import PnnModule
from photonnn.photonic_components.beamsplitters.mmi import Mmi


class Mzi(PnnModule):
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
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None):
        """Constructor for the Mzi class."""
        super(Mzi, self).__init__()

        i_loss_0, i_loss_1 = self._check_if_losses(mmi_i_losses)
        imbalance_0, imbalance_1 = self._check_if_imbalanced(mmi_imbalances)
        self._mmi_0 = Mmi(n_mmis=n_mzi, i_loss=i_loss_0, imbalance=imbalance_0)
        self._mmi_1 = Mmi(n_mmis=n_mzi, i_loss=i_loss_1, imbalance=imbalance_1)

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

        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, mmi_1_diag_matrix)
        )
        return tf_matrix

    def _check_if_imbalanced(self, mmi_imbalances: torch.Tensor):
        """Check and return imbalances for the MMI.

        Args:
            mmi_imbalances (torch.Tensor): Tensor of imbalances.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Imbalances for the two MMIs.
        """
        if mmi_imbalances is None:
            return None, None
        elif mmi_imbalances.shape[0] == 1:
            return mmi_imbalances, mmi_imbalances
        else:
            return mmi_imbalances[0], mmi_imbalances[1]

    def _check_if_losses(self, mmi_i_losses: torch.Tensor):
        """Check and return insertion losses for the MMI.

        Args:
            mmi_i_losses (torch.Tensor): Tensor of insertion losses.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Insertion losses for the two MMIs.
        """
        if mmi_i_losses is None:
            return None, None
        elif mmi_i_losses.shape[0] == 1:
            return mmi_i_losses, mmi_i_losses
        else:
            return mmi_i_losses[0], mmi_i_losses[1]
