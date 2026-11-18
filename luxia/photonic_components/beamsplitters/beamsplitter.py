import torch
from luxia import LxModule
from luxia.params.bs_params import BeamsplitterParams
import copy


class Beamsplitter(LxModule):
    """Class for a beamsplitter. A beamsplitter is a device that splits an
    input beam of light into two output beams. The model follows the following

    Attributes:
        bs_params (dataclass): Dataclass containing the parameters of the
        beamsplitter.
    """

    def __init__(self, bs_params: BeamsplitterParams):
        """Constructor for the Beamsplitter class."""
        super(Beamsplitter, self).__init__()
        self._params = copy.deepcopy(bs_params)
        self._are_coeff_set = False

    def forward(self, portA: torch.Tensor, portB: torch.Tensor):
        """Forward pass through the beamsplitter.

        Args:
            portA (torch.Tensor): Input tensor from port A.
            portB (torch.Tensor): Input tensor from port B.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors from the
            beamsplitter.
        """
        self._check_if_coefficients_are_set()

        if not isinstance(portA, torch.Tensor) or not isinstance(portB, torch.Tensor):
            raise TypeError("Inputs must be torch tensors")

        y_0 = self._C_00 * portA + self._C_01 * portB
        y_1 = self._C_10 * portA + self._C_11 * portB

        return y_0, y_1

    def register_coefficients(self,
                              c00: torch.Tensor, c01: torch.Tensor,
                              c10: torch.Tensor, c11: torch.Tensor):
        """Register the coefficients for the beamsplitter.

        Args:
            c00 (torch.Tensor): Coefficient for portA to y_0.
            c01 (torch.Tensor): Coefficient for portB to y_0.
            c10 (torch.Tensor): Coefficient for portA to y_1.
            c11 (torch.Tensor): Coefficient for portB to y_1.
        """
        if not all(isinstance(c, torch.Tensor) for c in [c00, c01, c10, c11]):
            raise TypeError("All coefficients must be torch tensors")

        self.register_buffer('_C_00', c00)
        self.register_buffer('_C_01', c01)
        self.register_buffer('_C_10', c10)
        self.register_buffer('_C_11', c11)
        self._are_coeff_set = True

    def get_tf_matrix(self):
        """Get the transfer function matrix for the beamsplitter.

        Returns:
            torch.Tensor: Transfer function matrix.
        """
        self._check_if_coefficients_are_set()

        if self._C_00.shape[-1] == 1:
            tf_matrix = [
                torch.stack(
                        [torch.tensor([self._C_00, self._C_01]),
                         torch.tensor([self._C_10, self._C_11])]
                )
            ]*self._params.number_of_splitters
        else:
            tf_matrix = torch.stack(
                [
                    torch.stack([self._C_00, self._C_01], dim=-1),
                    torch.stack([self._C_10, self._C_11], dim=-1)
                ],
                dim=-1
            )
        # Create a block diagonal matrix with the mmi tfunction matrices
        tf_matrix = torch.block_diag(*tf_matrix)
        return tf_matrix

    def _check_if_coefficients_are_set(self):
        """Check if the coefficients have been set."""
        if not self._are_coeff_set:
            raise ValueError(
                    "Coefficients have not been set. Please call: "
                    "register_coefficients' in your constructor to set the "
                    "coefficients of transfer function matrix of the"
                    "beamsplitter")
