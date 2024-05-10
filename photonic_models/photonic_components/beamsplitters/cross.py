
import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cross(nn.Module):

    """
    Two input - two output cross beamsplitter.
        The model follows the following S-matrix:

        A -- --C                | 1 | j |
            X               1 x |   |   |
        B -- --D                | j | 1 |

        Args:
            n_crosses: number of Cross beamsplitters, to set one after the
            other downsides
            i_loss: Tesor of size n_crosses, this is required for setting the
            insertion loss
            crosstalk: Tensor of size n_crosses, this term is only valid for
            cross type beamsplitter
    """

    def __init__(self,
                 n_crosses: int,
                 i_loss: torch.Tensor = None,
                 crosstalk: torch.Tensor = None
                 ):
        # Constructor of the Class Cross
        super(Cross, self).__init__()
        if i_loss is None:
            matCoeff = 1
        else:
            matCoeff = (1 - i_loss)

        c0 = matCoeff * torch.tensor([1+0j], device=device)
        c1 = matCoeff * torch.tensor([0+1j], device=device)

        coeff0 = c0
        coeff1 = c1
        coeff2 = c0
        coeff3 = c1

        if crosstalk is not None:
            coeff0 = coeff0 * (1 - crosstalk)
            coeff1 = coeff1 * (1 + crosstalk)
            coeff2 = coeff2 * (1 + crosstalk)
            coeff3 = coeff3 * (1 - crosstalk)

        self._pCoeff0 = coeff0
        self._pCoeff1 = coeff1
        self._pCoeff2 = coeff2
        self._pCoeff3 = coeff3

    def forward(self, portA, portB):

        y_0 = self._pCoeff0 * portA + self._pCoeff1 * portB
        y_1 = self._pCoeff2 * portA + self._pCoeff3 * portB

        return y_0, y_1

    def get_tf_matrix(self):
        if self.i_loss is None and self.imbalance is None:
            tf_matrix = [torch.stack([
                torch.tensor([self._pCoeff0, self._pCoeff1], dtype=torch.complex64),
                torch.tensor([self._pCoeff2, self._pCoeff3], dtype=torch.complex64)
            ])]*self.n_mmis
        else:
            tf_matrix = torch.stack(
                [
                    torch.stack([self._pCoeff0, self._pCoeff1], dim=-1),
                    torch.stack([self._pCoeff2, self._pCoeff3], dim=-1)
                ],
                dim=-1
            )
        return tf_matrix
