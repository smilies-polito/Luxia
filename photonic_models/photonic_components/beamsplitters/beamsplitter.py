import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Beamsplitter(nn.Module):
    """
    Beamsplitter class that creates a dielectric beamsplitter with the
    following S-matrix:
        A -- --C                | 1 | 1 |
            X       1/sqrt(2) x |   |   |
        B -- --D                | 1 |-1 |

        Args:
            n_bs: number of beamsplitters, to set one after the other downsides
            i_loss: Tesor of size n_bs, this is required for setting indepent
                losses to each beamsplitters
            imbalance: Tensor of size n_bs, this is required for setting
                indepent imbalance to each beamsplitters.
    """

    def __init__(self,
                 n_bs: int,
                 i_loss: torch.Tensor = None,
                 imbalance: torch.Tensor = None
                 ):
        # Constructor of the Class Beamsplitter
        super(Beamsplitter, self).__init__()
        if i_loss is None:
            matCoeff = 1
        else:
            matCoeff = (1 - i_loss)

        matCoeff = matCoeff / torch.sqrt(torch.tensor(2.0))
        c0 = matCoeff * torch.tensor([1+0j], device=device)

        coeff0 = c0
        coeff1 = c0
        coeff2 = c0
        coeff3 = -c0

        if imbalance is not None:
            coeff0 = coeff0 * (1 - imbalance)
            coeff1 = coeff1 * (1 + imbalance)
            coeff2 = coeff2 * (1 + imbalance)
            coeff3 = coeff3 * (1 - imbalance)

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
