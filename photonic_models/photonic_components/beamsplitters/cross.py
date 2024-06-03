
import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cross(nn.Module):

    """
    Two input - two output cross beamsplitter.
        The model follows the following S-matrix:

        A -- --C                | 0 | 1 |
            X               1 x |   |   |
        B -- --D                | 1 | 0 |

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

        ones = torch.ones(1)
        zeros = torch.zeros(1)

        mC = ones if i_loss is None else torch.sqrt(i_loss)

        ct0 = mC * torch.sqrt(crosstalk) if crosstalk is not None else None
        ct1 = mC * torch.sqrt(1 - crosstalk) if crosstalk is not None else None

        self._C_00 = zeros if crosstalk is None else mC * ct0
        self._C_01 = ones if crosstalk is None else mC * ct1
        self._C_10 = ones if crosstalk is None else mC * ct1
        self._C_11 = zeros if crosstalk is None else mC * ct0

    def forward(self, portA, portB):

        y_0 = self._C_00 * portA + self._C_01 * portB
        y_1 = self._C_10 * portA + self._C_11 * portB
        return y_0, y_1

    def get_tf_matrix(self):
        if self.i_loss is None and self.imbalance is None:
            tf_matrix = [torch.stack([
                torch.tensor([self._C_00, self._C_01], dtype=torch.complex64),
                torch.tensor([self._C_10, self._C_11], dtype=torch.complex64)
            ])]*self.n_mmis
        else:
            tf_matrix = torch.stack(
                [
                    torch.stack([self._C_00, self._C_01], dim=-1),
                    torch.stack([self._C_10, self._C_11], dim=-1)
                ],
                dim=-1
            )
        return tf_matrix
