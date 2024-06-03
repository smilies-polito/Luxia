import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mmi(nn.Module):
    """
    Multi Mode Interferometer (MMI) beamsplitter. The model follows the
    following TF-matrix:

        A -- --C                | 1 | j |
            X       1/sqrt(2) x |   |   |
        B -- --D                | j | 1 |

    with Imbalance and Insertion Loss, the TF-matrix is:

        A -- --C                    | sqrt(0.5+Imb)     | j*sqrt(0.5-Imb)   |
            X       sqrt(i_loss) x  |                   |                   |
        B -- --D                    | j*sqrt(0.5-Imb)   | sqrt(0.5+Imb)     |
    """

    def __init__(self,
                 n_mmis: int,
                 i_loss: torch.Tensor = None,
                 imbalance: torch.Tensor = None
                 ):
        """

        Args:
            n_mmis: number of MMI beamsplitters, to set one after the other
            downsides
            i_loss: Tesor of size n_mmis, this is required for setting the
            insertion loss
            imbalance: Tensor of size n_mmis, this is required for setting the
            imbalance
        """
        # Constructor of the Class Mmi
        super(Mmi, self).__init__()

        self.n_mmis = n_mmis
        self.i_loss = i_loss
        self.imbalance = imbalance

        ones = torch.ones(1)
        mC = ones/torch.sqrt(ones*2) if i_loss is None else torch.sqrt(i_loss)

        c0 = mC * torch.tensor([1+0j], device=device)
        c1 = mC * torch.tensor([0+1j], device=device)

        imb0 = torch.sqrt(0.5+imbalance) if imbalance is not None else None
        imb1 = torch.sqrt(0.5-imbalance) if imbalance is not None else None

        self._C_00 = c0 if imbalance is None else mC * imb0
        self._C_01 = c1 if imbalance is None else mC * imb1
        self._C_10 = c1 if imbalance is None else mC * imb0
        self._C_11 = c0 if imbalance is None else mC * imb1

    def forward(self, portsA, portsB):

        y_0 = self._C_00 * portsA + self._C_01 * portsB
        y_1 = self._C_10 * portsA + self._C_11 * portsB

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
