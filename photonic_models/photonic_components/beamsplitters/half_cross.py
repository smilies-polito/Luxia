
import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HalfCross(nn.Module):

    """
    Two input - two output cross beamsplitter.
        The model follows the following S-matrix:

        A --
            X
        - -- --D

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
        super(HalfCross, self).__init__()
        matCoeff = torch.tensor(1, device=device)
        matCoeff = matCoeff if i_loss is None else (1-i_loss)

        c = torch.sqrt(matCoeff) * torch.tensor([1+0j], device=device)

        self._pCoeff = c if crosstalk is None else c * (torch.sqrt(1-crosstalk))

    def forward(self, portA):

        y = self._pCoeff * portA

        return y

    def get_coefficients(self):
        return self._pCoeff
