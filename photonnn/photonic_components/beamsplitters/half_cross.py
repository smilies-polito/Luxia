import torch
import torch.nn as nn


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
        one = torch.tensor([1+0j])
        mC = one if i_loss is None else (1-i_loss)*one
        mC = torch.sqrt(mC)

        self.register_buffer(
            '_C', mC if crosstalk is None else mC * (torch.sqrt(crosstalk)))

    def forward(self, portA):

        y = self._C * portA

        return y

    def get_coefficients(self):
        return self._C
