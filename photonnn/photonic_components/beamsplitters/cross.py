import torch
from photonnn.photonic_components.beamsplitters.beamsplitter import Beamsplitter


class Cross(Beamsplitter):

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
        super(Cross, self).__init__(n_bs=n_crosses)

        one = torch.tensor([1+0j])
        zro = torch.tensor([0+0j])

        mC = one if i_loss is None else torch.sqrt(i_loss)

        ct0 = None if crosstalk is None else mC * torch.sqrt(crosstalk)
        ct1 = None if crosstalk is None else mC * torch.sqrt(1 - crosstalk)

        c00 = zro if crosstalk is None else mC * ct0
        c01 = one if crosstalk is None else mC * ct1
        c10 = one if crosstalk is None else mC * ct1
        c11 = zro if crosstalk is None else mC * ct0

        self.register_coefficients(c00, c01, c10, c11)
