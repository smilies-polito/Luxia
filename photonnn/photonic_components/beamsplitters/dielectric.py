import torch
from .beamsplitter import Beamsplitter


class Dielectric(Beamsplitter):
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
        super(Dielectric, self).__init__(n_bs=n_bs)
        one = torch.tensor([1+0j])
        mC = one if i_loss is None else torch.sqrt(one - i_loss)
        mC = mC / torch.sqrt(torch.tensor(2.0))

        c00 = mC if imbalance is None else mC*(1-imbalance)
        c01 = mC if imbalance is None else mC*(1+imbalance)
        c10 = mC if imbalance is None else mC*(1+imbalance)
        c11 = -mC if imbalance is None else -mC*(1-imbalance)

        self.set_coefficients(c00, c01, c10, c11)
