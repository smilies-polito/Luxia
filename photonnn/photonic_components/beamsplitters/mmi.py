import torch
from photonnn.photonic_components.beamsplitters.beamsplitter import Beamsplitter


class Mmi(Beamsplitter):
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
                 n_mmis: int = 1,
                 i_loss: torch.Tensor = None,
                 imbalance: torch.Tensor = None
                 ):
        """
        Args:
            n_mmis: number of MMI beamsplitters, to set one after the other
            downsides
            i_loss: Tesor of size n_mmis, this is required for setting the
            insertion loss, it can be a scalar or a vector
            imbalance: Tensor of size n_mmis, this is required for setting the
            imbalance
        """
        # Constructor of the Class Mmi
        super(Mmi, self).__init__(n_bs=n_mmis)

        one = torch.tensor([1+0j])
        mC = one/torch.sqrt(torch.tensor(2.0)
                            ) if i_loss is None else torch.sqrt(i_loss)

        c0 = mC * torch.tensor([1+0j])
        c1 = mC * torch.tensor([0+1j])

        imb0 = None if imbalance is None else torch.sqrt(0.5+imbalance)
        imb1 = None if imbalance is None else torch.sqrt(0.5-imbalance)

        c00 = c0 if imbalance is None else mC*imb0
        c01 = c1 if imbalance is None else mC*imb1
        c10 = c1 if imbalance is None else mC*imb0
        c11 = c0 if imbalance is None else mC*imb1

        self.register_coefficients(c00, c01, c10, c11)
