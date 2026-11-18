from luxia.photonic_components.beamsplitters.beamsplitter import Beamsplitter
from luxia.params import MmiParams
import torch


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

    def __init__(self, mmi_params: MmiParams):
        super(Mmi, self).__init__(bs_params=mmi_params)
        one = torch.tensor([1+0j])

        if self._params.insertion_loss is None:
            mC = one / torch.sqrt(torch.tensor(2.0))
        else:
            loss_len = self._params.insertion_loss.size()
            bs_len = torch.Size([self._params.number_of_splitters])
            if loss_len == torch.Size([1]) or loss_len == bs_len:
                mC = torch.sqrt(self._params.insertion_loss)
            else:
                raise ValueError(
                    f"insertion_loss must be a tensor of size 1 "
                    f"or of size n_bs, got {loss_len}")

        c0 = mC * torch.tensor([1+0j])
        c1 = mC * torch.tensor([0+1j])
        c00, c01, c10, c11 = c0, c1, c1, c0

        if self._params.imbalance is not None:
            imb_len = self._params.imbalance.size()
            bs_len = torch.Size([self._params.number_of_splitters])
            if imb_len == torch.Size([1]) or imb_len == bs_len:
                imb0 = torch.sqrt(0.5 + self._params.imbalance)
                imb1 = torch.sqrt(0.5 - self._params.imbalance)
                c00 = mC * imb0
                c01 = mC * imb1
                c10 = mC * imb1
                c11 = mC * imb0
            else:
                raise ValueError(
                    f"imbalance must be a tensor of size 1 "
                    f"or of size n_bs, got {imb_len}")

        self.register_coefficients(c00, c01, c10, c11)
