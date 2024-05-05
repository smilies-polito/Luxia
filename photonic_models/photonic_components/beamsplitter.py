import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Beamsplitter(nn.Module):
    """
    Two port beam splitter.The model implements different models dependig on 
    different parameters.
    """

    def __init__(self,
                 n_bs: int,
                 i_loss: torch.Tensor = None,
                 imbalance: torch.Tensor = None,
                 crosstalk: torch.Tensor = None,
                 bs_type="mmi"):
        """
        Constructor of the Class Beamsplitter,

        Args:
            bs_type (): type of beamsplitter, you can chose between:
                mmi, cross or dielectric
            n_bs: number of beamsplitters, to set one after the other downsides
            i_loss: Tesor of size n_bs, this is required for setting indepent
                losses to each beamsplitters
            imbalance: Tensor of size n_bs, this is required for setting
                indepent imbalance to each beamsplitters.
            crosstalk: Tensor of size n_bs, this term is only valid for cross
                type beamsplitter

        mmi:
        A -- --C                | 1 | j |
            X       1/sqrt(2) x |   |   |
        B -- --D                | j | 1 |

        cross:
        A -- --C                | 1 | j |
            X               1 x |   |   |
        B -- --D                | j | 1 |

        dielectric:
        A -- --C                | 1 | 1 |
            X       1/sqrt(2) x |   |   |
        B -- --D                | 1 |-1 |

        """
        super(Beamsplitter, self).__init__()
        if i_loss is None:
            matCoeff = 1
        else:
            matCoeff = (1 - i_loss)

        if bs_type == "mmi":
            matCoeff = matCoeff / torch.sqrt(torch.tensor(2.0))
            c0 = matCoeff * torch.tensor([1+0j], device=device)
            c1 = matCoeff * torch.tensor([0+1j], device=device)
        if bs_type == "cross":
            c0 = matCoeff * torch.tensor([1+0j], device=device)
            c1 = matCoeff * torch.tensor([0+1j], device=device)
        if bs_type == "dielectric":
            matCoeff = matCoeff / torch.sqrt(torch.tensor(2.0))
            c0 = matCoeff * torch.tensor([1+0j], device=device)

        if bs_type == "dielectric":
            coeff0 = c0
            coeff1 = c0
            coeff2 = c0
            coeff3 = -c0
        else:
            coeff0 = c0
            coeff1 = c1
            coeff2 = c0
            coeff3 = c1

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
