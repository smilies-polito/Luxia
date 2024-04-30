import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Beamsplitter(nn.Module):
    """
    Two port beam splitter, for the moment it implments only the matrix of and MMI and it can be visualized as follows:

        A -- --C                  | 1 | j |
            X         1/sqrt(2) x |   |   |
        B -- --D                  | j | 1 |

    """

    def __init__(self):
        super(Beamsplitter, self).__init__()
        one_sqrt2 = 1 / torch.sqrt(torch.tensor(2.0))
        self._element1 = one_sqrt2 * torch.tensor([1+0j], device=device)
        self._element2 = one_sqrt2 * torch.tensor([0+1j], device=device)

    def forward(self, portA, portB):

        y_0 = self._element1 * portA + self._element2 * portB
        y_1 = self._element2 * portA + self._element1 * portB
        # return the elements in an alternative form
        y = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y
