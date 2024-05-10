import torch
import torch.nn as nn


# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mmi(nn.Module):
    """
    Multi Mode Interferometer (MMI) beamsplitter. The model follows the 
    following S-matrix:

        A -- --C                | 1 | j |
            X       1/sqrt(2) x |   |   |
        B -- --D                | j | 1 |
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

        matCoeff = 1 if i_loss is None else (1-i_loss)

        matCoeff = matCoeff / torch.sqrt(torch.tensor(2.0))
        c0 = matCoeff * torch.tensor([1+0j], device=device)
        c1 = matCoeff * torch.tensor([0+1j], device=device)

        self._pCoeff0 = c0 if imbalance is None else c0*(1-imbalance)
        self._pCoeff1 = c1 if imbalance is None else c1*(1+imbalance)
        self._pCoeff2 = c0 if imbalance is None else c0*(1+imbalance)
        self._pCoeff3 = c1 if imbalance is None else c1*(1-imbalance)

    def forward(self, portsA, portsB):
        """
        FIX:
        Note: n_A_inputs and n_B_inputs can change based on the fact that
        the number of the current layer or block is even or odd.
        However, n_A_inputs := n_B_inputs := n_mmis always

        Args:
            portsA: all "A" inputs, sampled every two (batch_size, n_A_inputs)
            portsB: all "B" inputs, sampled every two (batch_size, n_B_inputs)

        Return:
            y_0, y_1: tuple of (batch_size, n_mmis), (batch_size, n_mmis)
        """

        y_0 = self._pCoeff0 * portsA + self._pCoeff1 * portsB
        y_1 = self._pCoeff2 * portsA + self._pCoeff3 * portsB

        return y_0, y_1

    def getSMatrix(self):
        if self.i_loss is None and self.imbalance is None:
            s_matrix = [torch.stack([
                torch.tensor([self._pCoeff0, self._pCoeff1], dtype=torch.complex64),
                torch.tensor([self._pCoeff2, self._pCoeff3], dtype=torch.complex64)
            ])]*self.n_mmis
        else:
            s_matrix = torch.stack(
                [
                    torch.stack([self._pCoeff0, self._pCoeff1], dim=-1),
                    torch.stack([self._pCoeff2, self._pCoeff3], dim=-1)
                ],
                dim=-1
            )
        return s_matrix
