import torch
import torch.nn as nn
from luxia import LxModule


class Heater(LxModule):
    def __init__(self, n_heaters: int = 1):
        super(Heater, self).__init__()
        phi = 2*torch.pi*torch.rand(n_heaters)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, waveInput):
        """
        Args:
            waveInput: inputs sampled every two (batch_size, n_heaters)

        Return:
            waveOutput: (batch_size, n_heaters)
        """
        return waveInput * torch.exp(1.j*self.phase_shift)

    def getPhases(self):
        """
        Return the phase shifts of the heaters

        Args:
            None

        Return:
            torch.tensor: phase shifts of the heaters
        """
        return torch.exp(1.j*self.phase_shift)
