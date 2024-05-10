import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Heater(nn.Module):
    def __init__(self, n_heaters:int):
        super(Heater, self).__init__()
        phi = 2*torch.pi*torch.rand(n_heaters)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, waveInput):
        """
        Note: n_heaters can change based on the fact that
        the number of the current layer or block is even or odd.
        
        Args:
            waveInput: all "A" inputs, sampled every two (batch_size, n_heaters)

        Return:
            waveOutput: (batch_size, n_heaters)
        """
        waveOutput = waveInput * torch.exp(1.j*self.phase_shift)
        return waveOutput

    def getPhases(self):
        weights = torch.exp(1.j*self.phase_shift)
        return weights
