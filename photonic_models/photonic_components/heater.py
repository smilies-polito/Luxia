import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Heater(nn.Module):
    def __init__(self, n_heaters=1):
        super(Heater, self).__init__()
        phi = 2*torch.pi*torch.rand(n_heaters)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, waveInput):
        waveOutput = waveInput * torch.exp(1.j*self.phase_shift)
        return waveOutput

    def getPhases(self):
        weights = torch.exp(1.j*self.phase_shift)
        return weights
