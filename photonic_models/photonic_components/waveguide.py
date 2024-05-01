import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Waveguide(nn.Module):
    def __init__(self, n_wg=1):
        super(Waveguide, self).__init__()
        phi = 2*torch.pi*torch.rand(n_wg)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, waveInput):
        waveOutput = waveInput * torch.exp(1.j*self.phase_shift)
        return waveOutput
