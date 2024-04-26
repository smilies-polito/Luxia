import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class photodetector(nn.Module):
    def __init__(self):
        super(photodetector, self).__init__()

    def forward(self, waveInput):
        real_squared_part = torch.square(torch.real(waveInput))
        imaginary_squared_part = torch.square(torch.imag(waveInput))
        output = real_squared_part + imaginary_squared_part
        return output


class waveguide(nn.Module):
    def __init__(self, n_waveguides):
        super(waveguide, self).__init__()
        phi = 2*torch.pi*torch.rand(n_waveguides)
        self.phase_shift = nn.Parameter(phi, requires_grad=True)

    def forward(self, waveInput):
        waveOutput = waveInput * torch.exp(1.j*self.phase_shift)
        return waveOutput


# The matrix of an mmi can be visualized as follows:
#                  | 1 | j |
#      1/sqrt(2) x |   |   |
#                  | j | 1 |
#
class beamsplitter(nn.Module):
    def __init__(self):
        super(beamsplitter, self).__init__()
        one_sqrt2 = 1 / torch.sqrt(torch.tensor(2.0))
        self.element1 = one_sqrt2 * torch.tensor([1+0j], device=device)
        self.element2 = one_sqrt2 * torch.tensor([0+1j], device=device)

    def forward(self, portA, portB):

        y_0 = self.element1 * portA + self.element2 * portB
        y_1 = self.element2 * portA + self.element1 * portB

        # return the elements in an alternative form
        y = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y
