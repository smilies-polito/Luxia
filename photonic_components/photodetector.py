import torch
import torch.nn as nn

# Select GPU if available, else fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Photodetector(nn.Module):
    def __init__(self):
        super(Photodetector, self).__init__()

    def forward(self, waveInput):
        real_squared_part = torch.square(torch.real(waveInput))
        imaginary_squared_part = torch.square(torch.imag(waveInput))
        output = real_squared_part + imaginary_squared_part
        return output
