import torch
import torch.nn as nn


class Photodetector(nn.Module):
    def __init__(self):
        super(Photodetector, self).__init__()

    def forward(self, waveInput):
        """
        Args:
            waveInput: complex numbers (batch_size, n_inputs)
        Return:
            output: output batch, real numbers (batch_size, n_inputs)
        """
        real_squared_part = torch.square(torch.real(waveInput))
        imaginary_squared_part = torch.square(torch.imag(waveInput))
        output = real_squared_part + imaginary_squared_part

        return output
