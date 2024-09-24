import torch
from photonnn import PnnModule


class OpticalToElectrical(PnnModule):
    def __init__(self):
        super(OpticalToElectrical, self).__init__()

    def forward(self, x_complex):
        """
        Args:
            x_complex: complex numbers (batch_size, n_inputs)
        Return:
            output: output batch, real numbers (batch_size, n_inputs)
        """
        real_squared_part = torch.square(torch.real(x_complex))
        imaginary_squared_part = torch.square(torch.imag(x_complex))
        return real_squared_part + imaginary_squared_part
