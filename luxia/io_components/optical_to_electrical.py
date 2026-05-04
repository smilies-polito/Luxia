import torch

from luxia import LxModule


class OpticalToElectrical(LxModule):
    """Convert a complex optical tensor to a real electrical power tensor."""

    def __init__(self, debug: bool = False):
        super(OpticalToElectrical, self).__init__(debug=debug)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        """Compute optical intensity from a complex input tensor.

        Args:
            x_complex: Complex-valued tensor of shape (batch_size, n_inputs).

        Returns:
            torch.Tensor: Real-valued tensor of shape (batch_size, n_inputs).
        """
        real_squared_part = torch.square(torch.real(x_complex))
        imaginary_squared_part = torch.square(torch.imag(x_complex))
        return real_squared_part + imaginary_squared_part
