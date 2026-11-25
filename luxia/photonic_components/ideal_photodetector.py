import torch
from luxia import LxModule
from luxia.util.log_config import lx_log


class IdealPhotodetector(LxModule):
    def __init__(self, debug=False):
        super(IdealPhotodetector, self).__init__(debug=debug)

    def forward(self, waveInput):
        """
        Args:
            waveInput: complex numbers (batch_size, n_inputs)
        Return:
            output: output batch, real numbers (batch_size, n_inputs)
        """
        real_squared_part = torch.square(torch.real(waveInput))
        imaginary_squared_part = torch.square(torch.imag(waveInput))
        optical_power = real_squared_part + imaginary_squared_part
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(f"Wave input: \n{waveInput}")
            lx_log.debug(f"Real part squared: \n{real_squared_part}")
            lx_log.debug(f"Imaginary part squared: \n{imaginary_squared_part}")
            lx_log.debug(f"|Y|^2 ->: \n{optical_power}")
            lx_log.debug(f"|Y|: \n{torch.abs(waveInput)}")
            lx_log.debug(f"Total Optical Power: {torch.sum(optical_power)}")
        return optical_power
