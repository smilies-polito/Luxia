import torch
from luxia import LxModule
from luxia.util.log_config import lx_log


class ElectricalToOptical(LxModule):
    def __init__(self, debug=False):
        super(ElectricalToOptical, self).__init__(debug=debug)

    def forward(self, x_reals):
        x_complex = torch.complex(x_reals, torch.zeros_like(x_reals))
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(" Input x_reals: %s", x_reals)
            lx_log.debug(" Output x_complex: %s", x_complex)
        return x_complex
