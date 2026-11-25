import torch
import torch.nn as nn
from luxia import LxModule
from luxia.util.log_config import lx_log


class ElectricalAmplifier(LxModule):
    def __init__(self,
                 n_amplifiers: int = 1,
                 gain: float = 1.0,
                 trainable: bool = True,
                 debug: bool = False
                 ):
        """
        Electrical Amplifier Module
        This module amplifies the input analog signal by a gain factor.
        The gain can be either fixed or trainable.

        Args:
            n_amplifiers: Number of amplifiers in the module/Tensor
                shape (n_amplifiers, batch_size)
            gain: Fixed gain value or initial value for trainable gain
            trainable: If True, the gain is a trainable parameter; if False,
                it is fixed to the gain value.
            debug: If True, enables debug logging
        """
        super(ElectricalAmplifier, self).__init__(debug=debug)
        if not trainable:
            self.gain = nn.Parameter(gain, requires_grad=False)
        else:
            self.gain = nn.Parameter(
                torch.rand(n_amplifiers), requires_grad=True
            )

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(f"x: \n{x_analog}")
        
        y = x_analog * self.gain
        
        if self._debug:
            lx_log.debug(f"y: \n{y}")

        return y
