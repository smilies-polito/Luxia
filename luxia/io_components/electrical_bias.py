import torch
import torch.nn as nn
from luxia import LxModule
from luxia.util.log_config import lx_log


class ElectricalBias(LxModule):
    def __init__(self,
                 n_adders: int = 1,
                 bias: float = 0.0,
                 trainable: bool = True,
                 debug: bool = False
                 ):
        """
        Electrical Bias Adder Module
        This module adds a bias to the input analog signal.

        Args:
            n_adders: Number of bias adders in the module/Tensor
                shape (n_adders, batch_size)
            bias: Fixed bias value or rnd value for trainable bias
            trainable: If True, the bias is a trainable parameter; if False,
                it is fixed to the bias value.
            debug: If True, enables debug logging
        """
        super(ElectricalBias, self).__init__(debug=debug)
        if not trainable:
            self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        else:
            self.bias = nn.Parameter(torch.rand(n_adders), requires_grad=True)

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(f"x: \n{x_analog}")

        y = x_analog + self.bias

        if self._debug:
            lx_log.debug(f"y: \n{y}")

        return y
