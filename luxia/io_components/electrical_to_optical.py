import torch
from luxia import LxModule


class ElectricalToOptical(LxModule):
    def __init__(self, debug=False):
        super(ElectricalToOptical, self).__init__(debug=debug)

    def forward(self, x_reals):
        x_complex = torch.complex(x_reals, torch.zeros_like(x_reals))
        if self.debug:
            print(" - EO input: ", x_reals)
            print(" - EO output: ", x_complex)
        return x_complex
