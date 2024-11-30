import torch
from luxia import LxModule


class ElectricalToOptical(LxModule):
    def __init__(self):
        super(ElectricalToOptical, self).__init__()

    def forward(self, x_reals):
        return torch.complex(x_reals, torch.zeros_like(x_reals))
