import torch
from photonnn import PnnModule


class ElectricalToOptical(PnnModule):
    def __init__(self):
        super(ElectricalToOptical, self).__init__()

    def forward(self, x_reals):
        return torch.complex(x_reals, torch.zeros_like(x_reals))
