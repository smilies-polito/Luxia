import torch
import torch.nn as nn


class ElectricalToOptical(nn.Module):
    def __init__(self):
        super(ElectricalToOptical, self).__init__()

    def forward(self, x_reals):
        return torch.complex(x_reals, torch.zeros_like(x_reals))
