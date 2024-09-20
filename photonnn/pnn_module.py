import torch


class PnnModule(torch.nn.Module):
    def __init__(self):
        super(PnnModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError
