import torch


class LxModule(torch.nn.Module):
    def __init__(self):
        super(LxModule, self).__init__()

    def forward(self, x):
        raise NotImplementedError
