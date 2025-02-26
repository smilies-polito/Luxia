import torch


class LxModule(torch.nn.Module):
    def __init__(self, debug=False):
        super(LxModule, self).__init__()
        self.debug = debug

    def forward(self, x):
        raise NotImplementedError
