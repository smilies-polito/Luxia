import torch


class LxModule(torch.nn.Module):
    def __init__(self, debug=False):
        super(LxModule, self).__init__()
        self._debug = debug

    def forward(self, x):
        raise NotImplementedError

    def set_debug_mode(self, debug):
        self._debug = debug
