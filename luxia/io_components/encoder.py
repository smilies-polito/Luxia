import torch
from luxia import LxModule


class Encoder(LxModule):
    def __init__(self, n_inputs: int, n_features: int):
        super(Encoder, self).__init__()

        self.encode = torch.nn.Linear(n_features, n_inputs)

    def forward(self, x_complex):
        x_reals = self.encode(x_complex)
        return torch.clamp(x_reals, 0.0, 0.1)
