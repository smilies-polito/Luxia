import torch
from torch import nn
from .photonic_components.waveguide import Waveguide
from .photonic_components.mzi import Mzi
from .photonic_components.photodetector import Photodetector


class ClementsBlock(nn.Module):
    def __init__(self, n_blocks: int):
        super(ClementsBlock, self).__init__()
        self.heater = Waveguide(n_wg=n_blocks)
        self.mzi = Mzi(n_mzi=n_blocks)

    def forward(self, portsA, portsB):

        outPs = self.heater(portsA)
        y_0, y_1 = self.mzi(outPs, portsB)

        # return the elements in an alternative form
        y_complex = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y_complex


class ClementsLayer(nn.Module):
    def __init__(self, n_inputs, is_even_layer):
        super(ClementsLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_blocks = n_inputs // 2
        self.is_even_layer = is_even_layer

        if self.n_inputs % 2 == 0:
            is_e = True
        else:
            is_e = False

        self.is_even_input = is_e

        if self.is_even_layer is False:
            self.n_blocks = self.n_blocks - 1

        self.ph_blocks = ClementsBlock(self.n_blocks)

    def forward(self, x_complex):
        y_complex = torch.zeros_like(x_complex)
        if self.is_even_input is True:
            if self.is_even_layer is False:
                y_complex[:, 0] = x_complex[:, 0]
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, 1:-1]
                x_complex_0 = x_complex[:, ::2]
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:-1] = self.ph_blocks(x_complex_0, x_complex_1)
            else:
                x_complex_0 = x_complex[:, ::2]
                x_complex_1 = x_complex[:, 1::2]
                y_complex = self.ph_blocks(x_complex_0, x_complex_1)
        else:
            if self.is_even_layer is True:
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, :-1]
                x_complex_0 = x_complex[:, ::2]
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, :-1] = self.ph_blocks(x_complex_0, x_complex_1)
            else:
                y_complex[:, 0] = x_complex[:, 0]
                x_complex = x_complex[:, 1:]
                x_complex_0 = x_complex[:, ::2]
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:] = self.ph_blocks(x_complex_0, x_complex_1)
        return y_complex


class ClementsNetwork(nn.Module):
    def __init__(self, n_layers: int, n_inputs: int):
        super(ClementsNetwork, self).__init__()

        self.ph_layers = nn.ModuleList()

        for i in range(n_layers):
            if i % 2 == 0:
                self.ph_layers.append(ClementsLayer(
                    n_inputs, is_even_layer=True))
            else:
                self.ph_layers.append(ClementsLayer(
                    n_inputs, is_even_layer=False))

    def forward(self, x_complex):
        for layer in self.ph_layers:
            x_complex = layer(x_complex)
        return x_complex


class ClementsModel(nn.Module):
    def __init__(
            self, n_inputs: int, n_layers: int, n_classes: int, n_networks=2):
        super(ClementsModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_networks = n_networks

        self.photodetectors = Photodetector()

        self.ph_networks = nn.ModuleList()
        self.diag_layer = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.n_networks):
            self.ph_networks.append(ClementsNetwork(
                n_layers=self.n_layers, n_inputs=self.n_inputs))

            self.diag_layer.append(nn.Parameter(
                torch.rand(self.n_inputs), requires_grad=True))

            self.biases.append(nn.Parameter(
                torch.rand(self.n_inputs), requires_grad=True))

    def forward(self, x):

        # Add features with zeros if the architecture N > inputs
        x_reals = self._resizeX(x)

        for i_ph in range(self.n_networks):
            x_complex = torch.complex(x_reals, torch.zeros_like(x_reals))
            y_complex = self.ph_networks[i_ph](x_complex)
            d = self.photodetectors(y_complex)
            x_reals = d * self.diag_layer[i_ph] + self.biases[i_ph]

        lower_limit, upper_limit = self._getLimits()
        y = x_reals[:, lower_limit:upper_limit]
        return y

    def _resizeX(self, x):
        # Check if the number of features exceeds the capacity of the photonic
        # network
        x_size = x.shape[-1]
        if x_size > self.n_inputs:
            print(
                "The number of features exceeds the capacity of the photonic"
                f"network. The maximum allowed number is : {self.n_inputs}")
            exit()

        # Resize the input data if the number of features is less than the
        # number of inputs of the photnic network
        if x_size < self.n_inputs:
            x_resized = torch.zeros(
                (x.shape[0], self.n_inputs), dtype=x.dtype, device=x.device)

            start_limit = (self.n_inputs - x_size) // 2
            end_limit = start_limit + x_size
            x_resized[..., start_limit:end_limit] = x
            return x_resized
        return x

    def _getLimits(self):
        lower_limit = self.n_inputs//2 - (self.n_classes//2)
        upper_limit = self.n_inputs//2 + self.n_classes//2

        if self.n_classes % 2 != 0:
            upper_limit = upper_limit + 1

        return lower_limit, upper_limit
