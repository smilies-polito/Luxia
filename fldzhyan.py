import torch
from torch import nn
import torch.nn.functional as F
import photonic_components as opCmp


class fldzhyanBlock(nn.Module):
    def __init__(self, n_blocks: int):
        super(fldzhyanBlock, self).__init__()
        self.heater = opCmp.waveguide(n_waveguides=n_blocks)
        self.mmi = opCmp.beamsplitter()

    def forward(self, portsA, portsB):
        outPs = self.heater(portsA)
        y_complex = self.mmi(outPs, portsB)
        return y_complex


class fldzhyanLayer(nn.Module):
    def __init__(self, n_inputs, is_even_layer):
        super(fldzhyanLayer, self).__init__()
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

        self.ph_blocks = fldzhyanBlock(self.n_blocks)

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


class fldzhyanNetwork(nn.Module):
    def __init__(self, n_layers: int, n_inputs: int):
        super(fldzhyanNetwork, self).__init__()

        self.ph_layers = nn.ModuleList()

        for i in range(n_layers):
            if i % 2 == 0:
                self.ph_layers.append(fldzhyanLayer(
                    n_inputs, is_even_layer=True))
            else:
                self.ph_layers.append(fldzhyanLayer(
                    n_inputs, is_even_layer=False))

    def forward(self, x_complex):
        for layer in self.ph_layers:
            x_complex = layer(x_complex)
        return x_complex


class fldzhyanModel(nn.Module):
    def __init__(
            self, n_inputs: int, n_layers: int, n_classes: int, n_networks=2):
        super(fldzhyanModel, self).__init__()
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_networks = n_networks

        self.photodetectors = opCmp.photodetector()

        self.ph_networks = nn.ModuleList()
        self.diag_layer = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.n_networks):
            self.ph_networks.append(fldzhyanNetwork(
                n_layers=self.n_layers, n_inputs=self.n_inputs))

            self.diag_layer.append(nn.Parameter(
                torch.rand(self.n_inputs), requires_grad=True))

            self.biases.append(nn.Parameter(
                torch.rand(self.n_inputs), requires_grad=True))

    def forward(self, x_reals):
        # Add 0 inputs if the architecture N > inputs
        x_size_inputs = x_reals.shape[-1]
        x_size_outputs = self.n_classes
        # add 0 at x keeping important data in the middle
        if x_size_inputs < self.n_inputs:
            toadd0 = self.n_inputs - x_size_inputs
            x_reals = F.pad(x_reals, pad=(
                toadd0//2, (toadd0+1)//2), mode='constant', value=0)

        for i_ph in range(self.n_networks):
            x_complex = torch.complex(x_reals, torch.zeros_like(x_reals))
            y_complex = self.ph_networks[i_ph](x_complex)
            d = self.photodetectors(y_complex)
            pred = d * self.diag_layer[i_ph] + self.biases[i_ph]
            x_reals = pred

        mid_min = self.n_inputs//2 - x_size_outputs//2
        mid_max = self.n_inputs//2 + x_size_outputs//2 + 1
        pred = pred[:, mid_min:mid_max]
        return pred
