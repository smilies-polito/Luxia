import torch
from neurophox.torch import RMTorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ElectronicFunction(torch.nn.Module):
    def __init__(self, n_inputs):
        super(ElectronicFunction, self).__init__()

        self.diag_layer = torch.nn.Parameter(
            torch.rand(n_inputs), requires_grad=True
        )

        self.bias = torch.nn.Parameter(
            torch.rand(n_inputs), requires_grad=True
        )

    def forward(self, x):
        # photodetector
        x = x[0] ** 2 + x[1] ** 2
        # electronic function
        return x * self.diag_layer + self.bias


class NxClementsModel(torch.nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 capacity=2,
                 n_networks=2):

        super(NxClementsModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.networks = n_networks

        self.clementsNet = torch.nn.ModuleList()
        self.electronicFunc = torch.nn.ModuleList()

        for i in range(n_networks):
            # rm = RMTorch(units=n_inputs, num_layers=n_layers)
            rm = RMTorch(units=n_inputs)
            # rm = clements_decomposition(rm)
            self.clementsNet.append(rm.to(device))

            self.electronicFunc.append(
                ElectronicFunction(n_inputs).to(device))

    def forward(self, x):

        # Add features with zeros if the architecture N > inputs
        x_reals = self._resizeX(x)
        for i in range(self.networks):
            x_complex = torch.stack(
                [x_reals, torch.zeros_like(x_reals)], dim=0)

            x_complex = self.clementsNet[i](x_complex)
            x_reals = self.electronicFunc[i](x_complex)
        y = x_reals
        lower_limit, upper_limit = self._getLimits()
        y = y[:, lower_limit:upper_limit]
        return y

    def _resizeX(self, x):
        # Check if n features exceeds the capacity of the photonic network
        x_size = x.shape[-1]
        if x_size > self.n_inputs:
            print("The number of features exceeds the capacity of the photonic"
                  "network. The maximum allowed number is: ", self.n_inputs)
            exit()

        # Resize the input data if n features is less than the network capacity
        if x_size < self.n_inputs:
            x_resized = torch.zeros(
                (x.shape[0], self.n_inputs), dtype=x.dtype, device=x.device)

            start_limit = (self.n_inputs - x_size) // 2
            end_limit = start_limit + x_size
            x_resized[..., start_limit:end_limit] = x
            return x_resized
        return x

    def _getLimits(self):
        upper_limit = self.n_inputs//2 + self.n_classes//2
        lower_limit = self.n_inputs//2 - (self.n_classes//2)

        if self.n_classes % 2 != 0:
            upper_limit = upper_limit + 1

        return lower_limit, upper_limit
