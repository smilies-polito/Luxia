# Photontorch
import photontorch as pt
# Python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ElectronicFunction(torch.nn.Module):

    def __init__(self, size):
        super(ElectronicFunction, self).__init__()
        self.bias = torch.nn.Parameter(
            torch.rand(1, 1, size, 1), requires_grad=True)

        self.diag = torch.nn.Parameter(
            torch.rand(1, 1, size, 1), requires_grad=True)

    def forward(self, x):
        modrelu = x * self.diag + self.bias
        return modrelu


class PtClementsModel(torch.nn.Module):
    def __init__(
            self, n_inputs: int, n_layers: int, n_classes: int,
            capacity=2, n_networks=2):

        super(PtClementsModel, self).__init__()

        self.networks = n_networks
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        self.clementsNet = torch.nn.ModuleList()
        self.electronicFunc = torch.nn.ModuleList()

        for i in range(n_networks):
            self.clementsNet.append(
                pt.ClementsNxN(
                    N=n_inputs, capacity=capacity).terminate().to(device))

            self.electronicFunc.append(
                ElectronicFunction(n_inputs).to(device))

    def forward(self, x):

        # Add features with zeros if the architecture N > inputs
        x_reals = self._resizeX(x)

        # photontorch requres a tensor in the following order "c", "t", "w", "s", "b"
        # set features, batch
        x_reals = x_reals.T
        # add 2 new dimentions
        x_reals = x_reals[None, None, ...]

        for i in range(self.networks):
            # make input complex, using stack, photontorch does not support complex numbers
            x_complex = torch.stack([x_reals, torch.zeros_like(x_reals)], 0)
            # photontorch requires named tensors
            x_complex = x_complex.rename("c", "t", "w", "s", "b")
            x_reals = self.clementsNet[i](x_complex)
            x_reals = self.electronicFunc[i](x_reals)

        y = x_reals[0, 0, :, :]
        y = y.T

        lower_limit, upper_limit = self._getLimits()
        y = y[:, lower_limit:upper_limit]
        return y

    def _resizeX(self, x):
        # Check if the number of features exceeds the capacity of the photonic network
        x_size = x.shape[-1]
        if x_size > self.n_inputs:
            print(
                f"The number of features exceeds the capacity of the photonic network. The maximum allowed number is: {self.n_inputs}")
            exit()

        # Resize the input data if the number of features is less than the network capacity
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
