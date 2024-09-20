from .pnn_module import PnnModule
import torch
import torch.nn as nn


class PnnAccelerator(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 pnn_module_list: [PnnModule] = None,
                 ):
        super(PnnAccelerator, self).__init__()
        self._n_networks = n_networks
        self.low_limit, self.upp_limit = self._getLimits(n_inputs, n_classes)
        self._ph_model = torch.nn.Sequential(*pnn_module_list)

    def forward(self, x_reals):
        """
        This method is used to perform the forward pass of the photonic model.
        This is the approach use the sequential method of pytorch to perform the
        forward pass of the model.
        Args:
            x_reals (): input batch (batch_size, n_inputs)

        Returns: y
        """
        y_reals = self._ph_model(x_reals)
        y = y_reals[:, self.low_limit:self.upp_limit]
        return y

    def _getLimits(self, n_inputs, n_classes):
        """
        given the number of inputs and the number of classes it returns the
        lower and upper limits, which are used to slice the output of the model
        this limits are used to get the middle part of the output of the model

        Args:
            n_inputs (int): number of inputs
            n_classes (int): number of classes

        Returns: lower_limit, upper_limit
        """
        lower_limit = n_inputs//2 - (n_classes//2)
        upper_limit = n_inputs//2 + n_classes//2

        if n_classes % 2 != 0:
            upper_limit = upper_limit + 1

        return lower_limit, upper_limit
