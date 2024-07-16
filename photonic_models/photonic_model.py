import torch
import torch.nn as nn
from .photonic_components.photodetector import Photodetector
from .photonic_architectures.ph_architecture import PhotonicArchitecture


class RealToComplex(nn.Module):
    def __init__(self):
        super(RealToComplex, self).__init__()

    def forward(self, x_reals):
        return torch.complex(x_reals, torch.zeros_like(x_reals))


class ElectronicFunction(nn.Module):
    def __init__(self, n_inputs: int):
        super(ElectronicFunction, self).__init__()

        self.diag_layer = nn.Parameter(
            torch.rand(n_inputs), requires_grad=True)

        self.bias = nn.Parameter(torch.rand(n_inputs), requires_grad=True)

    def forward(self, x):
        return x * self.diag_layer + self.bias


class PhotonicModel(nn.Module):
    def __init__(self,
                 architecture_name: str,
                 n_inputs: int,
                 n_layers: int,
                 n_classes: int,
                 n_networks: int = 2,
                 mmi_i_losses: [] = None,
                 mmi_imbalances: [] = None,
                 is_matmul: bool = False,
                 ):
        super(PhotonicModel, self).__init__()
        self._n_networks = n_networks

        if is_matmul:
            self.forward = self.matmulForward
            # list of modules that will be used in the matmulForward
            self._real_to_complex = nn.ModuleList()
            self._ph_architecture = nn.ModuleList()
            self._photodetectors = nn.ModuleList()
            self._electronic_functions = nn.ModuleList()
        else:
            self.forward = self.fastForward
            # list of components that will be used in the fastForward
            # the order of the components is important. This list will be used
            # in the sequential model
            component_list = []

        self.low_limit, self.upp_limit = self._getLimits(n_inputs, n_classes)

        for i in range(self._n_networks):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            # create the components of the model
            real_to_complex = RealToComplex()
            ph_architecture = PhotonicArchitecture.create_architecture(
                    name=architecture_name,
                    n_layers=n_layers,
                    n_inputs=n_inputs,
                    mmi_i_losses=i_losses,
                    mmi_imbalances=imbalance,
                    is_matmul=is_matmul
                    )
            photodetectors = Photodetector()
            electronic_function = ElectronicFunction(n_inputs)

            if is_matmul:
                self._real_to_complex.append(real_to_complex)
                self._ph_architecture.append(ph_architecture)
                self._photodetectors.append(photodetectors)
                self._electronic_functions.append(electronic_function)
            else:
                component_list.append(real_to_complex)
                component_list.append(ph_architecture)
                component_list.append(photodetectors)
                component_list.append(electronic_function)

        if not is_matmul:
            self._ph_model = torch.nn.Sequential(*component_list)

    def fastForward(self, x_reals):
        """
        This method is used to perform the forward pass of the photonic model
        using the fast forward approach. This is the approach use the
        sequential method of pytorch to perform the forward pass of the model.
        this method uses the forward method of each component of the model
        including the photonic architecture. This method is vector based and
        use the slicing

        Args:
            x_reals (): input batch (batch_size, n_inputs)

        Returns: y
        """
        y_reals = self._ph_model(x_reals)
        y = y_reals[:, self.low_limit:self.upp_limit]
        return y

    def matmulForward(self, x_reals):
        """
        This method is used to perform the forward pass of the photonic model
        using the matrix multiplication approach. This is the approach used
        gets the transfer function matrix of each photonic network and multiply
        the input by this matrix.

        Args:
            x_reals (): input batch (batch_size, n_inputs)

        Returns:
        """
        for i_ph in range(self._n_networks):
            x_complex = self._real_to_complex[i_ph](x_reals)
            tf_matrix = self._ph_architecture[i_ph].get_tf_matrix()
            y_complex = torch.matmul(x_complex, tf_matrix)
            d = self._photodetectors[i_ph](y_complex)
            x_reals = self._electronic_functions[i_ph](d)

        y = x_reals[:, self.low_limit:self.upp_limit]
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
