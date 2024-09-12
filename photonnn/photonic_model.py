import torch
import torch.nn as nn
from .photonic_architectures.pnn_architectures import PnnArchitectures
from .electrical_to_optical import ElectricalToOptical
from .optical_to_electrical import OpticalToElectrical


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
            # list of submodules that will be used in the matmulForward
            self._electrical_to_optical = nn.ModuleList()
            self._ph_architecture = nn.ModuleList()
            self._optical_to_electrical = nn.ModuleList()
        else:
            self.forward = self.fastForward
            # list of submodules that will be used in the fastForward
            # the order of the components is important. This list will be used
            # in the sequential model
            component_list = []

        self.low_limit, self.upp_limit = self._getLimits(n_inputs, n_classes)

        for i in range(self._n_networks):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            # create the components of the model
            electrical_to_optical = ElectricalToOptical()
            ph_architecture = PnnArchitectures.create_architecture(
                    name=architecture_name,
                    n_layers=n_layers,
                    n_inputs=n_inputs,
                    mmi_i_losses=i_losses,
                    mmi_imbalances=imbalance,
                    is_matmul=is_matmul
                    )
            optical_to_electrical = OpticalToElectrical(n_inputs)

            if is_matmul:
                self._electrical_to_optical.append(electrical_to_optical)
                self._ph_architecture.append(ph_architecture)
                self._optical_to_electrical.append(optical_to_electrical)
            else:
                component_list.append(electrical_to_optical)
                component_list.append(ph_architecture)
                component_list.append(optical_to_electrical)

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
            x_complex = self._electrical_to_optical[i_ph](x_reals)
            tf_matrix = self._ph_architecture[i_ph].get_tf_matrix()
            y_complex = torch.matmul(x_complex, tf_matrix)
            x_reals = self._optical_to_electrical[i_ph](y_complex)

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
