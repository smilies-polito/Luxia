import torch
from torch import nn
from ..photonic_components.heater import Heater
from ..photonic_components.mzis.single_heater import SingleHeaterMzi
from ..util.util_functions import diag_matrix_alternating_tensor_with_ones


class ClementsBlock(nn.Module):
    """ Create the basic model of the clements architecture, the basic
        block is a phase shifter followed by a mzi module

                 _____ MZI _____
           wg0  |    wg1        |
        A -[w]--|--- --[w]-- ---|-C
                |   X       X   |
        B ------|--- ------- ---|-D
                | mmi0     mmi1 |
                 ---------------

        Args:
        n_blocks: number of clements blocks to create dowsizes
        mmi_i_losses: Tensor of insertion losses of the bs0 and bs1
        mmi_imbalances: Tensor of imbalances of the mmi0 and mmi1
    """

    def __init__(self, n_blocks: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(ClementsBlock, self).__init__()
        self._heater = Heater(n_heaters=n_blocks)
        self._mzi = SingleHeaterMzi(n_mzi=n_blocks,
                                    mmi_i_losses=mmi_i_losses,
                                    mmi_imbalances=mmi_imbalances)

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        y_0, y_1 = self._mzi(outPs, portsB)

        # return the elements in an alternative form
        y_complex = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y_complex

    def get_tf_matrix(self):
        # Get the phases of the heaters wg0
        heater_phases = self._heater.getPhases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(
                heater_phases)
        # Get the transfer function matrix of the mzi
        mzi_tf_matrix = self._mzi.get_tf_matrix()
        # Multiply the heater diagonal with the mzi transfer function matrix
        tf_matrix = torch.matmul(heater_diag_matrix, mzi_tf_matrix)
        return tf_matrix


class ClementsLayer(nn.Module):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A _________________ C
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B _________________ C

        ========================================================
        Fig. 2 Odd number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A _________________ C
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A _________________ C       A ------ ------- ---D

    Args:
      n_inputs: number of inputs of the layer, from here it can be said
      if it is an even or and odd layer
      is_even_layer: What type of layer to be created
      mmi_i_losses: Tesor of all the losses of the blocks, each block
      contains, 2 tensors of losses since each block uses 2 mmis
      mmi_imbalances: Tesor of all the imbalances of the blocks, each
      block contains, 2 tensors of imbalance since each block uses 2 mmis
    """

    def __init__(self, n_inputs: int,
                 is_even_layer: bool = True,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(ClementsLayer, self).__init__()
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._n_blocks = n_inputs // 2
        self._is_even_input = False

        if self._n_inputs % 2 == 0:
            self._is_even_input = True
            if self._is_even_layer is False:
                self._n_blocks = self._n_blocks - 1

        self._ph_blocks = ClementsBlock(
            n_blocks=self._n_blocks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
        )

    def forward(self, x_complex):
        y_complex = torch.zeros_like(x_complex)
        if self._is_even_input is True:
            if self._is_even_layer is True:
                # figure 1.a
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex = self._ph_blocks(x_complex_0, x_complex_1)
            else:
                # figure 1.b
                # Pass to the output all the first elements
                y_complex[:, 0] = x_complex[:, 0]
                # Pass to the output all the last elements
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, 1:-1]
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:-1] = self._ph_blocks(x_complex_0, x_complex_1)
        else:
            if self._is_even_layer is True:
                # figure 2.a
                # Pass to the output all the last elements
                y_complex[:, -1] = x_complex[:, - 1]
                x_complex = x_complex[:, :-1]
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, :-1] = self._ph_blocks(x_complex_0, x_complex_1)
            else:
                # figure 2.b
                # Pass to the output all the first elements
                y_complex[:, 0] = x_complex[:, 0]
                x_complex = x_complex[:, 1:]
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:] = self._ph_blocks(x_complex_0, x_complex_1)
        return y_complex

    def get_tf_matrix(self):
        if self._is_even_input is True:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix, ones)
        else:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(tf_matrix, ones)
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix)
        return tf_matrix


class ClementsArchitecture(nn.Module):
    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_fastforward: bool = False
                 ):
        super(ClementsArchitecture, self).__init__()

        if is_fastforward:
            layer_list = []
        else:
            self._layer_matrices = nn.ParameterList()

        for i in range(n_layers):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = ClementsLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance
            )

            if is_fastforward:
                layer_list.append(layer)
            else:
                self._layer_matrices.append(layer.get_tf_matrix())

        if is_fastforward:
            self._ph_layers = torch.nn.Sequential(*layer_list)

    def forward(self, x_complex):

        """
        Args:
            x_complex (): input batch of complex numbers (batch_size, n_inputs)

        Returns:
        """
        return self._ph_layers(x_complex)

    def get_tf_matrix(self):
        # tf_matrix = None
        # for layer in self._ph_layers:
        #     if tf_matrix is None:
        #         tf_matrix = layer.getTFMatrix()
        #     else:
        #         tf_matrix = torch.matmul(tf_matrix, layer.getTFMatrix())
        tf_matrix = torch.linalg.multi_dot(
            tuple(
                [self._layer_matrices[i]
                    for i in range(len(self._layer_matrices))
                 ]
            )
        )
        return tf_matrix
