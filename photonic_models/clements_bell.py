import torch
from torch import nn
from .photonic_components.heater import Heater
from .photonic_components.mzi_dual import MziDual
from .photonic_components.photodetector import Photodetector


class ClementsBellBlock(nn.Module):
    """ Create the basic model of the clements architecture, the basic
        block is a phase shifter followed by a mzi module

                 __ MZI DUAL ___
                |    wg1        |
        A ---------- --[w]-- ---|-C
                |   X       X   |
        B ---------- --[w]-- ---|-D
                |         wg    |
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
        super(ClementsBellBlock, self).__init__()
        self._mzi = MziDual(n_mzi=n_blocks,
                            mmi_i_losses=mmi_i_losses,
                            mmi_imbalances=mmi_imbalances)

    def forward(self, portsA, portsB):

        y_0, y_1 = self._mzi(portsA, portsB)
        # return the elements in an alternative form
        y_complex = torch.stack([y_0, y_1], axis=2).view(y_0.shape[0], -1)
        return y_complex

    def get_tf_matrix(self):
        tf_matrix = self._mzi.get_tf_matrix()
        return tf_matrix


class ClementsBellLayer(nn.Module):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:            b) Odd Layer:
            mzi_dual                 wg
        A -- --[w]-- ---C       A -----[w]------C
            X       X
        B -- --[w]-- ---D       B -- --[w]-- ---C
                                    X       X
        A -- --[w]-- ---C       A -- --[w]-- ---D
            X       X
        B -- --[w]-- ---D       B -- --[w]-- ---C
                                    X       X
        A -- --[w]-- ---C       A -- --[w]-- ---D
            X       X
        B -- --[w]-- ---D       B _____[w]_____ C
                                          wg

        ========================================================
        Fig. 2 Odd number of Inputs
          a) Even Layer:            b) Odd Layer:
             mzi_dual                wg
        A -- --[w]-- ---C       A -----[w]------C
            X       X
        B -- --[w]-- ---D       B -- --[w]-- ---C
                                    X       X
        A -- --[w]-- ---C       A -- --[w]-- ---D
            X       X
        B -- --[w]-- ---D       B -- --[w]-- ---C
                                    X       X
        A -- --[w]-- ---C       A -- --[w]-- ---D
            X       X
        B -- --[w]-- ---D       B -- --[w]-- ---C
                                    X       X
        A _____[w]______C       A -- --[w]-- ---D
                  wg
      Fig. 3 Start and last layer
             wg
        A -- --[w]-- ---C

        B -- --[w]-- ---D

        A -- --[w]-- ---C

        B -- --[w]-- ---D

        A -- --[w]-- ---C

        B -- --[w]-- ---D

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
        super(ClementsBellLayer, self).__init__()
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._n_blocks = n_inputs // 2
        self._is_even_input = False
        # the number of heaters could be 0, 1 or 2
        self._heaters = nn.ModuleList()

        if self._n_inputs % 2 == 0:
            self._is_even_input = True
            # number of waveguides to create
            nhts = 0
            if self._is_even_layer is False:
                self._n_blocks = self._n_blocks - 1
                nhts = 2
        else:
            nhts = 1

        self._ph_blocks = ClementsBellBlock(
            n_blocks=self._n_blocks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
        )

        # create the top and bottom heaters
        for i in range(nhts):
            self._heaters.append(Heater(1))

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
                # Pass to through the top heater all the first elements
                y_complex[:, 0] = self._heaters[0](x_complex[:, 0])
                # Pass to the bottom heater all the last elements
                y_complex[:, -1] = self._heaters[1](x_complex[:, - 1])
                x_complex = x_complex[:, 1:-1]
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:-1] = self._ph_blocks(x_complex_0, x_complex_1)
        else:
            if self._is_even_layer is True:
                # figure 2.a
                # Pass to the bottom heater all the last elements
                y_complex[:, -1] = self._heaters[0](x_complex[:, - 1])
                x_complex = x_complex[:, :-1]
                # Create a tensor with all As (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all Bs (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, :-1] = self._ph_blocks(x_complex_0, x_complex_1)
            else:
                # figure 2.b
                # Pass to through the top heater all the first elements
                y_complex[:, 0] = self._heaters[0](x_complex[:, 0])
                x_complex = x_complex[:, 1:]
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:] = self._ph_blocks(x_complex_0, x_complex_1)
        return y_complex

    def get_tf_matrix(self):
        if self._is_even_input is True:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                top_heater_phase = self._heaters[0].getPhases()
                bottom_heater_phase = self._heaters[1].getPhases()
                tf_matrix = torch.block_diag(
                    top_heater_phase, tf_matrix, bottom_heater_phase
                )
        else:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                bottom_heater_phase = self._heaters[0].getPhases()
                tf_matrix = torch.block_diag(tf_matrix, bottom_heater_phase)
            else:
                tf_matrix = self._ph_blocks.get_tf_matrix()
                top_heater_phase = self._heaters[0].getPhases()
                tf_matrix = torch.block_diag(top_heater_phase, tf_matrix)
        return tf_matrix


class ClementsBellArchitecture(nn.Module):
    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_fastforward: bool = False
                 ):
        super(ClementsBellArchitecture, self).__init__()
        ph_first_layer = Heater(n_heaters=n_inputs)
        ph_last_layer = Heater(n_heaters=n_inputs)
        if is_fastforward:
            layer_list = []
            layer_list.append(ph_first_layer)
        else:
            self._n_layers = n_layers
            self._ph_first_layer = ph_first_layer
            self._layer_matrices = nn.ParameterList()
            self._ph_last_layer = ph_last_layer

        for i in range(n_layers):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = ClementsBellLayer(
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
            layer_list.append(ph_last_layer)
            self._ph_layers = torch.nn.Sequential(*layer_list)

    def forward(self, x_complex):
        return self._ph_layers(x_complex)

    def get_tf_matrix(self):
        first_layer_phases = self._ph_first_layer.getPhases()
        first_layer_tf_matrix = torch.diag(first_layer_phases)
        last_layer_phases = self._ph_last_layer.getPhases()
        last_layer_tf_matrix = torch.diag(last_layer_phases)
        tf_matrix_mid_layers = torch.linalg.multi_dot(
            tuple(
                [self._layer_matrices[i]
                    for i in range(len(self._layer_matrices))
                 ]
            )
        )
        tf_matrix = torch.linalg.multi_dot(
            (first_layer_tf_matrix, tf_matrix_mid_layers, last_layer_tf_matrix)
        )
        return tf_matrix
