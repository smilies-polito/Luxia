import torch
from torch import nn
from .photonic_components.heater import Heater
from .photonic_components.beamsplitters.mmi import Mmi
from .util.util_functions import diag_matrix_alternating_tensor_with_ones


class FldzhyanBlock(nn.Module):
    def __init__(self, n_blocks: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        """ Create the basic model of the Fldzhyan architecture, the basic
        block is a phase shifter followed by a mmi

           wg0
        A -[w]-- ---C
                X
        B ------ ---D

        Args:
            n_blocks: number of fldzhyzn blocks to create dowsizes
            mmi_i_losses: Tensor of insertion losses of the mmis
            mmi_imbalances: Tensor of the imbalances of the mmis
        """
        super(FldzhyanBlock, self).__init__()
        self._heater = Heater(n_heaters=n_blocks)
        self._mmi = Mmi(n_mmis=n_blocks,
                        i_loss=mmi_i_losses,
                        imbalance=mmi_imbalances)

    def forward(self, portsA, portsB):
        """
        Note: n_A_inputs and n_B_inputs can change based on the fact that
        the number of the current layer or block is even or odd.
        However, n_A_inputs := n_B_inputs := n_blocks always and
        n_blocks = n_inputs // 2

        Args:
            portsA: all "A" inputs, sampled every two (batch_size, n_A_inputs)
            portsB: all "B" inputs, sampled every two (batch_size, n_B_inputs)

        Return:
            y_complex: (batch_size, 2*n_blocks)
        """
        outPs = self._heater(portsA) # (batch_size, n_blocks)
        y_0, y_1 = self._mmi(outPs, portsB) # (batch_size, n_blocks), (batch_size, n_blocks)

        y_complex = torch.stack([y_0, y_1], axis=2) # (batch_size, n_blocks, 2)

        # return the elements by alternating them
        y_complex = y_complex.view(y_0.shape[0], -1) # (batch_size, 2*n_blocks)

        return y_complex

    def getTFMatrix(self):
        # Get the transfer function matrix of the mmis
        mmi_tf_matrices = self._mmi.get_tf_matrix()
        # Create a block diagonal matrix with the mmi tfunction matrices
        mmi_diag_matrix = torch.block_diag(*mmi_tf_matrices)
        # Get the phases of the heaters
        heater_phases = self._heater.getPhases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)
        # Create the transfer function matrix of the block
        tf_matrix = torch.matmul(heater_diag_matrix, mmi_diag_matrix)
        return tf_matrix


class FldzhyanLayer(nn.Module):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:      b) Odd Layer:
        A -[w]-- ---C       A ------------C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B ------------C

        ==========================================
        Fig. 2 Odd f Inputs
          a) Even Layer:    b) Odd Layer:
        A -[w]-- ---C       A ___________ C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A -[w]-- ---C       A ------ -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A ----------C       A ------ -----D
    Args:
        n_inputs: n inputs of the layer, from here it can be said
        if it is an even or and odd layer
        is_even_layer: What type of layer to be created
        mmi_i_losses: Tensor of all the losses of the mmis of the block
        mmi_imbalances: Tensor with imbalances of the mmis of the block
    """

    def __init__(self, n_inputs: int,
                 is_even_layer: bool = True,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(FldzhyanLayer, self).__init__()
        self._n_inputs = n_inputs
        self._is_even_layer = is_even_layer
        self._n_blocks = n_inputs // 2
        self._is_even_input = (self._n_inputs % 2 == 0)

        if self._is_even_input and not self._is_even_layer:
            self._n_blocks = self._n_blocks - 1

        self._ph_blocks = FldzhyanBlock(
            n_blocks=self._n_blocks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
        )

    def forward(self, x_complex):
        """
        Args:
            x_complex: input batch in complex numbers (batch_size, n_inputs)
        
        Return:
            y_complex: output batch (batch_size, n_inputs)
        """
        y_complex = torch.zeros_like(x_complex)
        if self._is_even_input:
            if self._is_even_layer:
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
            if self._is_even_layer:
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
                # Create a tensor with all Bs (x_complex_0)
                x_complex_0 = x_complex[:, ::2]
                # Create a tensor with all As (x_complex_1)
                x_complex_1 = x_complex[:, 1::2]
                y_complex[:, 1:] = self._ph_blocks(x_complex_0, x_complex_1)
        return y_complex

    def getTFMatrix(self):
        if self._is_even_input is True:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.getTFMatrix()
            else:
                tf_matrix = self._ph_blocks.getTFMatrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix, ones)
        else:
            if self._is_even_layer is True:
                tf_matrix = self._ph_blocks.getTFMatrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(tf_matrix, ones)
            else:
                tf_matrix = self._ph_blocks.getTFMatrix()
                ones = torch.ones(1)
                tf_matrix = torch.block_diag(ones, tf_matrix)
        return tf_matrix


class FldzhyanArchitecture(nn.Module):
    """ Create the photonic architecture of the Fldzhyan

        Fig. 1 Even number of inputs (n_inputs = 6) and n_layers = 8

        |---------------------------- N = 8 -------------------------------|
        A -[w]-- ----------[w]-- ----------[w]-- ----------[w]-- -----------
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --------------- --------------- --------------- -----------

        Fig. 2 Odd number of inputs (n_inputs = 7) and n_layers = 8

        |---------------------------- N = 8 -------------------------------|
        A -[w]-- ----------[w]-- ----------[w]-- ----------[w]-- -----------
                X               X               X               x
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               X
        A -[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ------- ---
                X               X               X               X
        B ------ --[w]-- ------- --[w]-- ------- --[w]-- ------- --[w]-- ---
                        X               X               X               x
        A ------------------------------------------------------------------

        Args:
            n_layers: number of layers of the photonic architecture
            n_inputs: number of inputs of the photonic architecture
            mmi_i_losses: Tensor of all the losses of the mmis in the
            architecture
            mmi_imbalances: Tensor with imbalances of the mmis in the
            architecture
    """

    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_fastforward: bool = False
                 ):
        super(FldzhyanArchitecture, self).__init__()

        if is_fastforward:
            layer_list = []
        else:
            self._layer_matrices = nn.ParameterList()

        for i in range(n_layers):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = FldzhyanLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance
            )

            if is_fastforward:
                layer_list.append(layer)
            else:
                self._layer_matrices.append(layer.getTFMatrix())

        if is_fastforward:
            self._ph_layers = torch.nn.Sequential(*layer_list)

    def forward(self, x_complex):
        """
        Args:
            x_complex: input batch in complex numbers (batch_size, n_inputs)
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
                [self._layer_matrices[i] for i in range(len(self._layer_matrices))]
            )
        )
        return tf_matrix
