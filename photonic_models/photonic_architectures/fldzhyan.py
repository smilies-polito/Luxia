import torch
from .photonic_architecture import PhotonicBlock, PhotonicLayer
from .photonic_architecture import PhotonicArchitecture
from ..photonic_components.heater import Heater
from ..photonic_components.beamsplitters.mmi import Mmi
from ..util.util_functions import diag_matrix_alternating_tensor_with_ones


class FldzhyanBlock(PhotonicBlock):
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
        super(FldzhyanBlock, self).__init__(n_blocks=n_blocks)
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
        outPs = self._heater(portsA)  # (batch_size, n_blocks)
        # (batch_size, n_blocks), (batch_size, n_blocks)
        return self._mmi(outPs, portsB)


    def get_tf_matrix(self):
        # Get the transfer function matrix of the mmis
        mmi_diag_matrix = self._mmi.get_tf_matrix()
        # Get the phases of the heaters
        heater_phases = self._heater.getPhases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)
        # Create the transfer function matrix of the block
        tf_matrix = torch.matmul(heater_diag_matrix, mmi_diag_matrix)
        return tf_matrix


class FldzhyanLayer(PhotonicLayer):
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
        # When the number of inputs is even and is an odd layer, the number
        # of blocks is reduced by 1 otherwise the number of blocks is always
        # half of the number of inputs
        n_blks = n_inputs // 2
        if (n_inputs % 2 == 0) and not is_even_layer:
            n_blks = n_blks - 1
        fldzhyan_block = FldzhyanBlock(
            n_blocks=n_blks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
        )
        super(FldzhyanLayer, self).__init__(n_inputs=n_inputs,
                                            is_even_layer=is_even_layer,
                                            opt_ps=False,
                                            photonic_block=fldzhyan_block
                                            )


class FldzhyanArchitecture(PhotonicArchitecture):
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
                 mmi_i_losses: [] = None,
                 mmi_imbalances: [] = None,
                 is_matmul: bool = False
                 ):

        list_layers = []
        for i in range(n_layers):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = FldzhyanLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance
            )
            list_layers.append(layer)

        super(FldzhyanArchitecture, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=list_layers
        )
