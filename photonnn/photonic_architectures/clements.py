import torch
from photonnn.photonic_components.heater import Heater
from photonnn.photonic_components.mzis.single_heater import SingleHeaterMzi
from photonnn.util.util_functions import diag_matrix_alternating_tensor_with_ones
from .photonic_mesh import PhotonicMesh, PhotonicLayer, PhotonicBlock


class ClementsBlock(PhotonicBlock):
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
        super(ClementsBlock, self).__init__(n_blocks=n_blocks)
        self._heater = Heater(n_heaters=n_blocks)
        self._mzi = SingleHeaterMzi(n_mzi=n_blocks,
                                    mmi_i_losses=mmi_i_losses,
                                    mmi_imbalances=mmi_imbalances)

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        return self._mzi(outPs, portsB)

    def get_tf_matrix(self):
        # Get the phases of the heaters wg0
        heater_phases = self._heater.getPhases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(
                heater_phases)
        # Get the transfer function matrix of the mzi
        mzi_tf_matrix = self._mzi.get_tf_matrix()
        # Multiply the heater diagonal with the mzi transfer function matrix
        return torch.matmul(heater_diag_matrix, mzi_tf_matrix)


class ClementsLayer(PhotonicLayer):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A ------------------C
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B ------------------C

        ========================================================
        Fig. 2 Odd number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A ------------------C
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
        A ------------------C       A ------ ------- ---D

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
                 opt_ps: bool = False,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 ):

        # When the number of inputs is even and is an odd layer, the number
        # of blocks is reduced by 1 otherwise the number of blocks is always
        # half of the number of inputs
        n_blks = n_inputs // 2
        if (n_inputs % 2 == 0) and not is_even_layer:
            n_blks = n_blks - 1

        clements_block = ClementsBlock(
            n_blocks=n_blks,
            mmi_i_losses=mmi_i_losses,
            mmi_imbalances=mmi_imbalances
        )
        super(ClementsLayer, self).__init__(n_inputs=n_inputs,
                                            is_even_layer=is_even_layer,
                                            opt_ps=opt_ps,
                                            photonic_block=clements_block)


class ClementsMesh(PhotonicMesh):
    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_matmul: bool = False
                 ):

        list_layers = []
        for i in range(n_layers):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = ClementsLayer(n_inputs=n_inputs,
                                  is_even_layer=(i % 2 == 0),
                                  opt_ps=False,
                                  mmi_i_losses=i_losses,
                                  mmi_imbalances=imbalance)

            list_layers.append(layer)

        super(ClementsMesh, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=list_layers
        )
