import torch
from .photonic_architecture import PhotonicBlock, PhotonicLayer
from .photonic_architecture import PhotonicArchitecture
from ..photonic_components.heater import Heater
from ..photonic_components.beamsplitters.mmi import Mmi
from ..util.util_functions import diag_matrix_alternating_tensors


class FldzhyanBellEvenBlock(PhotonicBlock):
    """ Create the basic model of the fldzhyan bell architecture, the basic
        block is a phase shifter followed by a mzi module

               mmi
            A -- ---C
                X
            B -- ---D

        Args:
        n_blocks: number of fldzhyan bell blocks to create dowsizes
        mmi_i_losses: Tensor of insertion losses of the bs0 and bs1
        mmi_imbalances: Tensor of imbalances of the mmi0 and mmi1
    """

    def __init__(self, n_blocks: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(FldzhyanBellEvenBlock, self).__init__(n_blocks=n_blocks)
        self._mmi = Mmi(n_mmis=n_blocks,
                        i_loss=mmi_i_losses,
                        imbalance=mmi_imbalances)

    def forward(self, portsA, portsB):
        return self._mmi(portsA, portsB)

    def get_tf_matrix(self):
        return self._mmi.get_tf_matrix()


class FldzhyanBellOddBlock(PhotonicBlock):
    """ Create the basic model of the fldzhyan bell architecture, the basic
        block is a phase shifter followed by a mzi module

               mmi
        B -[w]-- -----C
                X
        A -[w]-- -----D

        Args:
        n_blocks: number of fldzhyan bell blocks to create dowsizes
        mmi_i_losses: Tensor of insertion losses of the bs0 and bs1
        mmi_imbalances: Tensor of imbalances of the mmi0 and mmi1
    """

    def __init__(self, n_blocks: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None
                 ):
        super(FldzhyanBellOddBlock, self).__init__(n_blocks=n_blocks)
        self._heater_0 = Heater(n_heaters=n_blocks)
        self._heater_1 = Heater(n_heaters=n_blocks)
        self._mmi = Mmi(n_mmis=n_blocks,
                        i_loss=mmi_i_losses,
                        imbalance=mmi_imbalances)

    def forward(self, portsA, portsB):
        outPs_0 = self._heater_0(portsA)
        outPs_1 = self._heater_1(portsB)
        return self._mmi(outPs_0, outPs_1)

    def get_tf_matrix(self):
        mmi_diag_matrix = self._mmi.get_tf_matrix()
        heater_phases_0 = self._heater_0.getPhases()
        heater_phases_1 = self._heater_1.getPhases()
        heater_diag_matrix = diag_matrix_alternating_tensors(
            heater_phases_0, heater_phases_1
        )
        tf_matrix = torch.matmul(heater_diag_matrix, mmi_diag_matrix)
        return tf_matrix


class FldzhyanDualHeaterBlock(PhotonicBlock):
    """ Create a basic block of 2 heaters for the clements bell architecture,
        this block is used to create the first and last layer of the
        architecture.
            +-------+
        A --|--[w]--|----C
            |       |
        B --|--[w]--|----D
            +-------+

        Args:
        n_blocks: number of clements blocks to create dowsizes
    """

    def __init__(self, n_blocks: int):
        super(FldzhyanDualHeaterBlock, self).__init__(n_blocks=n_blocks)
        self._heater_0 = Heater(n_heaters=self._n_blocks)
        self._heater_1 = Heater(n_heaters=self._n_blocks)

    def forward(self, portsA, portsB):
        y_0 = self._heater_0(portsA)
        y_1 = self._heater_1(portsB)
        return y_0, y_1

    def get_tf_matrix(self):
        phases_0 = self._heater_0.getPhases()
        phases_1 = self._heater_1.getPhases()
        return diag_matrix_alternating_tensors(phases_0, phases_1)


class FldzhyanBellLayer(PhotonicLayer):
    """
    ============ Fldzhyan Bell Even N inputs ============

      Fig. 1 Even number of Inputs
          a) Even Layer:      b) Odd Layer:
        A ------ ---C       A -[w]--------C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A ------ ---C       A -[w]-- -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A ------ ---C       A -[w]-- -----D
                X
        B ------ ---D       B -[w]--------C

    ============ Fldzhyan Bell Odd N inputs ============

      Fig. 2 Odd f Inputs
          a) Even Layer:    b) Odd Layer:
        A ------ ---C       A -[w]--------C
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A ------ ---C       A -[w]-- -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A ------ ---C       A -[w]-- -----D
                X
        B ------ ---D       B -[w]-- -----C
                                    X
        A------[w]--C       A -[w]-- -----D

    ============ Layer of Heaters ============
      Fig. 3 Start and last layer
             wg
        A -- --[w]-- ---C

        B -- --[w]-- ---D

        A -- --[w]-- ---C

        B -- --[w]-- ---D

        A -- --[w]-- ---C

        B -- --[w]-- ---D

    Args:
        n_inputs: n inputs of the layer, from here it can be said
        if it is an even or and odd layer
        is_even_layer: What type of layer to be created
        mmi_i_losses: Tensor of all the losses of the mmis of the block
        mmi_imbalances: Tensor with imbalances of the mmis of the block
    """

    def __init__(self, n_inputs: int,
                 is_even_layer: bool = True,
                 opt_ps: bool = False,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_heater_layer: bool = False
                 ):
        if is_heater_layer:
            fldzhyan_block = FldzhyanDualHeaterBlock(n_blocks=n_inputs//2)
        else:
            # When the number of inputs is even and is an odd layer, the number
            # of blocks is reduced by 1 otherwise the n_blks = n_inputs // 2
            n_blks = n_inputs // 2
            if (n_inputs % 2 == 0) and not is_even_layer:
                n_blks = n_blks - 1

            if is_even_layer:
                fldzhyan_block = FldzhyanBellEvenBlock(
                    n_blocks=n_blks,
                    mmi_i_losses=mmi_i_losses,
                    mmi_imbalances=mmi_imbalances
                )
            else:
                fldzhyan_block = FldzhyanBellOddBlock(
                    n_blocks=n_blks,
                    mmi_i_losses=mmi_i_losses,
                    mmi_imbalances=mmi_imbalances
                )
        super(FldzhyanBellLayer, self).__init__(
            n_inputs=n_inputs,
            is_even_layer=is_even_layer,
            opt_ps=opt_ps,
            photonic_block=fldzhyan_block
        )


class FldzhyanBellArchitecture(PhotonicArchitecture):
    def __init__(self, n_layers: int, n_inputs: int,
                 mmi_i_losses: torch.Tensor = None,
                 mmi_imbalances: torch.Tensor = None,
                 is_matmul: bool = False
                 ):

        layer_list = []
        layer_list.append(FldzhyanBellLayer(n_inputs=n_inputs,
                                            is_even_layer=True,
                                            opt_ps=True,
                                            is_heater_layer=True))

        for i in range(n_layers - 2):
            i_losses = None if mmi_i_losses is None else mmi_i_losses[i]
            imbalance = None if mmi_imbalances is None else mmi_imbalances[i]

            layer = FldzhyanBellLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                opt_ps=True,
                mmi_i_losses=i_losses,
                mmi_imbalances=imbalance
            )
            layer_list.append(layer)

        layer_list.append(FldzhyanBellLayer(n_inputs=n_inputs,
                                            is_even_layer=True,
                                            opt_ps=True,
                                            is_heater_layer=True))
        super(FldzhyanBellArchitecture, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=layer_list
        )
