from luxia.util.util_functions import diag_matrix_alternating_tensors
from luxia.photonic_components.mzis.dual_heater import DualHeaterMzi
from luxia.photonic_components.heater import Heater
from luxia.photonic_meshes.mesh import PhotonicMesh, PhotonicLayer, PhotonicBlock
from luxia.params import HeaterParams, MmiParams
from typing import Optional, List
import torch


class ClementsBellBlock(PhotonicBlock):
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
        mmi_params: List of MmiParams objects with the parameters of the mmis
        dac_params: DacParams object with the parameters of the DAC
    """

    def __init__(self,
                 n_blocks: int,
                 mmis_params: List = None,
                 dac_params: Optional[HeaterParams] = None,
                 ):
        super(ClementsBellBlock, self).__init__(n_blocks=n_blocks)
        self._mzi = DualHeaterMzi(n_mzi=self._n_blocks,
                                  mmis_params=mmis_params,
                                  dac_params=dac_params,
                                  )

    def forward(self, portsA, portsB):
        return self._mzi(portsA, portsB)

    def get_tf_matrix(self):
        return self._mzi.get_tf_matrix()


class ClementsDualHeaterBlock(PhotonicBlock):
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

    def __init__(self, n_blocks: int, dac_params: Optional[HeaterParams] = None):
        super(ClementsDualHeaterBlock, self).__init__(n_blocks=n_blocks)
        self._heater_0 = Heater(HeaterParams(number_of_heaters=self._n_blocks))
        self._heater_1 = Heater(HeaterParams(number_of_heaters=self._n_blocks))

    def forward(self, portsA, portsB):
        y_0 = self._heater_0(portsA)
        y_1 = self._heater_1(portsB)
        return y_0, y_1

    def get_tf_matrix(self):
        phases_0 = self._heater_0.get_phases()
        phases_1 = self._heater_1.get_phases()
        return diag_matrix_alternating_tensors(phases_0, phases_1)


class ClementsBellLayer(PhotonicLayer):
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
                 opt_ps: bool = False,
                 mmis_params: Optional[List] = None,
                 dac_params: Optional[HeaterParams] = None,
                 is_heater_layer: bool = False
                 ):
        if is_heater_layer:
            clements_block = ClementsDualHeaterBlock(n_blocks=n_inputs // 2)
        else:
            # When the number of inputs is even and is an odd layer, the number
            # of blocks is reduced by 1 otherwise the number of blocks is
            # always half of the number of inputs
            n_blks = n_inputs // 2
            if (n_inputs % 2 == 0) and not is_even_layer:
                n_blks = n_blks - 1

            clements_block = ClementsBellBlock(
                n_blocks=n_blks,
                mmis_params=mmis_params,
                dac_params=dac_params
            )

        super(ClementsBellLayer, self).__init__(
                n_inputs=n_inputs,
                is_even_layer=is_even_layer,
                opt_ps=opt_ps,
                photonic_block=clements_block
                )


class ClementsBellMesh(PhotonicMesh):
    def __init__(self, n_layers: int, n_inputs: int,
                 mmis_params: Optional[List] = None,
                 dac_params: Optional[HeaterParams] = None,
                 is_matmul: bool = False
                 ):
        layer_list = []
        layer_list.append(ClementsBellLayer(n_inputs=n_inputs,
                                            is_even_layer=True,
                                            opt_ps=True,
                                            is_heater_layer=True))

        for i in range(n_layers - 2):
            p = None if mmis_params is None else mmis_params[i]

            layer = ClementsBellLayer(
                n_inputs=n_inputs,
                is_even_layer=(i % 2 == 0),
                opt_ps=True,
                mmis_params=p,
                dac_params=dac_params
            )
            layer_list.append(layer)

        layer_list.append(ClementsBellLayer(n_inputs=n_inputs,
                                            is_even_layer=True,
                                            opt_ps=True))

        super(ClementsBellMesh, self).__init__(
            n_layers=n_layers,
            n_inputs=n_inputs,
            is_matmul=is_matmul,
            layer_list=layer_list
        )
