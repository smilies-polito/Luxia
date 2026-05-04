from luxia.photonic_components.beamsplitters.cross import Cross
from luxia.photonic_components.beamsplitters.half_cross import HalfCross
from luxia.photonic_meshes.clements.clements_block import ClementsBlock
from luxia.photonic_meshes.mesh.photonic_layer import PhotonicLayer
from luxia.params import BsCrossParams, BsHalfCrossParams
from luxia.util.util_functions import alternating_tensors
from luxia.util.log_config import lx_log
from luxia.params import DacParams
from typing import Optional, List
import torch.nn as nn
import torch


class BraidedLayerT1(PhotonicLayer):
    """
    Fig. 1 Even number of Inputs
        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        ========================================================
    Fig. 2 Odd number of Inputs
        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        A -[w]-- --[w]-- ---C
                X       X
        B ------ ------- ---D

        A ------------------C

    Args:
      n_inputs: number of inputs of the layer, from here it can be said
      if it is an even or and odd layer
      is_even_layer: What type of layer to be created
      mmi_i_losses: Tesor of all the losses of the blocks, each block
      contains, 2 tensors of losses since each block uses 2 mmis
      mmi_imbalances: Tesor of all the imbalances of the blocks, each
      block contains, 2 tensors of imbalance since each block uses 2 mmis
      bit_width: bit resolution of the phase shifters of the heaters
    """

    def __init__(self,
                 n_inputs: int,
                 mmis_params: Optional[List] = None,
                 dac_params: Optional[DacParams] = None,
                 debug: bool = False
                 ):

        # The number of blocks is always half of the number of inputs
        n_blks = n_inputs // 2

        # The type 1 of the braided layer uses a Clements block as the building
        # block, which is a combination of MMIs and phase shifters
        clements_block = ClementsBlock(
            n_blocks=n_blks,
            mmis_params=mmis_params,
            dac_params=dac_params,
        )
        super(BraidedLayerT1, self).__init__(n_inputs=n_inputs,
                                             is_even_layer=True,  # Always even
                                             opt_ps=False,  # Not used
                                             photonic_block=clements_block,
                                             debug=debug
                                             )

    def get_heaters_quantization_error(self):
        blk = self.get_photonic_block()
        qe_heater, qe_mzi = blk.get_heaters_quantization_error()
        return qe_heater, qe_mzi

    def get_heaters_snr(self):
        snr = self.get_photonic_block().get_heaters_snr()
        return snr

    def get_heaters_sqnr(self):
        sqnr = self.get_photonic_block().get_heaters_sqnr()
        return sqnr

    def set_heaters_parameters(self, params: DacParams):
        blk = self.get_photonic_block()
        blk.set_heater_params(params)


class BraidedLayerT2(PhotonicLayer):
    """
      Fig. 1 Even number of Inputs
        A --------HC-------C

        B ------ ------- ---C
                   X
        A ------ ------- ---D

        B ------ ------- ---C
                   X
        A ------ ------- ---D

        B ---------HC-------C

        ========================================================
        Fig. 2 Odd number of Inputs
        A ---------HC-------C

        B ------ ------- ---C
                   X
        A ------ ------- ---D

        B ------ ------- ---C
                   X
        A ------ ------- ---D

        B ------ ------- ---C
                   X
        A ------------------C

    Args:
        n_inputs: number of inputs of the layer, from here it can be said if
        it is the even or odd form of the layer
        cross_params: Paramsfor the Cross block, if None, default params
        hf_cross_params: Paramsfor the HalfCross block, if None, default params
        debug: If True, debug information will be printed
    """

    def __init__(self,
                 n_inputs: int,
                 cross_params: Optional[List] = None,
                 hf_cross_params: Optional[List] = None,
                 debug: bool = False
                 ):

        # The number of blocks is always half of the number of inputs when
        # the number of inputs is even, and one less when it is odd
        n_blks = n_inputs // 2
        if (n_inputs % 2 == 0):
            n_blks = n_blks - 1

        p = BsCrossParams(
            num_splitters=n_blks) if cross_params is None else cross_params
        # The type 2 of the braided layer uses a Cross block as the building
        # block, which is just a set of cross couplers
        # with no phase shifters
        cross_block = Cross(cross_params=p)
        super(BraidedLayerT2, self).__init__(n_inputs=n_inputs,
                                             is_even_layer=False,  # Always odd
                                             opt_ps=False,  # Not used
                                             photonic_block=cross_block,
                                             debug=debug
                                             )

        # If the number of inputs is even, we add a HalfCross block
        # to the first and last elements, otherwise only to the first element
        hc_p = BsHalfCrossParams(
            num_splitters=1) if hf_cross_params is None else hf_cross_params
        if self._is_even_input is True:
            hc_p = [hc_p, hc_p] if isinstance(
                hc_p, BsHalfCrossParams) else hc_p
            self._hlf_cross_block = nn.ModuleList([
                HalfCross(half_cross_params=hc_p[0]),
                HalfCross(half_cross_params=hc_p[1])
            ])
            self.forward = self._forward_even
        else:
            self._hlf_cross_block = HalfCross(half_cross_params=hc_p)
            self.forward = self._forward_odd

        # The type 2 of the braided layer does not have heaters and therefore
        # there is no quantization error, SNR or SQNR

    def _forward_even(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure T2.1
        """
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug("x: %s", x_complex)
        y_complex = torch.zeros_like(x_complex)

        # A_0 -{HC}> C_0 (first elements)
        y_complex[:, 0] = self._hlf_cross_block[0](x_complex[:, 0])
        # A_N -{HC}> C_N (lasts elements)
        y_complex[:, -1] = self._hlf_cross_block[1](x_complex[:, - 1])

        x_complex = x_complex[:, 1:-1]  # Remove the first and last elements

        x_complex_0 = x_complex[:, ::2]  # Get all B's (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all A's (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:-1] = alternating_tensors(y_0, y_1)
        if self._debug:
            lx_log.debug("y: %s", y_complex)
            lx_log.debug("|y|: %s", torch.abs(y_complex))
            lx_log.debug("sum(|y|): %s", torch.sum(torch.abs(y_complex)))
        return y_complex

    def _forward_odd(self, x_complex):
        """
        Create the forward pass of the layer according to the Figure T2.2
        """
        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug("x: %s", x_complex)
        y_complex = torch.zeros_like(x_complex)
        # A_0 -{p[0]}> C_0 (first elements)
        y_complex[:, 0] = self._hlf_cross_block(x_complex[:, 0])

        x_complex = x_complex[:, 1:]  # Remove the first elements
        x_complex_0 = x_complex[:, ::2]  # Get all Bs (x_complex_0)
        x_complex_1 = x_complex[:, 1::2]  # Get all As (x_complex_1)
        y_0, y_1 = self._ph_blocks(x_complex_0, x_complex_1)
        y_complex[:, 1:] = alternating_tensors(y_0, y_1)
        if self._debug:
            lx_log.debug("y: %s", y_complex)
            lx_log.debug("|y|: %s", torch.abs(y_complex))
            lx_log.debug("sum(|y|): %s", torch.sum(torch.abs(y_complex)))
        return y_complex

    def set_heaters_parameters(self, params: DacParams):
        return # There are no heaters in this layer

