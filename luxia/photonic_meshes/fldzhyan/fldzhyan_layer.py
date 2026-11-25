from luxia.photonic_meshes.mesh import PhotonicLayer
from .fldzhyan_block import FldzhyanBlock
from typing import Optional, List
from luxia.params import DacParams, MmiParams
import torch


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
                 mmi_params: Optional[MmiParams] = None,
                 dac_params: Optional[DacParams] = None,
                 ):
        # When the number of inputs is even and is an odd layer, the number
        # of blocks is reduced by 1 otherwise the number of blocks is always
        # half of the number of inputs
        n_blks = n_inputs // 2
        if (n_inputs % 2 == 0) and not is_even_layer:
            n_blks = n_blks - 1

        fldzhyan_block = FldzhyanBlock(
            n_blocks=n_blks,
            mmis_params=mmi_params,
            dac_params=dac_params
        )
        super(FldzhyanLayer, self).__init__(n_inputs=n_inputs,
                                            is_even_layer=is_even_layer,
                                            opt_ps=False,
                                            photonic_block=fldzhyan_block
                                            )

    def get_heaters_quantization_error(self):
        """
        Get the quantization error of the heaters in the layer
        :return: quantization error of the heaters in the layer
        """
        blk = self.get_photonic_block()
        q_error = blk.get_heater_quantization_error()
        return q_error

    def get_heaters_snr(self):
        """
        Get the SNR of the heaters in the layer
        :return: SNR of the heaters in the layer
        """
        blk = self.get_photonic_block()
        snr = blk.get_heater_snr()
        return snr

    def get_heaters_sqnr(self):
        """
        Get the SQNR of the heaters in the layer
        :return: SQNR of the heaters in the layer
        """
        blk = self.get_photonic_block()
        sqnr = blk.get_heater_sqnr()
        return sqnr

    def get_phase_shifts(self) -> torch.Tensor:
        blk = self.get_photonic_block()
        return blk._heater.phase_shift.unsqueeze(0)

    def set_phase_shifts(self, phases: torch.Tensor) -> None:
        blk = self.get_photonic_block()
        blk._heater.set_phase_shift(phases[0])

    def set_heaters_parameters(self, params: DacParams):
        """
        Set the parameters of the heaters in the layer
        :param params: parameters to be set
        """
        blk = self.get_photonic_block()
        blk.set_heater_params(params)

