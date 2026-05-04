from luxia.params.heater_params import HeaterParams
from luxia.photonic_meshes.mesh import PhotonicMesh, PhotonicLayer, PhotonicBlock
from luxia.photonic_components.beamsplitters.mmi import Mmi
from luxia.photonic_components.heater import Heater
from luxia.util.util_functions import diag_matrix_alternating_tensor_with_ones
from luxia.params import DacParams, BsMmiParams
from typing import Optional, List
import torch


class FldzhyanBlock(PhotonicBlock):
    """ Create the basic model of the Fldzhyan architecture, the basic
        block is a phase shifter followed by a mmi

           wg0
        A -[w]-- ---C
                X
        B ------ ---D

        Args:
            n_blocks: number of fldzhyzn blocks to create dowsizes
            mmis_params: List of the parameters of the mmis
            dac_params: DacParams object with the parameters of the DAC in the heater
        """

    def __init__(self,
                 n_blocks: int,
                 mmis_params: Optional[BsMmiParams] = None,
                 dac_params: Optional[DacParams] = None,
                 ):
        super(FldzhyanBlock, self).__init__(n_blocks=n_blocks)
        # If the mmis_params is None, create a default parameter
        if mmis_params is None:
            parm = BsMmiParams(num_splitters=n_blocks)
        else:
            if mmis_params.num_splitters != n_blocks:
                raise ValueError(
                    "The number of splitters in BsMmiParams"
                    " must be equal to the number of blocks")
            else:
                parm = mmis_params

        heater_params = HeaterParams(n_heaters=n_blocks, dac_params=dac_params)
        self._heater = Heater(heater_params=heater_params)
        self._mmi = Mmi(mmi_params=parm)

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
        heater_phases = self._heater.get_phases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)
        # Create the transfer function matrix of the block
        tf_matrix = torch.matmul(heater_diag_matrix, mmi_diag_matrix)
        return tf_matrix

    def get_heater_quantization_error(self):
        qe_heater = self._heater.get_quantization_error()
        return qe_heater

    def get_heater_snr(self):
        snr = (self._heater.get_dac_snr()).mean(dim=0)
        return snr

    def get_heater_sqnr(self):
        sqnr = (self._heater.get_dac_sqnr()).mean(dim=0)
        return sqnr

    def set_heater_params(self, dac_params: DacParams):
        self._heater.set_use_dac(True, dac_params)
        

