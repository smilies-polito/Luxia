from luxia.photonic_components.heater import Heater
from luxia.photonic_components.mzis.single_heater import SingleHeaterMzi
from luxia.util.util_functions import diag_matrix_alternating_tensor_with_ones
from luxia.photonic_meshes.mesh.photonic_block import PhotonicBlock
from typing import Optional, List
from luxia.params import ClementsBlockParams
import torch


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
            cb_params: parameters of the full clements block
    """

    def __init__(self, cb_params: Optional[ClementsBlockParams] = None):
        if cb_params is None:
            cb_params = ClementsBlockParams()

        super(ClementsBlock, self).__init__(n_blocks=cb_params.number_of_blocks)
        # heater_params in ClementsBlockParams is Optional[List[HeaterParams]];
        # Heater and SingleHeaterMzi expect a single Optional[HeaterParams].
        hp = cb_params.heater_params
        if isinstance(hp, list):
            hp = hp[0] if hp else None
        self._heater = Heater(hp)
        self._mzi = SingleHeaterMzi(n_mzi=cb_params.number_of_blocks,
                                    mmis_params=cb_params.mmi_params,
                                    heater_params=hp)

    def forward(self, portsA, portsB):
        outPs = self._heater(portsA)
        return self._mzi(outPs, portsB)

    def get_tf_matrix(self):
        # Get the phases of the heaters wg0
        heater_phases = self._heater.get_phases()
        # Create a diagonal matrix with the phases of the heaters
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(
                heater_phases)
        # Get the transfer function matrix of the mzi
        mzi_tf_matrix = self._mzi.get_tf_matrix()
        # Multiply the heater diagonal with the mzi transfer function matrix
        return torch.matmul(heater_diag_matrix, mzi_tf_matrix)

    def get_heaters_quantization_error(self):
        qe_heater = self._heater.get_quantization_error()
        qe_mzi = self._mzi.get_heater_quantization_error()
        return qe_heater, qe_mzi

    def get_heaters_snr(self):
        snr_heater = self._heater.get_dac_snr()
        snr_mzi = self._mzi.get_heater_snr()
        snr = torch.stack([snr_heater, snr_mzi], dim=0).mean(dim=0)
        return snr

    def get_heaters_sqnr(self):
        sqnr_heater = self._heater.get_dac_sqnr()
        sqnr_mzi = self._mzi.get_heater_sqnr()
        sqnr = torch.stack([sqnr_heater, sqnr_mzi], dim=0).mean(dim=0)
        return sqnr

    def set_heater_params(self, dac_params):
        self._heater.set_use_dac(True, dac_params)
        self._mzi.set_heater_params(dac_params)

