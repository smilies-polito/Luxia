from typing import Optional
from dataclasses import replace

import torch

from luxia.params import HeaterParams
from luxia.photonic_components.heater import Heater
from luxia.photonic_meshes.mesh.photonic_block import PhotonicBlock
from luxia.util.util_functions import diag_matrix_alternating_tensors


class ClementsDualHeaterBlock(PhotonicBlock):
    """Dual-heater block used for the first and last Bell layers.

    This block applies one heater on each straight waveguide in every 2-port group:

            +-------+
        A --|--[w]--|---- C
            |       |
        B --|--[w]--|---- D
            +-------+

    Args:
        n_blocks: Number of parallel 2-port heater pairs.
        heater_params: Optional heater configuration for both heaters.
    """

    def __init__(
        self,
        n_blocks: int,
        heater_params: Optional[HeaterParams] = None,
    ):
        super(ClementsDualHeaterBlock, self).__init__(n_blocks=n_blocks)

        if heater_params is None:
            heater_params = HeaterParams(n_heaters=n_blocks)
        else:
            heater_params = replace(heater_params, n_heaters=n_blocks)

        self._heater_0 = Heater(heater_params=heater_params)
        self._heater_1 = Heater(heater_params=heater_params)

    def forward(self, portsA, portsB):
        y_0 = self._heater_0(portsA)
        y_1 = self._heater_1(portsB)
        return y_0, y_1

    def get_tf_matrix(self):
        phases_0 = self._heater_0.get_phases()
        phases_1 = self._heater_1.get_phases()
        return diag_matrix_alternating_tensors(phases_0, phases_1)

    def get_heaters_quantization_error(self):
        qe_0 = self._heater_0.get_quantization_error()
        qe_1 = self._heater_1.get_quantization_error()
        return qe_0, qe_1

    def get_heaters_snr(self):
        snr_0 = torch.as_tensor(self._heater_0.get_dac_snr())
        snr_1 = torch.as_tensor(self._heater_1.get_dac_snr())
        return torch.stack([snr_0, snr_1], dim=0).mean(dim=0)

    def get_heaters_sqnr(self):
        sqnr_0 = torch.as_tensor(self._heater_0.get_dac_sqnr())
        sqnr_1 = torch.as_tensor(self._heater_1.get_dac_sqnr())
        return torch.stack([sqnr_0, sqnr_1], dim=0).mean(dim=0)

    def set_heater_params(self, dac_params):
        self._heater_0.set_use_dac(True, dac_params)
        self._heater_1.set_use_dac(True, dac_params)
