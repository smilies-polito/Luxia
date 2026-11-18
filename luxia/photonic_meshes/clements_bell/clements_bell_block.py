from typing import Optional

import torch

from luxia.params import ClementsBellBlockParams
from luxia.photonic_components.mzis.dual_heater import DualHeaterMzi
from luxia.photonic_meshes.mesh.photonic_block import PhotonicBlock


class ClementsBellBlock(PhotonicBlock):
    """Basic building block of the Clements-Bell architecture.

    The block consists of a dual-heater MZI with symmetric heaters on both
    waveguides. No external phase shifter is used before the MZI.

                 ____ MZI DUAL __
           wg0  |    wg1         |
        A -----|--- --[w]-- -----|- C
               |   X       X     |
        B -----|--- --[w]-- -----|- D
               | mmi0     mmi1   |
                -----------------

    Args:
        cb_params: Parameters describing the Clements-Bell block. If not
            provided, default parameters are used.
        debug: Debug flag propagated to child modules when needed.
    """

    def __init__(
        self,
        cb_params: Optional[ClementsBellBlockParams] = None,
        debug: bool = False,
    ):
        if cb_params is None:
            cb_params = ClementsBellBlockParams()

        super(ClementsBellBlock, self).__init__(n_blocks=cb_params.number_of_blocks)

        heater_params = cb_params.heater_params
        if isinstance(heater_params, list):
            if len(heater_params) == 0:
                raise ValueError(
                    "cb_params.heater_params is an empty list. "
                    "Expected at least one heater parameter set."
                )
            heater_params = heater_params[0]

        mmis_params = cb_params.mmi_params
        if mmis_params is not None and not isinstance(mmis_params, list):
            mmis_params = [mmis_params]

        self._mzi = DualHeaterMzi(
            n_mzi=cb_params.number_of_blocks,
            mmis_params=mmis_params,
            dac_params=heater_params,
        )

    def forward(self, portsA, portsB):
        """Propagate the two input ports through the Clements-Bell block."""
        return self._mzi(portsA, portsB)

    def get_tf_matrix(self):
        """Return the transfer-function matrix of the block."""
        return self._mzi.get_tf_matrix()

    def get_heaters_quantization_error(self):
        """Return the heater quantization errors of the block."""
        return self._mzi.get_heater_quantization_error()

    def get_heaters_snr(self):
        """Return the average heater SNR of the block."""
        snr_top_bottom = torch.as_tensor(self._mzi.get_heater_snr())
        return snr_top_bottom.mean(dim=0)

    def get_heaters_sqnr(self):
        """Return the average heater SQNR of the block."""
        sqnr_top_bottom = torch.as_tensor(self._mzi.get_heater_sqnr())
        return sqnr_top_bottom.mean(dim=0)

    def set_heater_params(self, dac_params):
        """Enable DAC settings for the heaters in the block."""
        self._mzi.set_heater_params(dac_params)
