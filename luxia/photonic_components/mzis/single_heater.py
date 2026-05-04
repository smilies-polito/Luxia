import torch
from luxia.photonic_components.mzis.mzi import Mzi
from luxia.photonic_components.heater import Heater
from luxia.util.util_functions import diag_matrix_alternating_tensor_with_ones
from typing import Optional, List
from luxia.params import BsMmiParams, HeaterParams


class SingleHeaterMzi(Mzi):
    """
    Mach-Zehnder Interferometer with a single waveguide heater.

         _____ MZI _____
        |    wg1        |
        |--- --[w]-- ---|-C
        |   X       X   |
        |--- ------- ---|-D
        |  mmi0   mmi1  |
         ---------------

    Args:
        n_mzi (int): Number of MZIs.
        mmis_params (List[BsMmiParams], optional): Parameters for the MMIs.
        heater_params (HeaterParams, optional): Parameters for the heater.
    """

    def __init__(
        self,
        n_mzi: int,
        mmis_params: Optional[List[BsMmiParams]] = None,
        heater_params: Optional[HeaterParams] = None,
    ):
        super(SingleHeaterMzi, self).__init__(
            n_mzi=n_mzi,
            mmis_params=mmis_params,
        )
        self._heater = Heater(heater_params=heater_params)

    def forward(self, portA: torch.Tensor, portB: torch.Tensor):
        """
        Forward pass through the MZI with a single heater.

        Args:
            portA: Input tensor from port A.
            portB: Input tensor from port B.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors from the MZI.
        """
        wave_0, wave_1 = self._mmi_0(portA, portB)
        outPs = self._heater(wave_0)
        return self._mmi_1(outPs, wave_1)

    def get_tf_matrix(self) -> torch.Tensor:
        mmi_0_diag_matrix = self._mmi_0.get_tf_matrix()
        mmi_1_diag_matrix = self._mmi_1.get_tf_matrix()

        heater_phases = self._heater.get_phases()
        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(heater_phases)

        # # DEBUG: print devices and dtypes
        # print("[DEBUG] mmi_0_diag_matrix device:", mmi_0_diag_matrix.device,
        #       "dtype:", mmi_0_diag_matrix.dtype)
        # print("[DEBUG] heater_diag_matrix device:", heater_diag_matrix.device,
        #       "dtype:", heater_diag_matrix.dtype)
        # print("[DEBUG] mmi_1_diag_matrix device:", mmi_1_diag_matrix.device,
        #       "dtype:", mmi_1_diag_matrix.dtype)

        tf_matrix = torch.linalg.multi_dot(
            (mmi_0_diag_matrix, heater_diag_matrix, mmi_1_diag_matrix)
        )
        return tf_matrix

    def get_heater_quantization_error(self):
        """
        Get the quantization error of the DAC.

        Returns:
            torch.Tensor: Quantization error.
        """
        return self._heater.get_quantization_error()

    def get_heater_snr(self):
        """Get the heater SNR."""
        return self._heater.get_dac_snr()

    def get_heater_sqnr(self):
        """Get the heater SQNR."""
        return self._heater.get_dac_sqnr()

    def set_heater_params(self, heater_params: HeaterParams):
        """Enable DAC settings for the internal heater."""
        self._heater.set_use_dac(True, heater_params)
