from typing import Optional

import torch

from luxia.params import ClementsBlockParams
from luxia.photonic_components.heater import Heater
from luxia.photonic_components.mzis.single_heater import SingleHeaterMzi
from luxia.photonic_meshes.mesh.photonic_block import PhotonicBlock
from luxia.util.util_functions import diag_matrix_alternating_tensor_with_ones


class ClementsBlock(PhotonicBlock):
    """Basic building block of the Clements architecture.

    The block consists of an input heater on the upper waveguide followed by a
    single-heater MZI.

                 _____ MZI _____
           wg0  |    wg1        |
        A -[w]--|--- --[w]-- ---|- C
                |   X       X   |
        B ------|--- ------- ---|- D
                | mmi0     mmi1 |
                 ---------------

    Args:
        cb_params: Parameters describing the full Clements block. If not
            provided, default parameters are used.
        debug: Debug flag propagated to child modules when needed.
    """

    def __init__(
        self,
        cb_params: Optional[ClementsBlockParams] = None,
        debug: bool = False,
    ):
        """Initialize the Clements block."""
        if cb_params is None:
            cb_params = ClementsBlockParams()

        super(ClementsBlock, self).__init__(n_blocks=cb_params.n_blocks)

        # Normalize heater parameters for single-heater modules.
        #
        # In the updated branch, ``heater_params`` may be stored as a list even
        # when this block only uses one external heater and one internal MZI
        # heater configuration. Both ``Heater`` and ``SingleHeaterMzi`` expect
        # a single HeaterParams object here, so we resolve that interface
        # mismatch at the block boundary.
        heater_params = cb_params.heater_params
        if isinstance(heater_params, list):
            if len(heater_params) == 0:
                raise ValueError(
                    "cb_params.heater_params is an empty list. "
                    "Expected at least one heater parameter set."
                )
            heater_params = heater_params[0]

        # Normalize MMI parameters for the internal MZI.
        #
        # ``SingleHeaterMzi`` expects a list-like container for the MMI
        # parameters. If the block already provides a list, keep it; otherwise
        # wrap the single parameter object.
        mmis_params = cb_params.mmi_params
        if not isinstance(mmis_params, list):
            mmis_params = [mmis_params]

        self._heater = Heater(heater_params=heater_params)

        self._mzi = SingleHeaterMzi(
            n_mzi=cb_params.n_blocks,
            mmis_params=mmis_params,
            heater_params=heater_params,
        )

    def forward(self, portsA, portsB):
        """Propagate the two input ports through the Clements block.

        The upper input first passes through the external heater and then both
        signals are processed by the internal single-heater MZI.

        Args:
            portsA: Input tensor for the upper waveguide.
            portsB: Input tensor for the lower waveguide.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors of the block.
        """
        outPs = self._heater(portsA)
        return self._mzi(outPs, portsB)

    def get_tf_matrix(self):
        """Return the transfer-function matrix of the full block.

        The full transfer matrix is the product of:
        1. the diagonal phase matrix produced by the external heater, and
        2. the transfer matrix of the internal MZI.

        Returns:
            torch.Tensor: Transfer-function matrix of the Clements block.
        """
        heater_phases = self._heater.get_phases()

        heater_diag_matrix = diag_matrix_alternating_tensor_with_ones(
            heater_phases
        )

        mzi_tf_matrix = self._mzi.get_tf_matrix()

        # Ensure consistent dtype before matrix multiplication.
        heater_diag_matrix = heater_diag_matrix.to(dtype=mzi_tf_matrix.dtype)

        return torch.matmul(heater_diag_matrix, mzi_tf_matrix)

    def get_heaters_quantization_error(self):
        """Return the heater quantization errors of the block.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Quantization errors of the
            external heater and the MZI heater.
        """
        qe_heater = self._heater.get_quantization_error()
        qe_mzi = self._mzi.get_heater_quantization_error()
        return qe_heater, qe_mzi

    def get_heaters_snr(self):
        """Return the average heater SNR of the block.

        The result is the average of the external heater SNR and the internal
        MZI heater SNR.

        Returns:
            torch.Tensor: Average SNR across both heaters.
        """
        snr_heater = torch.as_tensor(self._heater.get_dac_snr())
        snr_mzi = torch.as_tensor(self._mzi.get_heater_snr())
        snr = torch.stack([snr_heater, snr_mzi], dim=0).mean(dim=0)
        return snr

    def get_heaters_sqnr(self):
        """Return the average heater SQNR of the block.

        The result is the average of the external heater SQNR and the internal
        MZI heater SQNR.

        Returns:
            torch.Tensor: Average SQNR across both heaters.
        """
        sqnr_heater = torch.as_tensor(self._heater.get_dac_sqnr())
        sqnr_mzi = torch.as_tensor(self._mzi.get_heater_sqnr())
        sqnr = torch.stack([sqnr_heater, sqnr_mzi], dim=0).mean(dim=0)
        return sqnr

    def set_heater_params(self, dac_params):
        """Enable DAC settings for the heaters in the block.

        Args:
            dac_params: DAC parameter object to apply to both the external
                heater and the internal MZI heater.
        """
        self._heater.set_use_dac(True, dac_params)
        self._mzi.set_heater_params(dac_params)
