from luxia.photonic_meshes.mesh.photonic_layer import PhotonicLayer
from luxia.photonic_meshes.clements.clements_block import ClementsBlock
from luxia.params import ClementsBlockParams, ClementsLayerParams
from typing import Optional, List, Dict, Any, Tuple
from luxia.params import HeaterParams
import torch


class ClementsLayer(PhotonicLayer):
    """
      Fig. 1 Even number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A ------------------C
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B ------------------C

        ========================================================
        Fig. 2 Odd number of Inputs
          a) Even Layer:              b) Odd Layer:
        A -[w]-- --[w]-- ---C       A ------------------C
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A -[w]-- --[w]-- ---C       A ------ ------- ---D
                X       X
        B ------ ------- ---D       B -[w]-- --[w]-- ---C
                                            X       X
        A ------------------C       A ------ ------- ---D

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

    def __init__(self, params: ClementsLayerParams, debug: bool = False):
        clements_block = ClementsBlock(
            cb_params=params.clements_blk_params,
        )

        super(ClementsLayer, self).__init__(
            n_inputs=params.number_of_inputs,
            is_even_layer=params.is_even_layer,
            opt_ps=params.use_optional_blocks,
            photonic_block=clements_block,
            debug=debug,
        )

    def set_params(self, params: ClementsLayerParams) -> None:
        blk = self.get_photonic_block()
        blk.set_params(params.clements_blk_params)

    def get_params(self) -> ClementsLayerParams:
        blk = self.get_photonic_block()
        blk_params: ClementsBlockParams = blk.get_params()
        return ClementsLayerParams(
            number_of_inputs=self.get_n_inputs(),
            is_even_layer=self.get_is_even_layer(),
            use_optional_blocks=self.get_opt_ps(),
            clements_blk_params=blk_params,
        )

    # --- Heaters / DAC metrics -------------------------------------------------

    def get_heater_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated heater metrics for the underlying ClementsBlock.

        The returned dict has keys:
          - 'heater_0': {resolution_bits, snr_db, sqnr_db, quantization_error, power}
          - 'heater_1': {resolution_bits, snr_db, sqnr_db, quantization_error, power}
        """
        blk = self.get_photonic_block()
        metrics = blk.get_heater_metrics()
        return metrics

    def get_heaters_quantization_error(self) -> Tuple[Any, Any]:
        """
        Convenience wrapper returning quantization_error tensors for both heaters.
        """
        metrics = self.get_heater_metrics()
        qe_0 = metrics["heater_0"]["quantization_error"]
        qe_1 = metrics["heater_1"]["quantization_error"]
        return qe_0, qe_1

    def get_heaters_snr(self) -> Tuple[float, float]:
        """
        Convenience wrapper returning SNR for both heaters.
        """
        metrics = self.get_heater_metrics()
        snr_0 = float(metrics["heater_0"]["snr_db"])
        snr_1 = float(metrics["heater_1"]["snr_db"])
        return snr_0, snr_1

    def get_heaters_sqnr(self) -> Tuple[float, float]:
        """
        Convenience wrapper returning SQNR for both heaters.
        """
        metrics = self.get_heater_metrics()
        sqnr_0 = float(metrics["heater_0"]["sqnr_db"])
        sqnr_1 = float(metrics["heater_1"]["sqnr_db"])
        return sqnr_0, sqnr_1

    # NOTE: This function only works with the cpp library
    def get_power_consumption(self) -> float:
        blk = self.get_photonic_block()
        power = blk.get_power_consumption()
        return float(power)

    def get_phase_shifts(self) -> torch.Tensor:
        blk = self.get_photonic_block()
        return torch.stack([blk._heater.phase_shift,
                            blk._mzi._heater.phase_shift], dim=0)

    def set_phase_shifts(self, phases: torch.Tensor) -> None:
        blk = self.get_photonic_block()
        blk._heater.set_phase_shift(phases[0])
        blk._mzi._heater.set_phase_shift(phases[1])

    def set_heaters_parameters(self, params: HeaterParams) -> None:
        """
        If you had a C++ method set_heater_params on ClementsBlock, call it here.
        Otherwise, update through ClementsBlockParams and set_params.
        """
        blk = self.get_photonic_block()
        # Example placeholder; adjust to your actual C++ API:
        # blk.set_heater_params(params)
        raise NotImplementedError(
            "set_heaters_parameters needs a matching C++ API (set_heater_params) on ClementsBlock.")
