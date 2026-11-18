from copy import deepcopy
from typing import Any, Dict

import torch

from luxia.lx_module import LxModule
from luxia.params.dac_params import DacParams
from luxia.types import round_t
from luxia.util.log_config import lx_log
from luxia.util.util_math.ste_roundings import apply_ste_rounding


class Dac(LxModule):
    """
    @brief Digital-to-analog converter module.

    This module quantizes a normalized digital input tensor in the range
    [0, 1] and maps it to the configured analog output range.

    The module stores the last computed quantization metrics so that they
    can be queried through the canonical get_metrics() API.
    """

    def __init__(self, params: DacParams, debug: bool = False):
        """
        @brief Construct a DAC module.

        @param params
            DAC parameter object.
        @param debug
            Enable debug logging when True.
        """
        super(Dac, self).__init__(debug=debug)

        self._max_val = 0.0
        self._quantum = 0.0
        self._sqnr = torch.tensor(0.0, dtype=torch.float32)
        self._snr = torch.tensor(0.0, dtype=torch.float32)
        self._q_error = torch.tensor(0.0, dtype=torch.float32)
        self._params = deepcopy(params)

        self.set_params(params)

    def forward(self, x_float_digital: torch.Tensor) -> torch.Tensor:
        """
        @brief Apply DAC quantization and analog scaling.

        @param x_float_digital
            Input digital tensor. Each element shall lie in [0, 1].
        @return
            Quantized analog output tensor.
        @throws ValueError
            Raised if any input element is outside the range [0, 1].
        """
        if torch.any(x_float_digital < 0) or torch.any(x_float_digital > 1):
            raise ValueError("Input values must be between 0 and 1")

        x_int = x_float_digital * self._max_val

        # Apply STE-compatible rounding in the digital domain.
        x_digital = apply_ste_rounding(x_int, self._params.round_type)

        # Map the quantized digital code to the analog output domain.
        y_analog = self._quantum * x_digital

        # Reconstruct the normalized digital value after quantization.
        y_norm = x_digital / self._max_val

        # Store absolute quantization error for the last forward pass.
        qns = x_float_digital - y_norm
        self._q_error = torch.abs(qns)

        if x_float_digital.numel() == 0:
            self._snr = x_float_digital.new_zeros(())
            return y_analog

        # Compute SNR from signal and quantization-noise power.
        power_signal = torch.mean(torch.square(x_float_digital))
        power_error = torch.mean(torch.square(self._q_error))

        # Clamp powers to avoid invalid log10 operations.
        eps = torch.finfo(power_signal.dtype).eps
        power_signal = torch.clamp(power_signal, min=eps)
        power_error = torch.clamp(power_error, min=eps)

        self._snr = 10.0 * torch.log10(power_signal / power_error)

        if self.get_debug_mode():
            if x_float_digital.dim() >= 1:
                x_first = x_float_digital[0]
                y_first = y_analog[0]
                q_first = self._q_error[0]
                y_norm_first = y_norm[0]
            else:
                x_first = x_float_digital
                y_first = y_analog
                q_first = self._q_error
                y_norm_first = y_norm

            lx_log.debug("=== DAC debug ===")
            lx_log.debug(f"x_float_digital (first vec):\n{x_first}")
            lx_log.debug(f"y_analog (first vec):\n{y_first}")
            lx_log.debug(f"y_norm (first vec):\n{y_norm_first}")
            lx_log.debug(f"Quantization Error (first vec):\n{q_first}")
            lx_log.debug(f"SNR (dB): {self._snr}")
            lx_log.debug(f"SQNR (dB): {self._sqnr}")

            torch.set_printoptions(profile="default")

        return y_analog

    def set_rounding_type(self, round_type: round_t) -> None:
        """
        @brief Set the rounding mode used by the DAC.

        @param round_type
            Rounding mode to be used during quantization.
        """
        self._params.round_type = round_type

    def set_params(self, params: DacParams) -> None:
        """
        @brief Update DAC parameters.

        @param params
            New DAC parameter object.
        @throws ValueError
            Raised if the configured bit resolution is invalid.
        """
        self._params = deepcopy(params)
        self._max_val = float((2 ** self._params.number_of_bits) - 1)

        if self._max_val <= 0.0:
            raise ValueError("DAC number_of_bits must be >= 1")

        self._quantum = (
            self._params.max_analog_reference - self._params.min_analog_reference
        ) / self._max_val

        self._sqnr = 20.0 * torch.log10(
            torch.tensor(
                float(2 ** self._params.number_of_bits),
                dtype=torch.float32,
            )
        )

    def get_sqnr(self) -> torch.Tensor:
        """
        @brief Return theoretical SQNR in dB.

        @return
            SQNR tensor.
        """
        return self._sqnr

    def get_snr(self) -> torch.Tensor:
        """
        @brief Return measured SNR in dB from the last forward pass.

        @return
            SNR tensor.
        """
        return self._snr

    def get_quantization_error(self) -> torch.Tensor:
        """
        @brief Return quantization error from the last forward pass.

        @return
            Quantization-error tensor.
        """
        return self._q_error

    def get_average_qerror(self) -> torch.Tensor:
        """
        @brief Return the mean quantization error.

        @return
            Scalar tensor with the average quantization error.
        """
        if isinstance(self._q_error, torch.Tensor):
            if self._q_error.numel() == 0:
                return torch.tensor(0.0, dtype=torch.float32)

            return self._q_error.to(dtype=torch.float32).mean()

        return torch.tensor(float(self._q_error), dtype=torch.float32)

    def get_params(self) -> DacParams:
        """
        @brief Return current DAC parameters.

        @return
            Current DAC parameter object.
        """
        return self._params

    def get_metrics(self) -> Dict[str, Any]:
        """
        @brief Return DAC metrics in canonical dictionary form.

        @return
            Dictionary containing DAC metrics for aggregation.
        """
        return {
            "snr_db": self.get_snr(),
            "sqnr_db": self.get_sqnr(),
            # "quantization_error": self.get_quantization_error(),
            "average_qerror": self.get_average_qerror(),
        }
