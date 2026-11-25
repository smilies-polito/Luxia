from __future__ import annotations

import copy

import torch

from luxia.lx_module import LxModule
from luxia.params.adc_params import AdcParams
from luxia.types import round_t
from luxia.util.log_config import lx_log
from luxia.util.util_math.ste_roundings import apply_ste_rounding


class Adc(LxModule):
    """
    Analog-to-digital converter model.

    Design notes:
    - `params` is the single configuration source of truth.
    - `max_val`, `quantum`, and `sqnr` are derived from `params`, so they are
      exposed as properties instead of duplicated mutable state.
    - `snr` and `quantization_error` are runtime metrics produced by the most
      recent `forward()` call, so they remain stored state.
    """

    def __init__(self, params: AdcParams, debug: bool = False):
        super().__init__(debug=debug)

        # Runtime metrics from the last forward pass.
        self._snr = torch.tensor(float("nan"))
        self._q_error = torch.tensor([])

        # Store validated configuration through the managed attribute.
        self._params: AdcParams
        self.params = params

    @property
    def params(self) -> AdcParams:
        """
        Return a defensive copy of the ADC configuration.

        Returning a copy avoids accidental external mutation such as:
            adc.params.n_bits = 6

        That kind of mutation would bypass validation and make state changes
        harder to reason about.
        """
        return copy.deepcopy(self._params)

    @params.setter
    def params(self, params: AdcParams) -> None:
        """
        Validate and store ADC configuration.

        A deep copy keeps ownership local to the module and prevents aliasing
        bugs when the same params object is shared elsewhere in the codebase.
        """
        params_copy = copy.deepcopy(params)
        self._validate_params(params_copy)
        self._params = params_copy

    @property
    def max_val(self) -> int:
        """
        Maximum representable integer code for the configured ADC resolution.
        """
        return (1 << self._params.number_of_bits) - 1

    @property
    def quantum(self) -> float:
        """
        Quantization step size in the analog domain.
        """
        return (
            self._params.max_analog_reference - self._params.min_analog_reference
        ) / self.max_val

    @property
    def sqnr(self) -> torch.Tensor:
        """
        Ideal signal-to-quantization-noise ratio implied by the ADC bit depth.

        This depends only on the configuration, not on the input tensor.
        """
        return 20.0 * torch.log10(
            torch.tensor(
                2.0 ** self._params.number_of_bits,
                dtype=torch.float32,
            )
        )

    @property
    def snr(self) -> torch.Tensor:
        """
        Measured SNR from the most recent forward pass.
        """
        return self._snr

    @property
    def quantization_error(self) -> torch.Tensor:
        """
        Absolute normalized quantization error from the most recent forward pass.
        """
        return self._q_error

    def set_rounding_type(self, round_type: round_t) -> None:
        """
        Update only the rounding mode while keeping the rest of the ADC
        configuration unchanged.
        """
        self._params.round_type = round_type

    def set_params(self, params: AdcParams) -> None:
        """
        Backward-compatible wrapper around the `params` property.

        Keeping this method avoids breaking existing call sites while still
        moving the implementation toward a more Pythonic API.
        """
        self.params = params

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        """
        Quantize the analog input and return the normalized digital output.

        Returned values are normalized to [0, 1] with respect to the ADC code
        range, matching the original class behavior.
        """
        # Map analog values to the integer-code domain.
        x_int = (x_analog - self._params.min_analog_reference) / self.quantum

        # Quantize in the integer domain using the configured rounding rule.
        y_digital = apply_ste_rounding(
            x_int,
            self._params.round_type,
        )

        # Normalize both the input-domain code and the quantized code by the
        # maximum representable code. This matches the original metric flow.
        x_analog_norm = x_int / self.max_val
        y_analog_norm = y_digital / self.max_val

        # Quantization error is stored for external inspection and metrics.
        q_error = torch.abs(x_analog_norm - y_analog_norm)
        self._q_error = q_error

        # Compute SNR from normalized signal and error power.
        power_signal = torch.mean(torch.square(x_analog_norm))
        power_error = torch.mean(torch.square(q_error))

        # Guard the zero-error case to avoid invalid log10 behavior.
        if torch.any(power_error == 0):
            self._snr = torch.tensor(
                float("inf"),
                dtype=x_analog.dtype if x_analog.is_floating_point() else torch.float32,
                device=x_analog.device,
            )
        else:
            self._snr = 10.0 * torch.log10(power_signal / power_error)

        if self._debug:
            self._log_debug(
                x_analog=x_analog,
                x_int=x_int,
                y_digital=y_digital,
                x_analog_norm=x_analog_norm,
                y_analog_norm=y_analog_norm,
                q_error=q_error,
            )

        return y_analog_norm

    def get_metrics(self) -> dict[str, torch.Tensor | float]:
        """
        Return ADC metrics through the canonical module metrics API.

        `average_qerror` is reported as a Python float for easier logging and
        serialization, while tensor-valued metrics remain tensors.
        """
        qe = self.quantization_error
        average_qerror = (
            float(qe.mean().item())
            if isinstance(qe, torch.Tensor) and qe.numel() > 0
            else 0.0
        )

        return {
            "sqnr": self.sqnr,
            "snr": self.snr,
            "average_qerror": average_qerror,
        }

    @staticmethod
    def _validate_params(params: AdcParams) -> None:
        """
        Validate ADC configuration invariants in one place.

        Centralizing validation keeps constructor, property setter, and any
        future configuration path consistent.
        """
        if params.number_of_bits <= 0:
            raise ValueError("number_of_bits must be > 0")

        if params.max_analog_reference <= params.min_analog_reference:
            raise ValueError(
                "max_analog_reference must be greater than min_analog_reference"
            )

    def _log_debug(
        self,
        *,
        x_analog: torch.Tensor,
        x_int: torch.Tensor,
        y_digital: torch.Tensor,
        x_analog_norm: torch.Tensor,
        y_analog_norm: torch.Tensor,
        q_error: torch.Tensor,
    ) -> None:
        """
        Emit detailed debug information while keeping `forward()` readable.
        """
        torch.set_printoptions(profile="full")
        lx_log.setLevel("DEBUG")

        first_x_analog = x_analog[0] if x_analog.dim() >= 1 else x_analog
        first_x_int = x_int[0] if x_int.dim() >= 1 else x_int
        first_y_digital = y_digital[0] if y_digital.dim() >= 1 else y_digital
        first_x_analog_norm = (
            x_analog_norm[0] if x_analog_norm.dim() >= 1 else x_analog_norm
        )
        first_y_analog_norm = (
            y_analog_norm[0] if y_analog_norm.dim() >= 1 else y_analog_norm
        )
        first_q_error = q_error[0] if q_error.dim() >= 1 else q_error

        lx_log.debug(f"x_analog (first vec):\n{first_x_analog}")
        lx_log.debug(f"x_int (first vec):\n{first_x_int}")
        lx_log.debug(f"x_digital (first vec):\n{first_y_digital}")
        lx_log.debug(f"x_analog_norm (first vec):\n{first_x_analog_norm}")
        lx_log.debug(f"y_analog (first vec):\n{first_y_analog_norm}")
        lx_log.debug(f"Quantization Error (first vec):\n{first_q_error}")
        lx_log.debug(f"SNR (dB): {self.snr}")
        lx_log.debug(f"SQNR (dB): {self.sqnr}")

        torch.set_printoptions(profile="default")
