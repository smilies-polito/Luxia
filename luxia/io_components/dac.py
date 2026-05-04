from luxia.util.util_math.ste_roundings import apply_ste_rounding
from luxia.params.dac_params import DacParams
from luxia.util.log_config import lx_log
from luxia.lx_module import LxModule
from luxia.types import round_t
import torch
import copy


class Dac(LxModule):
    def __init__(self, params: DacParams, debug: bool = False):
        """

        Args:
            n_bits: bit resolution of the DAC
            max_analog_ref: Maximum reference value for the output analog
                signal (default: 1.0)
            min_analog_ref: Minimum reference value for the output analog
                signal (default: 0.0)
            is_floored: Floor or Ceil DAC (default: True (Floor))
            debug: Flag for debug mode, print messages (default: False)
        """
        super(Dac, self).__init__(debug=debug)
        self._max_val = 0.0
        self._quantum = 0.0
        self._sqnr = 0.0
        self._snr = 0.0
        self._q_error = 0
        self.set_params(params)

    def forward(self, x_float_digital):

        # Check tensor bounds element-wise rather than the whole tensor
        if torch.any(x_float_digital < 0) or torch.any(x_float_digital > 1):
            raise ValueError("Input values must be between 0 and 1")

        x_int = x_float_digital * self._max_val
        # Apply non differentiable rounding
        x_digital = apply_ste_rounding(x_int, self._params.round_type)
        # multiply the quanto n times to get the analog value
        y_analog = self._quantum * x_digital

        # Compute the quantization error and SNR
        y_norm = x_digital / self._max_val  # Original Amplitude Values
        qns = x_float_digital - y_norm  # Error Singnal Amplitude Values
        self._q_error = abs(qns)
        power_signal = torch.mean(torch.square(x_float_digital))
        power_error = torch.mean(torch.square(self._q_error))
        self._snr = 10 * torch.log10(power_signal / power_error)

        if self._debug:
            # Safely get first vector (handles 1D vs batched)
            if x_float_digital.dim() >= 1:
                x_first = x_float_digital[0]
                y_first = y_analog[0]
                q_first = self._q_error[0]
            else:
                x_first = x_float_digital
                y_first = y_analog
                q_first = self._q_error

            lx_log.debug("=== DAC debug ===")
            lx_log.debug(f"x_float_digital (first vec):\n{x_first}")
            lx_log.debug(f"y_analog (first vec):\n{y_first}")
            lx_log.debug(f"y_norm (first vec):\n{y_norm[0] if y_norm.dim() >= 1 else y_norm}")
            lx_log.debug(f"Quantization Error (first vec):\n{q_first}")
            lx_log.debug(f"SNR (dB): {self._snr}")
            lx_log.debug(f"SQNR (dB): {self._sqnr}")

            torch.set_printoptions(profile="default")
        return y_analog

    def set_rounding_type(self, round_type: round_t):
        self._params.round_type = round_type

    def set_params(self, params: DacParams):
        self._params = copy.deepcopy(params)
        self._max_val = 2 ** self._params.n_bits - 1
        self._quantum = (self._params.max_analog_ref -
                         self._params.min_analog_ref) / self._max_val
        self._sqnr = 20 * torch.log10(torch.tensor(2.0 ** self._params.n_bits))

        return None

    def get_sqnr(self):
        return self._sqnr

    def get_snr(self):
        return self._snr

    def get_quantization_error(self):
        return self._q_error

    def get_params(self):
        return self._params
