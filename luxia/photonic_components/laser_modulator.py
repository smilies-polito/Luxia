import torch
from typing import Optional

from luxia import LxModule
from luxia.params import LaserModulatorParams
from luxia.util.log_config import lx_log


class LaserModulator(LxModule):
    """
    Single-analog-input laser modulator.

    Input:
        x_analog: real power fraction per channel (e.g. in [0, 1])

    Behavior:
        If norm_input = True:
            P_out = x_analog * P_ref
            E_out = sqrt(x_analog * P_ref)
        If norm_input = False:
            E_out = x_analog  (already in field units)
    """

    def __init__(
        self,
        lmod_params: Optional[LaserModulatorParams],
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        lmod_params = LaserModulatorParams() if lmod_params is None else lmod_params

        self._norm_input = lmod_params.norm_input
        self._P_ref = float(lmod_params.P_ref)

    def forward(self, x_analog: torch.Tensor) -> torch.Tensor:
        # x_analog: real power fraction in [0, 1]

        if self._debug:
            lx_log.setLevel("DEBUG")
            lx_log.debug(f"LaserModulator input power fraction:\n{x_analog[0]}")

        if self._norm_input:
            # We want P_out = x * P_ref, so E_out = sqrt(x * P_ref).
            # Add small epsilon to avoid sqrt(0) gradient issues.
            eps = 1e-12

            P_ref_tensor = torch.tensor(
                self._P_ref, dtype=x_analog.dtype, device=x_analog.device
            ).clamp_min(eps)

            x_clamped = x_analog.clamp_min(0.0)
            x_field = torch.sqrt(x_clamped * P_ref_tensor + eps)
        else:
            # Already in field units
            x_field = x_analog

        if self._debug:
            lx_log.debug(f"LaserModulator field amplitude after scaling:\n{x_field[0]}")

        # Map to complex optical field (same shape)
        y_complex = torch.complex(x_field, torch.zeros_like(x_field))

        if self._debug:
            power = (y_complex.real ** 2 + y_complex.imag ** 2)
            lx_log.debug(f"LaserModulator output field:\n{y_complex[0]}")
            lx_log.debug(f"LaserModulator output power |E|^2:\n{power[0]}")
            lx_log.debug(f"LaserModulator tot output power |E|^2: {torch.sum(power)}\n")

        return y_complex
