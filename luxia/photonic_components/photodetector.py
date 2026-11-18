import torch
from luxia import LxModule
from luxia.util.log_config import lx_log
from luxia.params import PhotodetectorParams
from typing import Optional


class Photodetector(LxModule):
    def __init__(self, pd_params: Optional[PhotodetectorParams] = None, debug: bool = False):
        super().__init__(debug=debug)
        if pd_params is None:
            pd_params = PhotodetectorParams()
        self.register_buffer("responsivity", torch.tensor(pd_params.responsivity))
        self.register_buffer("dark_current", torch.tensor(pd_params.dark_current))
        self.register_buffer("P_ref", torch.tensor(pd_params.P_ref))
        self.norm_in = pd_params.norm_input

        # Store params object so we can update it later
        self._params = pd_params

    def forward(self, optical_field: torch.Tensor) -> torch.Tensor:
        # optical_field: assume shape (batch, n_outputs) at mesh output
        optical_power = torch.abs(optical_field) ** 2  # [arb]

        if self.norm_in:
            power_phys = optical_power * self.P_ref  # [W]
        else:
            power_phys = optical_power               # already [W]

        i_photo = self.responsivity * power_phys  # [A]
        i_total = i_photo + self.dark_current

        # Clamp very small values to dark current
        eps = 1e-9
        i_total = torch.where(
            i_total < self.dark_current + eps,
            self.dark_current.expand_as(i_total),
            i_total,
        )

        if self._debug:
            torch.set_printoptions(profile="full")
            lx_log.setLevel("DEBUG")

            if optical_power.dim() >= 2:
                # Log full vectors for sample 0
                in_power_0 = optical_power[0]    # (n_outputs,)
                out_current_0 = i_total[0]        # (n_outputs,)
            else:
                in_power_0 = optical_power
                out_current_0 = i_total

            lx_log.debug(f" Input Optical power (sample 0):\n{in_power_0}")
            lx_log.debug(f" Output Current (sample 0):\n{out_current_0}")

            torch.set_printoptions(profile="default")

        return i_total

    # -------------------------------------------------------------
    # Params access
    # -------------------------------------------------------------

    def set_params(self, pd_params: PhotodetectorParams) -> None:
        """
        Update photodetector parameters and refresh buffers.

        This is used at test time to inject non-ideal hardware configs
        (e.g. change responsivity, dark current, P_ref, norm_input).
        """
        self._params = pd_params

        # Update buffers so forward() uses the new values
        self.responsivity = torch.tensor(pd_params.responsivity,
                                         dtype=self.responsivity.dtype,
                                         device=self.responsivity.device)
        self.dark_current = torch.tensor(pd_params.dark_current,
                                         dtype=self.dark_current.dtype,
                                         device=self.dark_current.device)
        self.P_ref = torch.tensor(pd_params.P_ref,
                                  dtype=self.P_ref.dtype,
                                  device=self.P_ref.device)
        self.norm_in = pd_params.norm_input

    def get_params(self) -> PhotodetectorParams:
        """Return the current photodetector parameters."""
        return self._params
