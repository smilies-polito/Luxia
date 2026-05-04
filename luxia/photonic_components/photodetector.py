import torch
from luxia import LxModule
from luxia.util.log_config import lx_log
from luxia.params import PhotodetectorParams
from typing import Optional


class Photodetector(LxModule):
    """
    Photodetector model converting optical field to electrical current.

    Implements square-law detection:
        I = R * P + I_dark

    Features:
    - Optional input normalization to physical power
    - Dark current offset
    - Lower bound clamping for numerical stability

    Assumptions:
    - optical_field is complex-valued or magnitude-compatible
    - Input shape is (..., n_outputs)
    - Responsivity and currents are scalar (broadcastable)
    """

    def __init__(self, pd_params: Optional[PhotodetectorParams] = None, debug: bool = False):
        """
        Initialize photodetector.

        Args:
            pd_params: Photodetector configuration parameters
            debug: Enables verbose logging
        """
        super().__init__(debug=debug)

        # Defensive initialization of parameters
        if pd_params is None:
            pd_params = PhotodetectorParams()

        # Register constants as buffers (non-trainable, device-aware)
        self.register_buffer("responsivity", torch.tensor(pd_params.responsivity))  # [A/W]
        self.register_buffer("dark_current", torch.tensor(pd_params.dark_current))  # [A]
        self.register_buffer("P_ref", torch.tensor(pd_params.P_ref))                # [W]

        # Flag: whether input is normalized (dimensionless) or physical power
        self.norm_in = pd_params.norm_input

    def forward(self, optical_field: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Converts optical field to electrical current.

        Args:
            optical_field: Complex optical field tensor

        Returns:
            Electrical current tensor [A]
        """

        # Compute optical power via square-law detection: |E|^2
        optical_power = torch.abs(optical_field) ** 2  # [arb] or [W]

        # Convert to physical power if required
        if self.norm_in:
            # Scale normalized power to physical units
            power_phys = optical_power * self.P_ref  # [W]
        else:
            # Assume already in physical units
            power_phys = optical_power  # [W]

        # Photocurrent generation: I_photo = R * P
        i_photo = self.responsivity * power_phys  # [A]

        # Add dark current offset
        i_total = i_photo + self.dark_current  # [A]

        # Numerical stability: clamp very small values to dark current
        # Prevents negative or near-zero currents due to numerical noise
        eps = 1e-9

        i_total = torch.where(
            i_total < self.dark_current + eps,
            self.dark_current.expand_as(i_total),  # ensure shape consistency
            i_total,
        )

        # Debug logging (side effects: logging + global print settings)
        if self._debug:
            torch.set_printoptions(profile="full")
            lx_log.setLevel("DEBUG")

            # Extract representative sample for readability
            if optical_power.dim() >= 2:
                # Use first batch element
                in_power_0 = optical_power[0]     # shape: (n_outputs,)
                out_current_0 = i_total[0]       # shape: (n_outputs,)
            else:
                # Scalar or single vector case
                in_power_0 = optical_power
                out_current_0 = i_total

            lx_log.debug(f" Input Optical power (sample 0):\n{in_power_0}")
            lx_log.debug(f" Output Current (sample 0):\n{out_current_0}")

            # Restore default print configuration
            torch.set_printoptions(profile="default")

        return i_total
