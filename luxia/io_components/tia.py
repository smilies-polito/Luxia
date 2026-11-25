
import torch
from torch import nn
from luxia import LxModule
from luxia.util.log_config import lx_log
from luxia.params import TiaParams
from typing import Optional


class Tia(LxModule):
    """
    Transimpedance amplifier model with learnable cutoff current:

    - Linear region: V = R_tia * I  (R_tia fixed from TiaParams)
    - Symmetric saturation at ±V_max (soft using tanh for smooth gradients)
    - Cutoff current I_cut: current at which |V_lin| is mapped to V_max
      (I_cut is learned using scaled exponential with clamped parameter).
    
    Parameterization:
    - raw_value: Unconstrained parameter, clamped to [0, 1]
    - scale: Fixed multiplier (e.g., 20) for gradient boost
    - cutoff_current = exp(scale * (clamped(raw_value) - 1)) * fine_scale
    
    Coverage with scale=20: ~2 nA to 1 A (9+ orders of magnitude).
    """

    def __init__(
        self,
        tia_params: Optional[TiaParams] = None,
        debug: bool = False,
    ):
        super().__init__(debug=debug)

        if tia_params is None:
            tia_params = TiaParams()

        self._max_voltage = float(tia_params.max_voltage)
        self._scale = 20  # Gradient boost factor, covers ~2 nA to 1 A
        self._params = tia_params

        tia_gain = float(tia_params.tia_gain)
        self.register_buffer("tia_gain", torch.tensor(tia_gain))

        # === Cutoff current parameter (learnable) ===
        if tia_params.trainable_cutoff:
            print("TIa with trainable_cutoff")
            # Initialize uniformly in [0, 1]
            self.raw_value = nn.Parameter(torch.tensor(0.8))
            
            # Fine control: multiplicative adjustment (learnable)
            self.log_fine_scale = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
        else:
            # NOT TRAINABLE: Use TiaParams.cutoff_current directly
            base_cutoff = float(tia_params.cutoff_current)
            if base_cutoff <= 0.0:
                raise ValueError("cutoff_current must be > 0.")
            
            # Invert: psi = 1 + ln(I_cut) / scale, then clamp to [0, 1]
            base_cutoff_init = max(base_cutoff, 1e-12)
            raw_init = 1.0 + torch.log(torch.tensor(base_cutoff_init)) / self._scale
            raw_init = torch.clamp(raw_init, min=0.0, max=1.0)
            
            self.register_buffer("raw_value", raw_init)
            self.register_buffer("log_fine_scale", torch.tensor(0.0))

    @property
    def cutoff_current(self) -> torch.Tensor:
        """Positive cutoff current using exp(scale * (clamped - 1))."""
        # print(self.raw_value)
        clamped = torch.clamp(self.raw_value, min=0.0, max=1.0)
        base = torch.exp(self._scale * (clamped - 1.0))
        # print(self.log_fine_scale)
        fine_scale = torch.exp(self.log_fine_scale)
        return base * fine_scale

    @property
    def base_cutoff(self) -> torch.Tensor:
        """Base cutoff (without fine adjustment)."""
        clamped = torch.clamp(self.raw_value, min=0.0, max=1.0)
        return torch.exp(self._scale * (clamped - 1.0))

    @property
    def fine_scale(self) -> torch.Tensor:
        """Fine multiplicative adjustment factor."""
        return torch.exp(self.log_fine_scale)

    @property
    def raw_parameter(self) -> torch.Tensor:
        """Raw unconstrained parameter value (for monitoring)."""
        return self.raw_value

    def _compute_scale(self) -> torch.Tensor:
        """
        Compute scaling factor to map V_cut = R_tia * I_cut to V_max.
        
        Uses relative epsilon for numerical stability.
        """
        # print(self.cutoff_current)
        v_cut = self.tia_gain * self.cutoff_current
        
        # Relative epsilon for FP32/FP64 stability
        eps = torch.finfo(v_cut.dtype).eps * torch.max(
            torch.tensor(1.0, device=v_cut.device, dtype=v_cut.dtype),
            v_cut.abs().detach()
        )
        
        return self._max_voltage / (v_cut + eps)

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_current: Input current signal (batch_size, n_inputs), can be signed.

        Returns:
            output_voltage: Output voltage signal (batch_size, n_inputs)
        """
        v_lin = self.tia_gain * input_current
        scale = self._compute_scale()
        v_scaled = v_lin * scale
        output_voltage = self._max_voltage * torch.tanh(v_scaled)

        if self._debug:
            torch.set_printoptions(profile="full")
            lx_log.setLevel("DEBUG")

            if input_current.dim() >= 1:
                first_input = input_current[0]
                first_voltage = output_voltage[0]
            else:
                first_input = input_current
                first_voltage = output_voltage

            final_gain = float(self.tia_gain.detach().cpu().item())
            final_cutoff = float(self.cutoff_current.detach().cpu().item())
            final_raw = float(self.raw_parameter.detach().cpu().item())
            final_fine = float(self.fine_scale.detach().cpu().item())
            final_clamped = float(torch.clamp(self.raw_parameter, min=0.0, max=1.0).detach().cpu().item())

            lx_log.debug("=== TIA debug ===")
            lx_log.debug(f"Input current (first vec, first 8):\n{first_input}")
            lx_log.debug(f"Input current (Max Value):\n{first_input.max()}")
            lx_log.debug(f"Output voltage (first vec, first 8):\n{first_voltage}")
            lx_log.debug(f"Fixed TIA gain (R_tia, V/A): {final_gain}")
            lx_log.debug(f"Learned cutoff current I_cut (A-equivalent): {final_cutoff}")
            lx_log.debug(f"Raw parameter value (before clamp): {final_raw}")
            lx_log.debug(f"Clamped parameter value [0, 1]: {final_clamped}")
            lx_log.debug(f"Scale factor (for gradient boost): {self._scale}")
            lx_log.debug(f"Base cutoff (without fine): {final_cutoff / final_fine}")
            lx_log.debug(f"Fine scale (adjustment): {final_fine}")
            lx_log.debug(f"Max voltage (sat): {self._max_voltage}")
            lx_log.debug(f"Scaling factor (v_cut -> V_max): {scale.item()}")
            lx_log.debug(f"Effective resistance to use in hardware: {final_gain} Ohm")

            torch.set_printoptions(profile="default")
        return output_voltage

    # -------------------------------------------------------------
    # Params access
    # -------------------------------------------------------------

    def set_params(self, tia_params: TiaParams) -> None:
        """Update TIA parameters (test-time hardware config injection)."""
        self._params = tia_params
        self._max_voltage = float(tia_params.max_voltage)

        tia_gain = float(tia_params.tia_gain)
        self.register_buffer(
            "tia_gain",
            torch.tensor(tia_gain, dtype=self.tia_gain.dtype, device=self.tia_gain.device)
        )

        # Only update if NOT trainable
        if not tia_params.trainable_cutoff:
            base_cutoff = float(tia_params.cutoff_current)
            if base_cutoff <= 0.0:
                raise ValueError("cutoff_current must be > 0.")
            
            # Invert: psi = 1 + ln(I_cut) / scale, then clamp to [0, 1]
            base_cutoff_init = max(base_cutoff, 1e-12)
            raw_init = 1.0 + torch.log(torch.tensor(base_cutoff_init)) / self._scale
            raw_init = torch.clamp(raw_init, min=0.0, max=1.0)
            
            self.register_buffer("raw_value", raw_init)
            self.register_buffer("log_fine_scale", torch.tensor(0.0))

    def get_params(self) -> TiaParams:
        """Return the current TIA parameters."""
        return self._params

    # -------------------------------------------------------------
    # Training utilities
    # -------------------------------------------------------------

    @torch.no_grad()
    def clamp_raw_parameter(self) -> None:
        """
        Clamp raw_value to [0, 1] after optimizer.step().
        Call this in your training loop after each update.
        """
        self.raw_value.clamp_(min=0.0, max=1.0)

    def get_saturation_fraction(self, input_current: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        """
        Compute fraction of neurons in saturation region.
        
        Args:
            input_current: Input current tensor
            threshold: |v_scaled| > threshold considered saturated (default: 2.0 → tanh(2) ≈ 0.96)
        
        Returns:
            Fraction of saturated neurons [0, 1]
        """
        v_lin = self.tia_gain * input_current
        scale = self._compute_scale()
        v_scaled = v_lin * scale
        return (v_scaled.abs() > threshold).float().mean()
