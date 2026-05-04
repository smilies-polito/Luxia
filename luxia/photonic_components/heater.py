import torch
import torch.nn as nn
from typing import Optional

from luxia import LxModule
from luxia.io_components import Dac
from luxia.params import DacParams, HeaterParams


class Heater(LxModule):
    """
    Heater module representing a phase-shifting element.

    Applies a complex phase rotation to an input optical wave.
    Optionally includes DAC quantization and tracks power consumption.

    Assumptions:
    - phase_shift is expressed in radians.
    - Input tensor is complex-valued or compatible with complex multiplication.
    - Number of heaters matches last dimension (or broadcastable).
    """

    def __init__(self, heater_params: Optional[HeaterParams] = None, debug: bool = False):
        """
        Initialize Heater module.

        Args:
            heater_params: Configuration parameters for heater behavior.
            debug: Enables verbose debug logging if True.
        """
        super().__init__(debug=debug)

        # Ensure valid parameter object (defensive initialization)
        if heater_params is None:
            heater_params = HeaterParams(n_heaters=1)

        self._heater_params = heater_params

        # Initialize DAC with default parameters (safe defaults)
        self._dac = Dac(DacParams(
            n_bits=8,
            min_analog_ref=0.0,
            max_analog_ref=1.0,
            min_digital_ref=0.0,
            max_digital_ref=1.0,
        ))

        # Apply user-provided DAC configuration if enabled
        if self._heater_params.use_dac and self._heater_params.dac_params is not None:
            self._dac.set_params(self._heater_params.dac_params)
        else:
            # Enforce internal consistency: disable DAC if params are invalid
            self._heater_params.use_dac = False

        # Initialize learnable phase shifts (uniform random in [0,1))
        n_heaters = self._heater_params.n_heaters
        self.phase_shift = nn.Parameter(torch.rand(n_heaters))

        # Internal state: accumulated power consumption (scalar, float)
        self._power_consumption: float = 0.0

    def forward(self, wave_input: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Applies phase modulation to input wave.

        Args:
            wave_input: Input complex tensor.

        Returns:
            Phase-shifted output tensor.
        """
        if self._debug:
            print("Heater forward called with input:\n", wave_input)

        # Retrieve complex phase factors (exp(j*phi))
        phase_factor = self.get_phases().to(wave_input.device)

        # Element-wise modulation (broadcast assumed valid)
        wave_output = wave_input * phase_factor

        if self._debug:
            print("Heater output:\n", wave_output)

        return wave_output

    @property
    def is_dac_used(self) -> bool:
        """
        Indicates whether DAC quantization is enabled.
        """
        return self._heater_params.use_dac

    def set_use_dac(self, use_dac: bool, params: Optional[DacParams] = None) -> None:
        """
        Enable or disable DAC usage.

        Args:
            use_dac: Flag to enable/disable DAC.
            params: Optional DAC parameter override.

        Side effects:
            - Updates internal DAC configuration.
            - May modify stored HeaterParams.
        """
        self._heater_params.use_dac = use_dac

        if not use_dac:
            # Early exit: DAC disabled, no configuration needed
            return

        if params is not None:
            # Explicit override takes priority
            self._dac.set_params(params)
            self._heater_params.dac_params = params
        elif self._heater_params.dac_params is not None:
            # Reuse stored parameters
            self._dac.set_params(self._heater_params.dac_params)
        # Else: DAC enabled but no params → retains previous/default config

    def set_phase_shift(self, phase_shift_tensor: torch.Tensor) -> None:
        """
        Set phase shift values (no gradient tracking).

        Args:
            phase_shift_tensor: Tensor of phase values (radians).

        Note:
            Shape must match internal parameter.
        """
        with torch.no_grad():
            self.phase_shift.copy_(phase_shift_tensor)

    def get_phase_shift(self) -> torch.Tensor:
        """
        Get a detached copy of phase shifts.

        Returns:
            Tensor clone (no gradient link).
        """
        return self.phase_shift.detach().clone()

    def get_phases(self) -> torch.Tensor:
        """
        Compute complex phase factors exp(j * phi).

        Includes:
        - Phase wrapping to [0, 2π)
        - Optional DAC quantization
        - Power consumption estimation

        Returns:
            Complex tensor of phase factors.
        """
        two_pi = 2.0 * torch.pi

        # Wrap phase into principal domain [0, 2π)
        angle = self.phase_shift
        phi = torch.remainder(angle, two_pi)

        # Apply DAC quantization if enabled
        if self.is_dac_used:
            phi_analog = self._dac(phi)

            # Normalize based on DAC analog range
            p = self._dac.get_params()
            min_ref = p.min_analog_ref
            max_ref = p.max_analog_ref

            # Defensive check: avoid division by zero
            if max_ref != min_ref:
                phi = (phi_analog - min_ref) / (max_ref - min_ref)
                phi = phi * two_pi

        # Estimate power consumption (simple linear model)
        if self._heater_params.power > 0.0:
            phi_norm = phi / two_pi
            power = (phi_norm * self._heater_params.power).sum().item()

            # Store scalar result (side effect)
            self._power_consumption = power

        # Return complex exponential (Euler representation)
        return torch.exp(1j * phi)

    def get_quantization_error(self) -> torch.Tensor:
        """
        Get DAC quantization error.

        Returns:
            Zero tensor if DAC disabled, else DAC error tensor.
        """
        if not self.is_dac_used:
            return torch.zeros_like(self.phase_shift)

        return self._dac.get_quantization_error()

    def get_dac_resolution(self) -> Optional[int]:
        """
        Get DAC resolution in bits.

        Returns:
            Number of bits if DAC enabled, else None.
        """
        if self.is_dac_used:
            return self._dac.get_params().n_bits

        return None

    def get_dac_snr(self) -> float:
        """
        Get DAC signal-to-noise ratio.

        Returns:
            SNR if DAC enabled, else 0.0.
        """
        return self._dac.get_snr() if self.is_dac_used else 0.0

    def get_dac_sqnr(self) -> float:
        """
        Get DAC signal-to-quantization-noise ratio.

        Returns:
            SQNR if DAC enabled, else 0.0.
        """
        return self._dac.get_sqnr() if self.is_dac_used else 0.0

    def get_dac_params(self) -> Optional[DacParams]:
        """
        Get current DAC parameters.

        Returns:
            DAC parameters if enabled, else None.
        """
        return self._dac.get_params() if self.is_dac_used else None

    def get_params(self) -> HeaterParams:
        """
        Get heater configuration parameters.
        """
        return self._heater_params

    def get_power_consumption(self) -> float:
        """
        Get last computed power consumption.

        Returns:
            Scalar power value.
        """
        return self._power_consumption

    def set_power_consumption(self, power: float) -> None:
        """
        Manually set power consumption.

        Args:
            power: Scalar value.

        Note:
            This overrides internally computed value.
        """
        self._power_consumption = float(power)
