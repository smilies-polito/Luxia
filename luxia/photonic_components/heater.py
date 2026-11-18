import torch
import torch.nn as nn
from typing import Optional

from luxia import LxModule
from luxia.io_components import Dac
from luxia.params import DacParams, HeaterParams


class Heater(LxModule):
    def __init__(self, heater_params: Optional[HeaterParams] = None, debug: bool = False):
        super().__init__(debug=debug)

        if heater_params is None:
            heater_params = HeaterParams(number_of_heaters=1)
        self._heater_params = heater_params

        self._dac = Dac(DacParams(
            number_of_bits=8,
            min_analog_reference=0.0,
            max_analog_reference=1.0,
            min_digital_reference=0.0,
            max_digital_reference=1.0,
        ))

        if self._heater_params.use_dac and self._heater_params.dac_params is not None:
            self._dac.set_params(self._heater_params.dac_params)
        else:
            self._heater_params.use_dac = False

        n_heaters = self._heater_params.number_of_heaters
        self.phase_shift = nn.Parameter(torch.rand(n_heaters))
        self._power_consumption: float = 0.0

    def forward(self, wave_input: torch.Tensor) -> torch.Tensor:
        if self._debug:
            print("Heater forward called with input:\n", wave_input)

        phase_factor = self.get_phases().to(wave_input.device)
        wave_output = wave_input * phase_factor

        if self._debug:
            print("Heater output:\n", wave_output)

        return wave_output

    @property
    def is_dac_used(self) -> bool:
        return self._heater_params.use_dac

    def set_use_dac(self, use_dac: bool, params: Optional[DacParams] = None) -> None:
        self._heater_params.use_dac = use_dac
        if not use_dac:
            return

        if params is not None:
            self._dac.set_params(params)
            self._heater_params.dac_params = params
        elif self._heater_params.dac_params is not None:
            self._dac.set_params(self._heater_params.dac_params)

    def set_phase_shift(self, phase_shift_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            self.phase_shift.copy_(phase_shift_tensor)

    def get_phase_shift(self) -> torch.Tensor:
        return self.phase_shift.detach().clone()

    def get_phases(self) -> torch.Tensor:
        two_pi = 2.0 * torch.pi
        angle = self.phase_shift
        phi = torch.remainder(angle, two_pi)

        if self.is_dac_used:
            phi_analog = self._dac(phi)
            p = self._dac.get_params()
            min_ref = p.min_analog_reference
            max_ref = p.max_analog_reference
            if max_ref != min_ref:
                phi = (phi_analog - min_ref) / (max_ref - min_ref)
                phi = phi * two_pi

        if self._heater_params.power > 0.0:
            phi_norm = phi / two_pi
            power = (phi_norm * self._heater_params.power).sum().item()
            self._power_consumption = power

        return torch.exp(1j * phi)

    def get_quantization_error(self) -> torch.Tensor:
        if not self.is_dac_used:
            return torch.zeros_like(self.phase_shift)
        return self._dac.get_quantization_error()

    def get_dac_resolution(self) -> Optional[int]:
        if self.is_dac_used:
            return self._dac.get_params().number_of_bits
        return None

    def get_dac_snr(self) -> float:
        return self._dac.get_snr() if self.is_dac_used else 0.0

    def get_dac_sqnr(self) -> float:
        return self._dac.get_sqnr() if self.is_dac_used else 0.0

    def get_dac_params(self) -> Optional[DacParams]:
        return self._dac.get_params() if self.is_dac_used else None

    def get_params(self) -> HeaterParams:
        return self._heater_params

    def get_power_consumption(self) -> float:
        return self._power_consumption

    def set_power_consumption(self, power: float) -> None:
        self._power_consumption = float(power)
