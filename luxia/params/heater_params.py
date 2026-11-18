from dataclasses import dataclass
from typing import Optional

from .dac_params import DacParams


@dataclass
class HeaterParams:
    number_of_heaters: int = 1
    dac_params: Optional[DacParams] = None
    use_dac: bool = False
    power: float = 20e-3          # power per 2π per heater in Watts
    max_phase_factor: float = 1.0  # fraction of full 2π range (0 < f <= 1)

