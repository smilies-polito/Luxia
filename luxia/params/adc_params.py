from dataclasses import dataclass
from luxia.types import round_t


@dataclass
class AdcParams:
    number_of_bits: int = 8
    max_analog_reference: float = 1.0
    min_analog_reference: float = 0.0
    round_type: round_t = round_t.FLOOR
