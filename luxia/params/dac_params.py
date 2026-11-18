from dataclasses import dataclass
from luxia.types import round_t


@dataclass
class DacParams:
    number_of_bits: int = 8
    min_analog_reference: float = 0.0
    max_analog_reference: float = 1.0
    min_digital_reference: float = 0.0
    max_digital_reference: float = 1.0
    round_type: round_t = round_t.FLOOR

