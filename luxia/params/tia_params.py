from dataclasses import dataclass

@dataclass
class TiaParams:
    max_voltage: float = 1.0
    tia_gain: float = 1.0          # fixed hardware R_tia
    cutoff_current: float = 1e-3   # initial guess
    trainable_cutoff: bool = True  # learn dynamic range

