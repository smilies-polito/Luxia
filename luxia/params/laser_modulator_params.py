from dataclasses import dataclass


@dataclass
class LaserModulatorParams:
    is_intensity_modulation: bool = False
    norm_input: bool = False
    P_ref: float = 1e-3
