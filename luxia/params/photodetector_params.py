from dataclasses import dataclass


@dataclass
class PhotodetectorParams:
    responsivity: float = 1
    dark_current: float = 0.0
    norm_input: bool = False
    P_ref: float = 1e-3
