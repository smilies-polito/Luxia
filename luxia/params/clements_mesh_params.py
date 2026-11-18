from dataclasses import dataclass
from luxia.params import ClementsBlockParams, DacParams, ClementsLayerParams, HeaterParams, MmiParams
from typing import Optional, List

@dataclass
class ClementsMeshParams:
    number_of_inputs: int
    number_of_layers: int
    mmis_params: Optional[List[MmiParams]] = None
    heater_params: Optional[List] = None
    is_matmul: bool = False
    window_size: int = 1
