from dataclasses import dataclass
from luxia.params import ClementsBlockParams, DacParams, ClementsLayerParams, HeaterParams, BsMmiParams
from typing import Optional, List

@dataclass
class ClementsMeshParams:
    n_inputs: int
    n_layers: int
    mmis_params: Optional[List[BsMmiParams]] = None
    heater_params: Optional[List] = None
    is_matmul: bool = False
    window_size: int = 1
