from dataclasses import dataclass
from typing import Optional, List
from luxia.params import HeaterParams, BsMmiParams

@dataclass
class ClementsBlockParams:
    n_blocks: int = 1
    mmi_params: Optional[List[BsMmiParams]] = None
    heater_params: Optional[List[HeaterParams]] = None
