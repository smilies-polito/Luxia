from dataclasses import dataclass
from typing import Optional, List
from luxia.params import HeaterParams, MmiParams

@dataclass
class ClementsBlockParams:
    number_of_blocks: int = 1
    mmi_params: Optional[List[MmiParams]] = None
    heater_params: Optional[List[HeaterParams]] = None
