"""Parameters for ClementsBellBlock."""

from dataclasses import dataclass
from typing import Optional, List, Union

from luxia.params import HeaterParams, BsMmiParams


@dataclass
class ClementsBellBlockParams:
    """Parameters for the Clements-Bell block (dual-heater MZI).

    Matches ClementsBlockParams structure exactly for consistency.

    Args:
        n_blocks: Number of parallel Clements-Bell blocks (determines mesh dimension).
        mmi_params: MMI parameters for input/output MMIs. Optional list of BsMmiParams.
        heater_params: Heater parameters for both top/bottom MZI heaters. Optional list of HeaterParams.
    """
    n_blocks: int = 1
    
    mmi_params: Optional[List[BsMmiParams]] = None
    heater_params: Optional[List[HeaterParams]] = None
