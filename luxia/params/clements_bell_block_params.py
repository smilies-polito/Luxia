"""Parameters for ClementsBellBlock."""

from dataclasses import dataclass
from typing import Optional, List, Union

from luxia.params import HeaterParams, MmiParams


@dataclass
class ClementsBellBlockParams:
    """Parameters for the Clements-Bell block (dual-heater MZI).

    Matches ClementsBlockParams structure exactly for consistency.

    Args:
        n_blocks: Number of parallel Clements-Bell blocks (determines mesh dimension).
        mmi_params: MMI parameters for input/output MMIs. Optional list of MmiParams.
        heater_params: Heater parameters for both top/bottom MZI heaters. Optional list of HeaterParams.
    """
    number_of_blocks: int = 1
    
    mmi_params: Optional[List[MmiParams]] = None
    heater_params: Optional[List[HeaterParams]] = None
