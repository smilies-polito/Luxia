from dataclasses import dataclass
from typing import Optional, List
from luxia.params import HeaterParams, MmiParams


@dataclass
class FldzhyanBlockParams:
    """
    Parameters for a single Fldzhyan block (2×2 MZI cell).

    A Fldzhyan block contains:
      - One heater group, described by a single HeaterParams object.
        HeaterParams.number_of_heaters determines how many physical heater elements
        are modeled inside the block.
      - One MMI splitter configuration, described by a single MmiParams object.

    This differs from ClementsBlockParams, which uses lists of heater/MMI
    parameters (typically two of each). Here, a single shared configuration
    is used for all heaters and MMIs in the block.
    """

    number_of_blocks: int = 1
    mmi_params: Optional[MmiParams] = None
    heater_params: Optional[HeaterParams] = None

    def __post_init__(self) -> None:
        # Ensure mmi_params and heater_params are either None or single objects,
        # not lists. This enforces the Fldzhyan design at the Python level.
        if self.mmi_params is not None and isinstance(self.mmi_params, list):
            raise ValueError(
                "FldzhyanBlockParams.mmi_params must be a single MmiParams "
                "object, not a list. For Fldzhyan, use one shared MMI config."
            )

        if self.heater_params is not None and isinstance(self.heater_params, list):
            raise ValueError(
                "FldzhyanBlockParams.heater_params must be a single HeaterParams "
                "object, not a list. For Fldzhyan, use one shared heater config."
            )
