from dataclasses import dataclass
from typing import Optional, List

from luxia.params import HeaterParams, MmiParams


@dataclass
class ClementsBellLayerParams:
    """Parameters for one Clements-Bell layer.

    Args:
    number_of_inputs: Number of input waveguides.
        is_even_layer: Whether this is an even or odd layer.
        opt_ps: Flag forwarded to PhotonicLayer.
        mmi_params: Optional list of MMI parameters for the Bell block.
        heater_params: Optional list of heater parameters for Bell blocks or
            heater-only layers.
        is_heater_layer: If True, build a straight dual-heater boundary layer
            instead of a Bell MZI layer.
    """
    number_of_inputs: int
    is_even_layer: bool = True
    mmi_params: Optional[List[MmiParams]] = None
    heater_params: Optional[List[HeaterParams]] = None
    is_heater_layer: bool = False
