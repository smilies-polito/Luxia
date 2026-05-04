from dataclasses import dataclass
from luxia.params import ClementsBlockParams
from luxia.params import HeaterParams

@dataclass
class ClementsLayerParams:
    n_inputs: int
    is_even_layer: bool = True
    opt_ps: bool = False
    clements_blk_params: ClementsBlockParams = None
    opt_heater_params: HeaterParams = None
