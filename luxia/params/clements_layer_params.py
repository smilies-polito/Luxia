from dataclasses import dataclass
from luxia.params import ClementsBlockParams
from luxia.params import HeaterParams

@dataclass
class ClementsLayerParams:
    number_of_inputs: int
    is_even_layer: bool = True
    use_optional_blocks: bool = False
    clements_blk_params: ClementsBlockParams = None
    opt_heater_params: HeaterParams = None
