from dataclasses import dataclass
from luxia.params import FldzhyanBlockParams
from luxia.params import HeaterParams


@dataclass
class FldzhyanLayerParams:
    """
    Parameters for a single Fldzhyan layer.

    A Fldzhyan layer contains:
      - One FldzhyanBlock (with a single inner heater group).
      - Optional edge heaters at the top/bottom boundaries, controlled by
        use_optional_blocks and opt_heater_params.

    The layer follows the Fldzhyan interferometer topology:

        Suren A. Fldzhyan, Mikhail Yu. Saygin, and Stanislav S. Straupe,
        "Optimal design of error-tolerant reprogrammable multiport interferometers",
        Optics Letters 45(9), 2632–2635 (2020).
        DOI: 10.1364/OL.385433
    """

    number_of_inputs: int
    is_even_layer: bool = True
    use_optional_blocks: bool = False
    fldzhyan_blk_params: FldzhyanBlockParams = None
    opt_heater_params: list[HeaterParams] = None

    def __post_init__(self) -> None:
        if self.opt_heater_params is not None:
            if not isinstance(self.opt_heater_params, list):
                raise ValueError(
                    "opt_heater_params must be None or a list of HeaterParams"
                )
            if len(self.opt_heater_params) > 2:
                raise ValueError(
                    "opt_heater_params must contain at most 2 HeaterParams "
                    "(top and/or bottom edge heaters)"
                )
