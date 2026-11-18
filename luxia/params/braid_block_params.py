from dataclasses import dataclass
from typing import Optional
from luxia.params.bs_params import CrossParams


@dataclass
class BraidBlockParams:
    """
    Parameters for a single BraidBlock (2×2 passive crossing cell).

    A BraidBlock contains:
      - One crossing configuration, described by a single CrossParams object.
        This models the insertion loss and crosstalk of the waveguide crossings.
      - No heaters or phase shifters (passive component only).

    This differs from ClementsBlockParams and FldzhyanBlockParams, which include
    heater and MMI parameters. The BraidBlock is a purely passive crossing layer
    in the braid mesh architecture.

    Reference:
        Federico Marchesin et al., "Braided interferometer mesh for robust
        photonic matrix-vector multiplications with non-ideal components",
        Optics Express 33(2), 2227–2246 (2025).
        DOI: 10.1364/OE.547206
    """

    number_of_blocks: int = 1
    cross_params: Optional[CrossParams] = None

    def __post_init__(self) -> None:
        # Ensure cross_params is either None or a single object, not a list.
        # This enforces the BraidBlock design at the Python level.
        if self.cross_params is not None and isinstance(self.cross_params, list):
            raise ValueError(
                "BraidBlockParams.cross_params must be a single CrossParams "
                "object, not a list. For BraidBlock, use one shared crossing config."
            )
        
        # Default to ideal crossing if not provided
        if self.cross_params is None:
            self.cross_params = CrossParams()
