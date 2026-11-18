from dataclasses import dataclass
from typing import Optional
from luxia.params.braid_block_params import BraidBlockParams


@dataclass
class BraidLayerParams:
    """
    Parameters for a single BraidLayer (passive crossing layer).

    A BraidLayer contains:
      - One BraidBlock (passive 2×2 crossing cell) with no heaters or phase shifters.
      - Structural parameters: n_inputs and is_even_layer.

    This differs from ClementsLayerParams and FldzhyanLayerParams, which include
    heater and phase shifter configurations. The BraidLayer is a purely passive
    crossing layer in the braid mesh architecture.

    Reference:
        Federico Marchesin et al., "Braided interferometer mesh for robust
        photonic matrix-vector multiplications with non-ideal components",
        Optics Express 33(2), 2227–2246 (2025).
        DOI: 10.1364/OE.547206

    Attributes:
        number_of_inputs: Number of inputs to the layer
        is_even_layer: True if this is an even-indexed layer in the mesh
        braid_block_params: BraidBlock parameters (crossing configuration)
    """

    number_of_inputs: int = 2
    is_even_layer: bool = True
    braid_block_params: Optional[BraidBlockParams] = None

    def __post_init__(self) -> None:
        # Default to BraidBlockParams() if not provided
        if self.braid_block_params is None:
            self.braid_block_params = BraidBlockParams()
        
        # Ensure braid_block_params is a single object, not a list
        if isinstance(self.braid_block_params, list):
            raise ValueError(
                "BraidLayerParams.braid_block_params must be a single "
                "BraidBlockParams object, not a list."
            )
