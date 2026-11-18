from dataclasses import dataclass, field
from typing import Optional, List, Union
from luxia.params.clements_layer_params import ClementsLayerParams
from luxia.params.braid_layer_params import BraidLayerParams


@dataclass
class BraidMeshParams:
    """
    Parameters for a full BraidMesh (braid interferometer mesh).

    The braid mesh consists of alternating ClementsLayer and BraidLayer:
      - Even indices (0, 2, 4, ...): ClementsLayer with heaters/phase shifters
      - Odd indices (1, 3, 5, ...): BraidLayer (passive crossing layer)

    For N inputs, the braid architecture has:
      - (N - 1) Clements layers
      - (N - 2) BraidLayer (crossing layers)
      - Total: (2N - 3) layers

    The mesh always starts and ends with a Clements layer.

    Reference:
        Federico Marchesin et al., "Braided interferometer mesh for robust
        photonic matrix-vector multiplications with non-ideal components",
        Optics Express 33(2), 2227–2246 (2025).
        DOI: 10.1364/OE.547206

    Attributes:
        number_of_inputs: Number of mesh inputs (minimum 2)
        number_of_layers: Total number of layers (must equal 2*number_of_inputs - 3)
        layer_params: List of layer parameters, alternating ClementsLayerParams
                      and BraidLayerParams
        is_matmul: Enable matrix-multiplication mode
        use_dropout: Enable dropout mask generation during training
        dropout_probability: Dropout probability in [0, 1]
    """

    number_of_inputs: int = 2
    number_of_layers: int = 1
    layer_params: List[Union[ClementsLayerParams, BraidLayerParams]] = field(default_factory=list)
    is_matmul: bool = False
    use_dropout: bool = False
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.number_of_inputs < 2:
            raise ValueError(
                f"BraidMeshParams.number_of_inputs must be >= 2 (minimum for braid architecture), "
                f"got {self.number_of_inputs}"
            )

        expected_layers = 2 * self.number_of_inputs - 3
        if self.number_of_layers != expected_layers:
            raise ValueError(
                f"BraidMeshParams.number_of_layers must be 2*number_of_inputs - 3 = {expected_layers} "
                f"for number_of_inputs={self.number_of_inputs}, but got {self.number_of_layers}"
            )

        if len(self.layer_params) != self.number_of_layers:
            raise ValueError(
                f"BraidMeshParams.layer_params length ({len(self.layer_params)}) "
                f"must match number_of_layers ({self.number_of_layers})"
            )

        for idx, layer_param in enumerate(self.layer_params):
            is_clements_layer = (idx % 2 == 0)

            if is_clements_layer:
                if not isinstance(layer_param, ClementsLayerParams):
                    raise ValueError(
                        f"BraidMeshParams.layer_params[{idx}] must be ClementsLayerParams "
                        f"(even index), but got {type(layer_param).__name__}"
                    )
            else:
                if not isinstance(layer_param, BraidLayerParams):
                    raise ValueError(
                        f"BraidMeshParams.layer_params[{idx}] must be BraidLayerParams "
                        f"(odd index), but got {type(layer_param).__name__}"
                    )

        clements_count = sum(1 for lp in self.layer_params if isinstance(lp, ClementsLayerParams))
        braid_count = sum(1 for lp in self.layer_params if isinstance(lp, BraidLayerParams))

        expected_clements = self.number_of_inputs - 1
        expected_braid = self.number_of_inputs - 2

        if clements_count != expected_clements:
            raise ValueError(
                f"BraidMeshParams: expected {expected_clements} ClementsLayerParams "
                f"(number_of_inputs - 1), but got {clements_count}"
            )

        if braid_count != expected_braid:
            raise ValueError(
                f"BraidMeshParams: expected {expected_braid} BraidLayerParams "
                f"(number_of_inputs - 2), but got {braid_count}"
            )
