from dataclasses import dataclass, fields
from typing import List, Any

from luxia.params import FldzhyanLayerParams


def _indent_repr(obj: Any, level: int = 1, indent: str = "    ") -> str:
    """Helper to pretty-print nested dataclasses with indentation."""
    from dataclasses import is_dataclass, fields as dc_fields

    if not is_dataclass(obj):
        return repr(obj)

    lines = [f"{indent * level}{obj.__class__.__name__}("]
    for f in dc_fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            inner = _indent_repr(value, level + 1, indent)
            lines.append(f"{indent * (level + 1)}{f.name}={inner},")
        elif isinstance(value, list):
            if value and is_dataclass(value[0]):
                items = [_indent_repr(v, level + 2, indent) for v in value]
                list_str = "[\n" + ",\n".join(items) + f"\n{indent * (level + 1)}]"
                lines.append(f"{indent * (level + 1)}{f.name}={list_str},")
            else:
                lines.append(f"{indent * (level + 1)}{f.name}={repr(value)},")
        else:
            lines.append(f"{indent * (level + 1)}{f.name}={repr(value)},")
    lines.append(f"{indent * level})")
    return "\n".join(lines)


@dataclass
class FldzhyanMeshParams:
    """
    Parameters for a Fldzhyan rectangular photonic mesh.
    The mesh is defined as an ordered sequence of FldzhyanLayerParams objects.
    The user must provide exactly number_of_layers layer-parameter objects, one per
    physical layer in the rectangular mesh topology.

    The mesh follows the Fldzhyan interferometer topology:

        Suren A. Fldzhyan, Mikhail Yu. Saygin, and Stanislav S. Straupe,
        "Optimal design of error-tolerant reprogrammable multiport interferometers",
        Optics Letters 45(9), 2632–2635 (2020).
        DOI: 10.1364/OL.385433
    """

    number_of_inputs: int
    number_of_layers: int
    layer_params: List[FldzhyanLayerParams]
    is_matmul: bool = False
    use_dropout: bool = False
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.number_of_inputs <= 0:
            raise ValueError("number_of_inputs must be > 0")

        if self.number_of_layers <= 0:
            raise ValueError("number_of_layers must be > 0")

        if len(self.layer_params) != self.number_of_layers:
            raise ValueError(
                "layer_params must have the same length as number_of_layers"
            )

        for idx, lp in enumerate(self.layer_params):
            if lp.number_of_inputs != self.number_of_inputs:
                raise ValueError(
                    f"layer_params[{idx}].number_of_inputs must equal number_of_inputs "
                    f"({self.number_of_inputs}), got {lp.number_of_inputs}"
                )

            expected_even_layer = (idx % 2) == 0
            if lp.is_even_layer != expected_even_layer:
                raise ValueError(
                    f"layer_params[{idx}].is_even_layer must be "
                    f"{expected_even_layer} for layer index {idx}"
                )

    def __repr__(self) -> str:
        return _indent_repr(self, level=0, indent="    ")
