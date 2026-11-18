from dataclasses import dataclass
from typing import Optional, List

from luxia.params import HeaterParams, MmiParams


@dataclass
class ClementsBellMeshParams:
    """Parameters for the Clements-Bell mesh.

    Args:
    number_of_layers: Total number of mesh layers.
    number_of_inputs: Number of input waveguides.
        mmis_params: Optional per-layer MMI parameters for the internal Bell layers.
        heater_params: Optional heater parameters shared across the mesh.
        is_matmul: Forwarded to PhotonicMesh.
    """
    number_of_layers: int
    number_of_inputs: int
    mmis_params: Optional[List] = None
    heater_params: Optional[List[HeaterParams]] = None
    is_matmul: bool = False
    window_size: int = 1
