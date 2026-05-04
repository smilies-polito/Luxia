from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BeamsplitterParams:
    """Dataclass containing the parameters of the beamsplitters.

    Attributes:
        num_splitters: Number of splitters
        i_loss: (For all types) Insertion loss
        imbalance: (For Dielectric BS & MMI) Imbalance
        crosstalk: (For Cross BS & Half-cross BS) Crosstalk
    """
    num_splitters: int = 1
    i_loss: Optional[torch.Tensor] = None
    imbalance: Optional[torch.Tensor] = None
    crosstalk: Optional[torch.Tensor] = None


@dataclass
class BsCrossParams(BeamsplitterParams):
    """Beam splitter cross configuration"""
    pass  # Inherits everything from BeamSplitterParams


@dataclass
class BsDielectricParams(BeamsplitterParams):
    """Dielectric beam splitter"""
    pass  # Inherits everything from BeamSplitterParams


@dataclass
class BsHalfCrossParams(BeamsplitterParams):
    """Half-cross beam splitter"""
    pass  # Inherits everything from BeamSplitterParams


@dataclass
class BsMmiParams(BeamsplitterParams):
    """MMI beam splitter"""
    pass  # Inherits everything from BeamSplitterParams
