from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class MmiParams:
    """Parameters for a 2x2 Multimode Interferometer (MMI) coupler."""
    number_of_splitters: int = 1
    insertion_loss: Optional[torch.Tensor] = None
    imbalance: Optional[torch.Tensor] = None


@dataclass
class CrossParams:
    """Parameters for a 2x2 waveguide crossing."""
    number_of_splitters: int = 1
    insertion_loss: Optional[torch.Tensor] = None
    crosstalk: Optional[torch.Tensor] = None


# Python-only types — deferred rename, still use BeamsplitterParams base
@dataclass
class BeamsplitterParams:
    num_splitters: int = 1
    i_loss: Optional[torch.Tensor] = None
    imbalance: Optional[torch.Tensor] = None
    crosstalk: Optional[torch.Tensor] = None


@dataclass
class BsDielectricParams(BeamsplitterParams):
    """Dielectric beam splitter"""
    pass


@dataclass
class BsHalfCrossParams(BeamsplitterParams):
    """Half-cross beam splitter"""
    pass
