import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from collections import OrderedDict

from .lx_core import LxCore


class LxAccelerator(nn.Module, LxCore):
    """
    Pure photonic accelerator core - FLEXIBLE OUTPUT SELECTION.

    Modes:
    - 'central': central slice of channels (no pooling)
    - 'full':    all n_inputs channels (no pooling)
    - 'custom':  specific channels (no pooling)
    - 'max_pool': select channels (like above), then split them into
                  n_classes contiguous segments and return the max
                  per segment -> [batch_size, n_classes].
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        lx_module_list: Union[Dict[str, nn.Module], List[nn.Module]],
        output_mode: str = 'central',
        custom_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if lx_module_list is None:
            raise ValueError("lx_module_list cannot be None")

        self.n_inputs = n_inputs
        self._original_n_classes = n_classes  # logical number of classes
        self.n_classes = n_classes
        self.output_mode = output_mode
        self.custom_channels = custom_channels

        # Validate custom channels if provided
        if custom_channels is not None:
            if any(ch < 0 or ch >= n_inputs for ch in custom_channels):
                raise ValueError(f"Channels must be in [0, {n_inputs-1}]")

        # normalize to ModuleDict
        if isinstance(lx_module_list, dict):
            self._ph_model = nn.ModuleDict(lx_module_list)
        elif isinstance(lx_module_list, list):
            ordered = OrderedDict()
            for i, m in enumerate(lx_module_list):
                ordered[f"lx_module_{i}"] = m
            self._ph_model = nn.ModuleDict(ordered)
        else:
            raise ValueError(
                "lx_module_list must be dict[str, Module] or list[Module]")

    @property
    def _core_model(self) -> nn.ModuleDict:
        return self._ph_model

    def _evenly_pooled_max(
        self, y: torch.Tensor, n_kernels: int
    ) -> torch.Tensor:
        """
        Split last dimension of y into n_kernels contiguous segments
        and return the max of each segment.

        y: [B, C]
        returns: [B, n_kernels]
        """
        B, C = y.shape
        if n_kernels <= 0:
            raise ValueError("n_kernels must be positive")
        if C == 0:
            raise ValueError("No channels to pool over")

        # Clamp to available channels
        n_kernels = min(n_kernels, C)

        # Segment boundaries in [0, C]
        indices = torch.linspace(0, C, steps=n_kernels + 1, dtype=torch.long)
        indices[-1] = C  # ensure exact

        pooled = []
        for i in range(n_kernels):
            start = indices[i].item()
            end = indices[i + 1].item()
            if end <= start:
                seg = y[:, start:start + 1]
            else:
                seg = y[:, start:end]  # [B, seg_len]
            # seg_max, _ = seg.max(dim=1, keepdim=True)  # [B, 1]
            seg_max = seg.sum(dim=1, keepdim=True)  # [B, 1]
            pooled.append(seg_max)

        return torch.cat(pooled, dim=1)  # [B, n_kernels]

    def forward(self, x_reals: torch.Tensor) -> torch.Tensor:
        """
        x_reals: [B, n_inputs]

        Returns:
        - if output_mode in {'central','full','custom'}: raw selected channels
        - if output_mode == 'max_pool': [B, n_classes], each entry is the
          max of one of n_classes evenly sized channel segments.
        """
        # if x_reals.shape[-1] != self.n_inputs:
        #     raise ValueError(
        #         f"Expected input dimension {self.n_inputs}, "
        #         f"got {x_reals.shape[-1]}"
        #     )

        # Photonic forward pass
        y_reals = x_reals
        for module in self._ph_model.values():
            y_reals = module(y_reals)

        # STEP 1: select base channels
        if self.output_mode == 'central':
            low, upp = self.get_limits(self.n_inputs, self._original_n_classes)
            y_sel = y_reals[:, low:upp]               # [B, C_sel]
            return y_sel

        elif self.output_mode == 'full':
            return y_reals                             # [B, n_inputs]

        elif self.output_mode == 'custom':
            if self.custom_channels is None:
                raise ValueError("custom_channels must be set for 'custom'")
            return y_reals[:, self.custom_channels]    # [B, C_sel]

        elif self.output_mode == 'max_pool':
            # choose which channels to pool over; here we pool over ALL
            # outputs (you can change to central/custom if you prefer)
            y_sel = y_reals                            # [B, C_sel = n_inputs]
            y_out = self._evenly_pooled_max(
                y_sel, self._original_n_classes
            )                                          # [B, n_classes]
            self.n_classes = y_out.shape[-1]
            return y_out

        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

    # Mode setters
    def set_central_slicing(self):
        self.output_mode = 'central'
        self.n_classes = self._original_n_classes

    def set_full_output(self):
        self.output_mode = 'full'
        self.n_classes = self.n_inputs

    def set_custom_channels(self, channels: List[int]):
        if any(ch < 0 or ch >= self.n_inputs for ch in channels):
            raise ValueError(f"Channels must be in [0, {self.n_inputs-1}]")
        self.custom_channels = channels
        self.output_mode = 'custom'
        self.n_classes = len(channels)

    def set_max_pool(self):
        """Pool over all output channels into n_classes bins by max."""
        self.output_mode = 'max_pool'
        self.n_classes = self._original_n_classes

    def get_current_config(self) -> Dict:
        return {
            'n_inputs': self.n_inputs,
            'n_classes': self.n_classes,
            'output_mode': self.output_mode,
            'custom_channels': getattr(self, 'custom_channels', None),
            'central_limits': self.get_limits(self.n_inputs, self._original_n_classes)
        }

    @staticmethod
    def get_limits(n_inputs: int, n_classes: int) -> Tuple[int, int]:
        lower_limit = n_inputs // 2 - (n_classes // 2)
        upper_limit = n_inputs // 2 + n_classes // 2
        if n_classes % 2 != 0:
            upper_limit += 1
        return lower_limit, upper_limit
