import torch
import torch.nn as nn
from typing import Dict, List, Union, Any

from .lx_accelerator import LxAccelerator
from .lx_core import LxCore


class LxHybridAccelerator(nn.Module, LxCore):
    """
    Hybrid electronic-photonic accelerator.

    Architecture
    ------------
    - Optional electronic encoder : maps n_features -> n_inputs
    - Photonic core              : LxAccelerator
    - Electronic classifier      : maps photonic outputs -> n_classes

    Notes
    -----
    This accelerator always uses the full photonic output.
    Output slicing and pooling modes are ignored by design.
    """

    def __init__(
        self,
        n_features: int,
        n_inputs: int,
        n_classes: int,
        lx_module_list: Union[Dict[str, nn.Module], List[nn.Module]],
    ) -> None:
        super().__init__()

        if lx_module_list is None:
            raise ValueError("lx_module_list cannot be None")

        self.n_features = n_features
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        # -----------------------------------------------------------------
        # Electronic encoder
        # -----------------------------------------------------------------
        self._has_encoder = n_features > n_inputs

        if self._has_encoder:
            self._electronic_encoder = nn.Linear(n_features, n_inputs)
        else:
            self._electronic_encoder = nn.Identity()

        # -----------------------------------------------------------------
        # Photonic accelerator core
        # -----------------------------------------------------------------
        self._ph_core = LxAccelerator(
            n_inputs=n_inputs,
            n_classes=n_inputs,
            lx_module_list=lx_module_list,
        )
        self._ph_core.set_full_output()

        # -----------------------------------------------------------------
        # Electronic classifier
        # -----------------------------------------------------------------
        self._classifier = nn.Linear(n_inputs, n_classes)

    # ---------------------------------------------------------------------
    # LxCore hook
    # ---------------------------------------------------------------------
    @property
    def _core_model(self) -> nn.ModuleDict:
        """
        Return the underlying photonic core module container.
        """
        return self._ph_core._core_model

    # ---------------------------------------------------------------------
    # Output mode control
    # ---------------------------------------------------------------------
    def set_central_slicing(self) -> None:
        """
        Ignore central slicing request.

        Hybrid accelerator operation is defined on the full photonic
        output. This method enforces full-output mode.
        """
        self._ph_core.set_full_output()

    def set_full_output(self) -> None:
        """
        Enforce full photonic output mode.
        """
        self._ph_core.set_full_output()

    def set_custom_channels(self, channels: List[int]) -> None:
        """
        Ignore custom channel selection request.

        Hybrid accelerator operation is defined on the full photonic
        output. The provided channel list is intentionally ignored.
        """
        _ = channels
        self._ph_core.set_full_output()

    def set_max_pool(self) -> None:
        """
        Ignore max-pool request.

        Hybrid accelerator operation is defined on the full photonic
        output. This method enforces full-output mode.
        """
        self._ph_core.set_full_output()

    # ---------------------------------------------------------------------
    # Configuration helpers
    # ---------------------------------------------------------------------
    def get_current_config(self) -> Dict[str, Any]:
        """
        Return the current hybrid accelerator configuration.
        """
        return {
            "n_features": self.n_features,
            "n_inputs": self.n_inputs,
            "n_classes": self.n_classes,
            "has_encoder": self._has_encoder,
            "photonic_output_mode": "full",
            "ignored_modes": ["central", "custom", "max_pool"],
            "photonic_core": self._ph_core.get_current_config(),
        }

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return hybrid accelerator metrics.
        """
        return {
            "photonic_core": self._ph_core.get_metrics(),
            "electronic_encoder": {
                "type": self._electronic_encoder.__class__.__name__,
                "enabled": self._has_encoder,
            },
            "classifier": {
                "type": self._classifier.__class__.__name__,
                "in_features": self._classifier.in_features,
                "out_features": self._classifier.out_features,
            },
        }

    # ---------------------------------------------------------------------
    # Forward path
    # ---------------------------------------------------------------------
    def forward(self, x_reals: torch.Tensor) -> torch.Tensor:
        """
        Execute the hybrid forward path.
        """
        x_reals = self._electronic_encoder(x_reals)
        x_reals = torch.clamp(x_reals, 0.0, 1.0)
        y_reals = self._ph_core(x_reals)
        return self._classifier(y_reals)
