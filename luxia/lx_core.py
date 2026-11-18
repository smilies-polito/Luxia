from typing import Any, Dict, List
import torch.nn as nn


class LxCore:
    """
    Mixin providing a uniform API over the underlying photonic core.

    Classes using this mixin must implement a `_core_model` property
    that returns the nn.ModuleDict (or similar) holding the photonic modules.

    Example:

        class LxAccelerator(nn.Module, LxCore):
            def __init__(...):
                ...
                self._ph_model = nn.ModuleDict(...)
            @property
            def _core_model(self) -> nn.ModuleDict:
                return self._ph_model

        class LxHybridAccelerator(nn.Module, LxCore):
            def __init__(...):
                ...
                self._ph_core = LxAccelerator(...)
            @property
            def _core_model(self) -> nn.ModuleDict:
                return self._ph_core._ph_model
    """

    @property
    def _core_model(self) -> nn.ModuleDict:
        """
        Return the container holding the photonic core modules.

        This property shall be implemented in subclasses.
        """
        raise NotImplementedError(
            "_core_model property must be implemented in subclasses using LxCore"
        )

    # ---------------------------------------------------------------------
    # Generic accessors over lx modules
    # ---------------------------------------------------------------------
    def get_lx_module(self, name: str) -> nn.Module:
        """
        Return a single underlying lx module by name.
        """
        core = self._core_model
        if name not in core:
            raise KeyError(f"lx module '{name}' not found in _core_model")
        return core[name]

    def get_lx_modules_by_prefix(self, prefix: str) -> List[nn.Module]:
        """
        Return all lx modules whose name starts with the given prefix.
        """
        core = self._core_model
        return [module for module_name, module in core.items()
                if module_name.startswith(prefix)]

    def get_lx_module_names(self) -> List[str]:
        """
        Return the list of lx module names.
        """
        return list(self._core_model.keys())

    # ---------------------------------------------------------------------
    # Metrics aggregation
    # ---------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return metrics for all modules in the photonic core.

        Each entry is indexed by module name. Modules that do not expose
        get_metrics() shall return an empty dictionary entry.
        """
        metrics: Dict[str, Any] = {}

        for module_name, module in self._core_model.items():
            getter = getattr(module, "get_metrics", None)
            metrics[module_name] = getter() if callable(getter) else {}

        return metrics

    # ---------------------------------------------------------------------
    # Parameter setting for underlying lx modules
    # ---------------------------------------------------------------------
    def set_lx_module_params(self, module_name: str, params: Any) -> None:
        """
        Set parameters for an underlying lx module.

        The target module shall expose a set_params() method.
        """
        module = self.get_lx_module(module_name)

        if not hasattr(module, "set_params"):
            raise AttributeError(
                f"lx module '{module_name}' does not have a 'set_params' method"
            )

        module.set_params(params)

    # ---------------------------------------------------------------------
    # Photonic-only state dict helpers
    # ---------------------------------------------------------------------
    def photonic_state_dict(self):
        """
        Return a state_dict containing only the photonic core parameters.
        """
        return self._core_model.state_dict()

    def load_photonic_state_dict(self, state_dict) -> None:
        """
        Load parameters into the photonic core from a photonic state_dict.
        """
        self._core_model.load_state_dict(state_dict)
