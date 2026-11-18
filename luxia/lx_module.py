import torch
from typing import Any, Dict


class LxModule(torch.nn.Module):
    """
    Base class for Luxia modules.

    All derived modules shall expose a uniform metrics API through
    get_metrics(). Modules with no available metrics shall return
    an empty dictionary.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the base lx module.

        Parameters
        ----------
        debug : bool
            Debug mode enable flag.
        """
        super(LxModule, self).__init__()
        self._debug = debug

    # ---------------------------------------------------------------------
    # Core forward interface
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the module forward pass.

        This method shall be implemented by all derived classes.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented."
        )

    # ---------------------------------------------------------------------
    # Debug control
    # ---------------------------------------------------------------------
    def set_debug_mode(self, debug: bool) -> None:
        """
        Set the module debug mode.

        Parameters
        ----------
        debug : bool
            New debug mode state.
        """
        self._debug = debug

    def get_debug_mode(self) -> bool:
        """
        Return the current debug mode state.
        """
        return self._debug

    # ---------------------------------------------------------------------
    # Metrics interface
    # ---------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return module metrics in dictionary form.

        Derived classes should override this method when metrics are
        available. The default implementation returns an empty dictionary
        so that metric collection can be performed uniformly at a higher
        abstraction level.
        """
        return {}
