"""
luxia_data_types.py - Automatic dtype-aware tensor creation for LuxIA.

Provides lx_tensor(), lx_complex(), lx_real() that automatically detect
32/64-bit dtype from LxModule context without passing module=self.

USAGE:
    from luxia_data_types import lx_tensor, lx_real
    
    one = lx_tensor([1+0j])        # auto complex128/64 from context
    sqrt2 = lx_real(2.0)           # auto float64/32 from context
"""

import threading
import torch
from typing import Optional, Union, Any


# Thread-local context for automatic module detection
_context = threading.local()


class LuxiaContext:
    """Context manager: automatically detects dtype/device for all tensors."""
    
    def __init__(self, module):
        self.module = module
        self._old_module = getattr(_context, 'module', None)
    
    def __enter__(self):
        _context.module = self.module
        return self
    
    def __exit__(self, *args):
        _context.module = self._old_module


def _get_dtype(dtype_hint: Optional[str] = None) -> torch.dtype:
    """Get dtype from context, hint, or fallback."""
    module = getattr(_context, 'module', None)
    
    if module is not None:
        if dtype_hint == 'complex':
            return module._complex_dtype()
        elif dtype_hint == 'real':
            return module._real_dtype()
    
    # Fallbacks
    return {
        'complex': torch.complex128,
        'real': torch.float64,
    }.get(dtype_hint, torch.complex128)


def lx_tensor(
    data: Union[list, torch.Tensor], 
    dtype_hint: Optional[str] = 'complex',
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """lx_tensor: Auto 32/64-bit complex/real tensors.
    
    Detects dtype from LxModule context automatically.
    
    Args:
        data: Tensor data (list, array, scalar)
        dtype_hint: 'complex' (default), 'real', or None
        device: Optional device override
    
    Returns:
        torch.Tensor matching module dtype
    """
    dtype = _get_dtype(dtype_hint)
    return torch.tensor(data, dtype=dtype, device=device)


def lx_complex(data: Union[list, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    """lx_complex: Auto 32/64-bit complex tensor."""
    return lx_tensor(data, dtype_hint='complex', device=device)


def lx_real(data: Union[list, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    """lx_real: Auto 32/64-bit real tensor."""
    return lx_tensor(data, dtype_hint='real', device=device)


def lx_eye(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """lx_eye: Auto dtype identity matrix."""
    dtype = _get_dtype('complex')
    return torch.eye(n, dtype=dtype, device=device)


def lx_zeros(*size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """lx_zeros: Auto dtype zero tensor."""
    dtype = _get_dtype('complex')
    return torch.zeros(*size, dtype=dtype, device=device)


def lx_ones(*size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """lx_ones: Auto dtype ones tensor."""
    dtype = _get_dtype('complex')
    return torch.ones(*size, dtype=dtype, device=device)


# Context manager for explicit scope (optional)
def context(module):
    """Explicit context: with context(self): lx_tensor() works."""
    return LuxiaContext(module)
