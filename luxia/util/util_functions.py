from __future__ import annotations
from typing import Iterable

import torch


# ------------------------------------------------------------------
# Alternating / diagonal helpers
# ------------------------------------------------------------------

def alternating_tensors(tensor0: torch.Tensor, tensor1: torch.Tensor) -> torch.Tensor:
    """
    Create a tensor with the inputs interleaved along the last dimension.

    tensor0, tensor1: same shape [B, N] -> output [B, 2*N]
    """
    if tensor0.shape != tensor1.shape:
        raise ValueError(
            f"alternating_tensors: shapes must match, "
            f"got {tensor0.shape} and {tensor1.shape}"
        )
    if tensor0.ndim < 1:
        raise ValueError("alternating_tensors: inputs must be at least 1D")

    # Stack along a new last axis and flatten the last two dims
    stacked = torch.stack([tensor0, tensor1], dim=-1)  # [..., 2]
    out = stacked.view(*tensor0.shape[:-1], -1)
    return out


def diag_matrix_alternating_tensors(
    tensor0: torch.Tensor,
    tensor1: torch.Tensor,
) -> torch.Tensor:
    """
    Create a diagonal matrix with entries [t0[0], t1[0], t0[1], t1[1], ...].

    tensor0, tensor1: shape [N] (or any same 1D shape).
    """
    if tensor0.shape != tensor1.shape:
        raise ValueError(
            f"diag_matrix_alternating_tensors: shapes must match, "
            f"got {tensor0.shape} and {tensor1.shape}"
        )
    if tensor0.ndim != 1:
        raise ValueError(
            f"diag_matrix_alternating_tensors: expected 1D inputs, "
            f"got ndim={tensor0.ndim}"
        )

    stacked = torch.stack((tensor0, tensor1), dim=1)  # [N, 2]
    vec = stacked.view(-1)                            # [2N]
    return torch.diag(vec)


def diag_matrix_alternating_tensor_with_ones(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a diagonal matrix with entries [x[0], 1, x[1], 1, ...].

    tensor: shape [N].
    """
    if tensor.ndim != 1:
        raise ValueError(
            f"diag_matrix_alternating_tensor_with_ones: expected 1D input, "
            f"got ndim={tensor.ndim}"
        )

    ones = torch.ones_like(tensor)
    stacked = torch.stack((tensor, ones), dim=1)  # [N, 2]
    vec = stacked.view(-1)                        # [2N]
    return torch.diag(vec)


# ------------------------------------------------------------------
# Generic tensor shape utilities
# ------------------------------------------------------------------

def ensure_min_ndim(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """
    Ensure tensor has at least `ndim` dimensions by adding leading singleton dims.

    Examples:
      [F]      , ndim=2 -> [1, F]
      [B, F]   , ndim=3 -> [1, B, F]
      [B,F,T]  , ndim=3 -> [B,F,T] (unchanged)
    """
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")
    if x.ndim >= ndim:
        return x
    pad = ndim - x.ndim
    new_shape = (1,) * pad + tuple(x.shape)
    return x.view(*new_shape)


def expand_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Add a singleton last dimension and expand it to `size`.

    Examples:
      x: [F]       , size=10 -> [F, 10]
      x: [B, F]    , size=10 -> [B, F, 10]
      x: [..., F]  , size=10 -> [..., F, 10]
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return x.unsqueeze(-1).expand(*x.shape, size)


def repeat_last_dim(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    Repeat the last dimension `repeats` times (via repeat, not broadcast).

    Examples:
      x: [F]       , repeats=3 -> [F, 3]
      x: [B, F, 1] , repeats=3 -> [B, F, 3]
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    return x.repeat(*([1] * (x.ndim - 1)), repeats)


def flatten_tail(x: torch.Tensor, n_tail_dims: int) -> torch.Tensor:
    """
    Flatten the last `n_tail_dims` dimensions into a single dimension.

    Examples:
      x: [B,F,T] , n_tail_dims=2 -> [B, F*T]
      x: [F,T]   , n_tail_dims=2 -> [F*T]
      x: [B,C,H,W], n_tail_dims=3 -> [B, C*H*W]
    """
    if n_tail_dims < 1:
        raise ValueError(f"n_tail_dims must be >= 1, got {n_tail_dims}")
    if x.ndim < n_tail_dims:
        raise ValueError(
            f"flatten_tail: tensor has ndim={x.ndim} < n_tail_dims={n_tail_dims}"
        )
    head_shape = x.shape[:-n_tail_dims]
    tail = x.shape[-n_tail_dims:]
    flat_tail = 1
    for s in tail:
        flat_tail *= s
    return x.reshape(*head_shape, flat_tail)


def unflatten_tail(
    x: torch.Tensor,
    tail_shape: Iterable[int],
) -> torch.Tensor:
    """
    Inverse of flatten_tail when you know the original tail shape.

    Examples:
      x: [B, F*T] , tail_shape=(F, T)    -> [B, F, T]
      x: [F*T]    , tail_shape=(F, T)    -> [F, T]
      x: [B, C*H*W], tail_shape=(C,H,W)  -> [B, C, H, W]
    """
    tail_shape = tuple(int(s) for s in tail_shape)
    if len(tail_shape) == 0:
        raise ValueError("tail_shape must be non-empty")
    return x.reshape(*x.shape[:-1], *tail_shape)


def move_last_to(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Move the last dimension to a given position `dim` (like np.moveaxis).

    Examples:
      x: [B,F,T], dim=0 -> [T,B,F]
      x: [B,F,T], dim=1 -> [B,T,F]
    """
    ndim = x.ndim
    if not (-ndim <= dim < ndim):
        raise ValueError(f"dim={dim} out of range for ndim={ndim}")
    if dim < 0:
        dim += ndim
    if dim == ndim - 1:
        return x
    perm = list(range(ndim))
    last = perm.pop(-1)
    perm.insert(dim, last)
    return x.permute(*perm)


def take_last(x: torch.Tensor, n: int = 1) -> torch.Tensor:
    """
    Take the last `n` elements along the last dimension.

    n=1 returns x[..., -1] with the last dim removed.
    n>1 returns x[..., -n:] keeping the last dim.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if x.shape[-1] < n:
        raise ValueError(
            f"take_last: last dim has size {x.shape[-1]} < n={n}"
        )
    if n == 1:
        return x[..., -1]
    return x[..., -n:]

