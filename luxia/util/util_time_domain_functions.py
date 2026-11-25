# luxia/util/util_time_domain_simulation.py
from __future__ import annotations
from typing import Tuple

import torch

from luxia.util.util_functions import (
    expand_last_dim,
    take_last,
)


def samples_per_period(clk_freq: float, dt: float) -> int:
    """
    Compute integer samples_per_period from clock frequency and time step.

    Raises if result is < 1.
    """
    if clk_freq <= 0.0:
        raise ValueError(f"clk_freq must be > 0, got {clk_freq}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    spp_f = 1.0 / (clk_freq * dt)
    spp = int(round(spp_f))
    if spp < 1:
        raise ValueError(
            f"Invalid samples_per_period={spp} "
            f"(computed from clk_freq={clk_freq}, dt={dt})"
        )
    return spp


def period_end_indices(
    total_T: int,
    samples_per_period: int,
    *,
    device: torch.device | None = None,
    require_full_period: bool = False,
) -> torch.Tensor:
    """
    Compute indices of period-end samples for a 1D time axis of length total_T.

    Returns a 1D LongTensor of indices < total_T.

    If require_full_period is True and no valid indices exist, raises ValueError.
    """
    if samples_per_period < 1:
        raise ValueError(f"samples_per_period must be >= 1, got {samples_per_period}")
    if total_T < 1:
        raise ValueError(f"total_T must be >= 1, got {total_T}")

    max_full_periods = total_T // samples_per_period
    if max_full_periods < 1:
        if require_full_period:
            raise ValueError(
                f"Signal length {total_T} too short for one period "
                f"({samples_per_period} samples)"
            )
        return torch.empty(0, dtype=torch.long, device=device)

    idx = (
        torch.arange(
            1,
            max_full_periods + 1,
            device=device,
            dtype=torch.long,
        )
        * samples_per_period
        - 1
    )
    idx = idx[idx < total_T]
    if idx.numel() == 0 and require_full_period:
        raise ValueError(
            f"No valid period-end indices for total_T={total_T}, "
            f"samples_per_period={samples_per_period}"
        )
    return idx


def collapse_last_dim(
    x: torch.Tensor,
    *,
    mode: str,
    dt: float | None = None,
    clk_freq: float | None = None,
) -> torch.Tensor:
    """
    Collapse the last (time) axis of a tensor x[..., T] according to mode.

    Modes:
      - "last": return x[..., -1]
      - "per_period": sample at clock-aligned period ends using dt and clk_freq
      - "per_period_center": sample near the middle of each period
    """
    if x.ndim < 1:
        raise ValueError(f"collapse_last_dim: expected at least 1D, got {x.shape}")

    if mode == "last":
        return take_last(x, 1)

    if mode in ("per_period", "per_period_center"):
        if dt is None or clk_freq is None:
            raise ValueError(f"{mode} collapse requires dt and clk_freq")

        spp = samples_per_period(clk_freq, dt)
        T = x.shape[-1]

        if mode == "per_period":
            idx = period_end_indices(
                total_T=T,
                samples_per_period=spp,
                device=x.device,
                require_full_period=False,
            )
        else:  # "per_period_center"
            max_full_periods = T // spp
            if max_full_periods < 1:
                # fallback: use last sample
                return take_last(x, 1)

            # start indices of each full period: 0, spp, 2*spp, ...
            starts = torch.arange(
                max_full_periods,
                device=x.device,
                dtype=torch.long,
            ) * spp
            center_offset = spp // 2  # floor(spp/2)
            idx = starts + center_offset
            idx = idx[idx < T]
            if idx.numel() == 0:
                return take_last(x, 1)

        return x.index_select(-1, idx)

    raise ValueError(f"Unknown collapse mode {mode!r}")


def expand_ss_to_time(
    x_ss: torch.Tensor,
    *,
    clk_freq: float,
    dt: float,
) -> torch.Tensor:
    """
    Expand steady-state tensor by appending a time axis of length samples_per_period.

    Assumes last dimension of x_ss is the feature/channel axis.

    Examples:
      [F]       -> [F, T]
      [B, F]    -> [B, F, T]
      [..., F]  -> [..., F, T]
    """
    spp = samples_per_period(clk_freq, dt)
    return expand_last_dim(x_ss, spp)

