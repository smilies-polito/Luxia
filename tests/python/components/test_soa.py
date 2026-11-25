import math
import gc
import psutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import pytest

from luxia.photonic_components.soa import Soa, set_ase_seed
from luxia.photonic_components.op_butterworth_lpf import OpButterworthLPF
from luxia.params import ButterworthLPFParams, SoaParams


PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

DT_DEFAULT = 0.5e-12
# DT_DEFAULT = 50e-12
CLK_FREQ_DEFAULT = 10e9

# Shared LPF / simulation defaults
LPF_CUTOFF_DEFAULT = 15e9
LPF_ORDER_DEFAULT = 2
N_PERIODS_DEFAULT = 2
LPF_CUTOFF_WIDE = 20e9  # for the “wide” LPF tests


# ================================================================
# Global plotting style
# ================================================================

PLOT_STYLES = {
    "time_label": "Time t [s]",
    "power_label": "Power [W]",
    "field_real_label": "Real [sqrt(W)]",
    "field_imag_label": "Imag [sqrt(W)]",
    "amp_label": "Amplitude |E(t)| [sqrt(W)]",
    "phase_label": "Phase [rad]",
    "lines": {
        "Pin": dict(color="C0", linestyle="--", linewidth=1.0, alpha=1.0),
        "Pout_SOA": dict(color="C1", linestyle="-", linewidth=1.0, alpha=1.0),
        "Pout_LPF": dict(color="C2", linestyle="-", linewidth=1.0, alpha=0.8),
        "Pout_ASE": dict(color="C3", linestyle="-", linewidth=0.9, alpha=0.6),
    },
    "markers": {
        "SOA_collapsed": dict(marker="o", facecolors="none", edgecolors="k"),
        "LPF_collapsed": dict(marker="s", facecolors="none", edgecolors="red"),
        "SOA_SS": dict(marker="^", facecolors="none", edgecolors="blue"),
        "LPF_period_ends": dict(marker="s", facecolors="none", edgecolors="blue"),
        "LPF_SS": dict(marker="o", facecolors="none", edgecolors="red"),
        "SOA_SS_samples": dict(marker="x", color="k"),
    },
}


# ================================================================
# 1) Helper constructors and utilities
# ================================================================

def default_soa_params(
    *,
    use_ase: bool = False,
    use_time_domain: bool = False,
    dt: float = DT_DEFAULT,
    clk_freq: float = CLK_FREQ_DEFAULT,
    is_input_ss: bool = True,
    collapse_to_ss: bool = False,
    batch_as_time: bool = False,
) -> SoaParams:
    return SoaParams(
        alpha=5.0,
        g0=6075.0,
        L=500e-6,
        P_sat=0.0211,
        P_ref=1e-3,
        norm_input=False,
        norm_output=False,
        newton_iters=15,
        newton_tol=1e-5,
        use_ase=use_ase,
        n_sp=2.0,
        B0=100e9,
        tau_c=300e-12,
        lam0=1550e-9,
        use_time_domain=use_time_domain,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=is_input_ss,
        collapse_to_ss=collapse_to_ss,
        batch_as_time=batch_as_time,
    )


def make_soa(
    *,
    use_ase: bool,
    use_time_domain: bool,
    dt: float = DT_DEFAULT,
    clk_freq: float = CLK_FREQ_DEFAULT,
    is_input_ss: bool = True,
    collapse_to_ss: bool = False,
    batch_as_time: bool = False,
) -> Soa:
    params = default_soa_params(
        use_ase=use_ase,
        use_time_domain=use_time_domain,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=is_input_ss,
        collapse_to_ss=collapse_to_ss,
        batch_as_time=batch_as_time,
    )
    soa = Soa(soa_params=params)
    soa.set_ase_seed(0)
    return soa


def make_lpf_params(
    *,
    dt: float = DT_DEFAULT,
    cutoff: float = LPF_CUTOFF_DEFAULT,
    order: int = LPF_ORDER_DEFAULT,
    use_time_domain: bool,
    collapse_to_ss: bool,
    clk_freq: float = CLK_FREQ_DEFAULT,
    is_input_ss: bool,
    use_ste: bool = False,
    batch_as_time: bool = False,
) -> ButterworthLPFParams:
    return ButterworthLPFParams(
        dt=dt,
        cutoff_freq=cutoff,
        order=order,
        use_time_domain=use_time_domain,
        collapse_to_ss=collapse_to_ss,
        clk_freq=clk_freq,
        use_ste=use_ste,
        is_input_ss=is_input_ss,
        batch_as_time=batch_as_time,
    )


def make_lpf_td_and_ss(
    *,
    dt: float,
    clk_freq: float,
    cutoff_td: float,
    order: int = LPF_ORDER_DEFAULT,
    is_input_ss_td: bool,
) -> tuple[OpButterworthLPF, OpButterworthLPF]:
    td_params = make_lpf_params(
        dt=dt,
        cutoff=cutoff_td,
        order=order,
        use_time_domain=True,
        collapse_to_ss=False,
        clk_freq=clk_freq,
        is_input_ss=is_input_ss_td,
    )
    ss_params = make_lpf_params(
        dt=dt,
        cutoff=cutoff_td,
        order=order,
        use_time_domain=True,
        collapse_to_ss=True,
        clk_freq=clk_freq,
        is_input_ss=is_input_ss_td,
    )
    return OpButterworthLPF(td_params), OpButterworthLPF(ss_params)


def powers_to_field(pin_powers: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(pin_powers).to(torch.complex64)


def field_to_power(input_field: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.abs(input_field))


def time_to_indices(sample_times: np.ndarray, dt: float, length: int) -> np.ndarray:
    idx = np.round(sample_times / dt).astype(int)
    return np.clip(idx, 0, length - 1)


def run_soa(pin_powers: torch.Tensor, *, use_ase: bool = False) -> Tuple[torch.Tensor, float]:
    soa = make_soa(use_ase=use_ase, use_time_domain=False)
    x = powers_to_field(pin_powers)
    t0 = time.perf_counter()
    y = soa(x)
    t1 = time.perf_counter()
    pout = field_to_power(y)
    return pout, (t1 - t0)


def run_soa_td_from_ss(
    P_in: torch.Tensor,
    *,
    use_ase: bool,
    collapse_to_ss: bool,
    batch_as_time: bool = False,
    dt: float = DT_DEFAULT,
    clk_freq: float = CLK_FREQ_DEFAULT,
) -> tuple[Soa, torch.Tensor]:
    Ein_ss = powers_to_field(P_in).to(torch.complex64)
    soa_td = make_soa(
        use_ase=use_ase,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=collapse_to_ss,
        batch_as_time=batch_as_time,
    )
    y = soa_td(Ein_ss)
    return soa_td, y


def period_samples_per_period(f_mod: float, dt: float) -> int:
    T_mod = 1.0 / f_mod
    return int(round(T_mod / dt))


def period_end_indices(
    n_periods: int,
    samples_per_period: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.arange(1, n_periods + 1, device=device) * samples_per_period - 1


def make_period_end_indices_from_signal(
    signal: torch.Tensor,
    *,
    dt: float,
    clk_freq: float,
) -> torch.Tensor:
    samples_per_period = period_samples_per_period(clk_freq, dt)
    total_T = signal.shape[-1]
    n_periods = total_T // samples_per_period
    idx = period_end_indices(n_periods, samples_per_period, device=signal.device)
    idx = idx[idx < total_T]
    return idx


def make_time_grid(
    *,
    num_periods: int = N_PERIODS_DEFAULT,
    dt: float = DT_DEFAULT,
    clk_freq: float = CLK_FREQ_DEFAULT,
) -> tuple[float, float, torch.Tensor]:
    T_mod = 1.0 / clk_freq
    t0 = 0.0
    t_end = num_periods * T_mod
    n_steps = int(round((t_end - t0) / dt))
    t_grid = torch.linspace(t0, t_end, n_steps + 1)
    return t0, t_end, t_grid


def spectrum_positive_half(
    signal: torch.Tensor,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    s_np = signal.cpu().numpy()
    N = len(s_np)
    freqs = np.fft.fftfreq(N, d=dt)
    S = np.fft.fft(s_np)
    mask = freqs >= 0
    return freqs[mask], np.abs(S[mask])


def get_real_imag(E: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    return E.real.cpu().numpy(), E.imag.cpu().numpy()


def get_amp_phase(E: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    return torch.abs(E).cpu().numpy(), torch.angle(E).cpu().numpy()


# ================================================================
# 2) PinFn implementations
# ================================================================

class PinFn:
    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, tau: torch.Tensor) -> torch.Tensor:
        out = self.forward(tau)
        if not isinstance(out, torch.Tensor):
            raise TypeError("PinFn(tau) must return a torch.Tensor")
        if out.shape != tau.shape:
            raise ValueError(
                f"PinFn(tau) must return same shape as tau, "
                f"got {out.shape}, expected {tau.shape}"
            )
        if not torch.is_floating_point(out):
            raise TypeError("PinFn output must be a floating-point tensor")
        return out


class ConstantPin(PinFn):
    def __init__(self, P0: float):
        self.P0 = float(P0)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.full_like(tau, self.P0)


class StepPin(PinFn):
    def __init__(self, P_low: float, P_high: float, t_step: float):
        self.P_low = float(P_low)
        self.P_high = float(P_high)
        self.t_step = float(t_step)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.where(
            tau < self.t_step,
            torch.full_like(tau, self.P_low),
            torch.full_like(tau, self.P_high),
        )


class SquareWavePin(PinFn):
    def __init__(self, P_low: float, P_high: float, f: float, duty: float):
        self.P_low = float(P_low)
        self.P_high = float(P_high)
        self.f = float(f)
        self.duty = float(duty)
        self.T = 1.0 / self.f

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(tau, self.T)
        return torch.where(
            phase < self.duty * self.T,
            torch.full_like(tau, self.P_high),
            torch.full_like(tau, self.P_low),
        )


class SinePin(PinFn):
    def __init__(self, P_bias: float, dP: float, f: float):
        self.P_bias = float(P_bias)
        self.dP = float(dP)
        self.f = float(f)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.P_bias + self.dP * torch.sin(2 * math.pi * self.f * tau),
            min=0.0,
        )


def make_pin_functions(f_mod: float, t_end) -> Dict[str, PinFn]:
    cnst1 = 1e-5
    cnst2 = 1e-6
    cnst3 = 1e-7
    step1 = 1e-3
    step2 = 1e-4
    step3 = 1e-6
    t_step = t_end / 3
    return {
        f"const_{cnst1}": ConstantPin(cnst1),
        f"const_{cnst2}": ConstantPin(cnst2),
        f"const_{cnst3}": ConstantPin(cnst3),
        f"step_{step1}": StepPin(0.0, step1, t_step=t_step),
        f"step_{step2}": StepPin(0.0, step2, t_step=t_step),
        f"step_{step3}": StepPin(0.0, step3, t_step=t_step),
        "square": SquareWavePin(1e-6, 1e-4, f=f_mod, duty=0.5),
        "sine_medium": SinePin(P_bias=5e-4, dP=4.9e-4, f=f_mod),
    }


# ================================================================
# 3) Plotting helpers
# ================================================================

def plot_lpf_frequency_response(
    freqs: torch.Tensor,
    H: torch.Tensor,
    cutoff: float,
    filename: str,
) -> None:
    H_mag = 20 * torch.log10(H.abs() + 1e-20)
    freqs_np = freqs.abs().cpu().numpy()
    H_mag_np = H_mag.cpu().numpy()
    mask = freqs_np >= 0

    plt.figure(figsize=(6, 4))
    plt.semilogx(freqs_np[mask], H_mag_np[mask])
    plt.axvline(cutoff, color="r", linestyle="--", label="cutoff")
    plt.xlabel("Frequency f [Hz]")
    plt.ylabel("|H(f)| [dB]")
    plt.title("Butterworth LPF frequency response")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.close()


def make_channel_axes(
    F: int,
    *,
    ncols: int = 1,
    figsize_x: float = 10.0,
    figsize_y_per_row: float = 3.0,
    sharex: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    if ncols == 1:
        fig, axes = plt.subplots(F, 1, figsize=(figsize_x, figsize_y_per_row * F), sharex=sharex)
        if F == 1:
            axes = np.array([axes])
    else:
        fig, axes = plt.subplots(
            nrows=F,
            ncols=ncols,
            figsize=(figsize_x * ncols, figsize_y_per_row * F),
            sharex=sharex,
            sharey="row",
        )
    return fig, axes


def plot_traces(
    ax: plt.Axes,
    t: np.ndarray,
    traces: Dict[str, np.ndarray],
    style_keys: Dict[str, str],
) -> None:
    for name, y in traces.items():
        style_key = style_keys.get(name)
        style = PLOT_STYLES["lines"].get(style_key, {})
        ax.plot(t, y, label=name, **style)


def plot_markers(
    ax: plt.Axes,
    x: np.ndarray,
    series: Dict[str, np.ndarray | float],
    style_keys: Dict[str, str],
    *,
    label_first_only: bool = True,
    channel_index: int = 0,
) -> None:
    for name, y_val in series.items():
        style_key = style_keys.get(name)
        style = PLOT_STYLES["markers"].get(style_key, {})
        if np.isscalar(y_val):
            x_vec = np.array([float(x[-1])], dtype=float)
            y_vec = np.array([float(y_val)], dtype=float)
        else:
            y_arr = np.asarray(y_val, dtype=float)
            if y_arr.shape[0] != x.shape[0]:
                raise ValueError(
                    f"plot_markers: y for '{name}' has shape {y_arr.shape}, "
                    f"expected ({x.shape[0]},)"
                )
            x_vec = x
            y_vec = y_arr

        label = name if (not label_first_only or channel_index == 0) else None
        ax.scatter(x_vec, y_vec, label=label, **style)


def plot_real_imag_fields(
    t_np: np.ndarray,
    Einput: torch.Tensor,
    E_clean: torch.Tensor,
    E_noisy: torch.Tensor,
    E_noise: torch.Tensor,
    E_out: torch.Tensor,
    name: str,
) -> None:
    E_clean_re, E_clean_im = get_real_imag(E_clean)
    E_noisy_re, E_noisy_im = get_real_imag(E_noisy)
    E_noise_re, E_noise_im = get_real_imag(E_noise)
    E_LPF_re, E_LPF_im = get_real_imag(E_out)
    Einput_re, Einput_im = get_real_imag(Einput)

    plt.figure(figsize=(20, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_np, Einput_re, label="Re{in_field}")
    plt.plot(t_np, E_clean_re, label="Re{E_clean(t)}")
    plt.plot(t_np, E_noisy_re, label="Re{E_clean+noise(t)}", alpha=0.7)
    plt.plot(t_np, E_noise_re, label="Re{noise(t)}", alpha=0.7)
    plt.plot(t_np, E_LPF_re, label="Re{LPF(t)}", alpha=0.7)
    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["field_real_label"])
    plt.title(f"Real/Imag: clean field, noisy field, noise, 10 GHz {name}")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_np, Einput_im, label="Img{in_field}")
    plt.plot(t_np, E_clean_im, label="Im{E_clean(t)}")
    plt.plot(t_np, E_noisy_im, label="Im{E_clean+noise(t)}", alpha=0.7)
    plt.plot(t_np, E_noise_im, label="Im{noise(t)}", alpha=0.7)
    plt.plot(t_np, E_LPF_im, label="Im{LPF(t)}", alpha=0.7)
    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["field_imag_label"])
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"soa_10GHz_{name}_E_real_imag_clean_noisy_noise.png")
    plt.close()


def plot_amp_phase(
    t_np: np.ndarray,
    E_in: torch.Tensor,
    E_out: torch.Tensor,
    Einput: torch.Tensor,
    name: str,
    num_periods: int,
) -> None:
    A_in, phi_in = get_amp_phase(E_in)
    A_out, phi_out = get_amp_phase(E_out)
    A_in_field, phi_in_field = get_amp_phase(Einput)

    plt.figure(figsize=(20, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_np, A_in, label="|E_in(t)| (SOA+ASE)")
    plt.plot(t_np, A_out, label="|E_out(t)| (after LPF)", alpha=0.8)
    plt.plot(t_np, A_in_field, label="A_in_field", alpha=0.8)
    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["amp_label"])
    plt.title(f"Amplitude through SOA+LPF, 10 GHz {name}")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_np, phi_in, label="arg(E_in(t)) (SOA+ASE)")
    plt.plot(t_np, phi_out, label="arg(E_out(t)) (after LPF)", alpha=0.8)
    plt.plot(t_np, phi_in_field, label="phi_in_field")
    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["phase_label"])
    plt.title(f"Phase through SOA+LPF, 10 GHz {name}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"soa_10GHz_{name}_{num_periods}periods_amp_phase_ase_lpf.png")
    plt.close()


def plot_dyn_vs_static(
    t_np: np.ndarray,
    Pin_np: np.ndarray,
    Pout_clean_dyn_np: np.ndarray,
    Pout_ase_np: np.ndarray,
    Pout_lpf_np: np.ndarray,
    t_period_ends: np.ndarray,
    Pout_filt_period_end: np.ndarray,
    sample_times: np.ndarray,
    Pout_clean_ss_np: np.ndarray,
    Pout_lpf_ss_np: np.ndarray,
    Pout_soa_collapsed_np: np.ndarray,
    name: str,
    n_periods: int,
) -> None:
    plt.figure(figsize=(20, 6))

    plt.subplot(2, 1, 1)
    traces_top = {
        "P_in(t)": Pin_np,
        "P_out_SOAdyn": Pout_clean_dyn_np,
        "P_out_ASE": Pout_ase_np,
    }
    style_keys_top = {
        "P_in(t)": "Pin",
        "P_out_SOAdyn": "Pout_SOA",
        "P_out_ASE": "Pout_ASE",
    }
    plot_traces(plt.gca(), t_np, traces_top, style_keys_top)
    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["power_label"])
    plt.title(f"SOA 10 GHz {name} input ({n_periods} periods): dyn vs static, ASE and LPF")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    traces_bot = {
        "P_out_SOAdyn": Pout_clean_dyn_np,
        "P_out_ASE": Pout_ase_np,
        "P_out_LPF": Pout_lpf_np,
    }
    style_keys_bot = {
        "P_out_SOAdyn": "Pout_SOA",
        "P_out_ASE": "Pout_ASE",
        "P_out_LPF": "Pout_LPF",
    }
    plot_traces(plt.gca(), t_np, traces_bot, style_keys_bot)

    marker_series_ss = {
        "SOA_SS_samples": Pout_clean_ss_np,
    }
    marker_keys_ss = {
        "SOA_SS_samples": "SOA_SS_samples",
    }
    plot_markers(
        plt.gca(),
        sample_times,
        marker_series_ss,
        marker_keys_ss,
        label_first_only=False,
        channel_index=0,
    )

    marker_series = {
        "LPF_SS": Pout_lpf_ss_np,
        "LPF_period_ends": Pout_filt_period_end,
        "SOA_collapsed": Pout_soa_collapsed_np,
    }
    marker_keys = {
        "LPF_SS": "LPF_SS",
        "LPF_period_ends": "LPF_period_ends",
        "SOA_collapsed": "SOA_collapsed",
    }
    plot_markers(
        plt.gca(),
        t_period_ends,
        marker_series,
        marker_keys,
        label_first_only=False,
        channel_index=0,
    )

    plt.xlabel(PLOT_STYLES["time_label"])
    plt.ylabel(PLOT_STYLES["power_label"])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"soa_10GHz_{name}_{n_periods}_periods_dyn_vs_static_ase_lpf.png")
    plt.close()


# ================================================================
# 4) Core helpers for TD runs and collapse
# ================================================================

def run_td_soa_once(
    soa: Soa,
    Ein: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return soa.solve_fields(Ein)


def run_td_soa_with_collapse(
    soa: Soa,
    Ein: torch.Tensor,
    idx: torch.Tensor,
    *,
    rng_state: torch.ByteTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soa._params.collapse_to_ss = True
    soa._rng.set_state(rng_state)
    E_det_coll, E_noisy_coll, E_noise_coll, _ = soa.solve_fields(Ein)
    assert E_noisy_coll.shape[-1] == idx.shape[-1]
    return E_det_coll, E_noisy_coll, E_noise_coll


def check_gradients(x: torch.Tensor, y: torch.Tensor, label: str) -> None:
    loss = (y.abs() ** 2).sum()
    loss.backward()
    assert x.grad is not None, f"{label}: grad is None"
    assert x.grad.shape == x.shape, f"{label}: wrong grad shape"
    assert torch.isfinite(x.grad).all(), f"{label}: non-finite grad"


# ================================================================
# 5) solve_fields unit tests (SS behavior + collapse)
# ================================================================

def test_solve_fields_no_ase():
    set_ase_seed(1234)
    soa = make_soa(use_ase=False, use_time_domain=False)
    x_power = torch.tensor([1e-3, 2e-3, 3e-3], dtype=torch.float32)
    x_field = powers_to_field(x_power)
    clean, noisy, noise, _ = soa.solve_fields(x_field)
    assert torch.allclose(noisy, clean, rtol=1e-6, atol=1e-9)
    assert torch.allclose(noise, torch.zeros_like(noise), rtol=0.0, atol=0.0)


def test_solve_fields_with_ase():
    set_ase_seed(1234)
    soa = make_soa(use_ase=True, use_time_domain=False)
    x_power = torch.tensor([1e-3, 2e-3, 3e-3], dtype=torch.float32)
    x_field = powers_to_field(x_power)
    clean, noisy, noise, _ = soa.solve_fields(x_field)
    assert torch.allclose(noisy, clean + noise, rtol=1e-6, atol=1e-9)
    assert not torch.allclose(noise, torch.zeros_like(noise), rtol=0.0, atol=0.0)


def test_solve_fields_time_domain_collapse_to_ss_input_ss():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=False,
    )
    soa_td_collapsed = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=True,
    )

    F = 4
    P_in = torch.linspace(1e-5, 4e-5, F, dtype=torch.float32)
    x = powers_to_field(P_in)

    noisy_td = soa_td(x)
    assert noisy_td.ndim == 2
    assert noisy_td.shape[0] == F

    noisy_collapsed = soa_td_collapsed(x)
    assert noisy_collapsed.shape == x.shape

    T_clk = 1.0 / clk_freq
    n_time = int(round(T_clk / dt))
    assert noisy_td.shape[1] == n_time

    noisy_last = noisy_td[..., -1]
    assert torch.allclose(noisy_collapsed, noisy_last, rtol=1e-6, atol=1e-8)


def test_solve_fields_time_domain_collapse_to_ss_explicit_time():
    set_aseed = set_ase_seed
    set_aseed(0)
    dt = DT_DEFAULT
    clk = CLK_FREQ_DEFAULT

    num_periods = N_PERIODS_DEFAULT
    t0, t_end, t_grid = make_time_grid(num_periods=num_periods, dt=dt, clk_freq=clk)

    fn = ConstantPin(1e-4)
    Pin = fn(t_grid)
    Ein = powers_to_field(Pin)

    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk,
        is_input_ss=False,
        collapse_to_ss=False,
    )
    soa_td_collapsed = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk,
        is_input_ss=False,
        collapse_to_ss=True,
    )

    noisy_full = soa_td(Ein)
    assert noisy_full.shape[-1] == Ein.shape[-1]

    idx = make_period_end_indices_from_signal(noisy_full, dt=dt, clk_freq=clk)
    n_periods = idx.numel()

    noisy_expected = noisy_full.index_select(-1, idx)

    noisy_coll = soa_td_collapsed(Ein)
    assert noisy_coll.shape == noisy_expected.shape
    assert noisy_coll.shape[-1] == n_periods
    assert torch.allclose(noisy_coll, noisy_expected, rtol=1e-6, atol=1e-8)


def test_time_domain_explicit_batched_collapse_shape():
    dt = DT_DEFAULT
    clk = CLK_FREQ_DEFAULT

    num_periods = 3
    t0, t_end, t_grid = make_time_grid(num_periods=num_periods, dt=dt, clk_freq=clk)

    B, F = 2, 3
    x_real = torch.sin(2 * math.pi * clk * t_grid)[None, None, :].expand(B, F, -1)
    x_imag = torch.zeros_like(x_real)
    Ein = torch.complex(x_real, x_imag)

    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk,
        is_input_ss=False,
        collapse_to_ss=False,
    )
    y_full = soa_td(Ein)
    assert y_full.shape == Ein.shape

    soa_td._params.collapse_to_ss = True
    y_coll = soa_td(Ein)

    # Indices at period centers instead of period ends
    T_clk = 1.0 / clk
    samples_per_period = int(round(T_clk / dt))
    total_T = y_full.shape[-1]
    n_periods = total_T // samples_per_period
    center_offset = samples_per_period // 2

    idx = torch.arange(n_periods, device=y_full.device) * samples_per_period + center_offset
    idx = idx.clamp(max=total_T - 1)

    y_expected = y_full.index_select(-1, idx)
    assert y_coll.shape == y_expected.shape
    assert torch.allclose(y_coll, y_expected, rtol=1e-6, atol=1e-8)


# ================================================================
# 6) LPF standalone test (with collapse)
# ================================================================

def test_butterworth_lpf_standalone():
    t_start = time.perf_counter()

    dt = 1e-12
    clk_freq = CLK_FREQ_DEFAULT
    T_clk = 1.0 / clk_freq
    samples_per_period = int(round(T_clk / dt))
    n_periods = N_PERIODS_DEFAULT
    T_total = samples_per_period * n_periods

    t = torch.arange(T_total, dtype=torch.float32) * dt

    f_sig = 5e9
    f_noise = 40e9
    E = torch.exp(1j * 2 * math.pi * f_sig * t) + 0.3 * torch.exp(
        1j * 2 * math.pi * f_noise * t
    )
    E = E.to(torch.complex64).view(1, 1, -1)

    cutoff = LPF_CUTOFF_DEFAULT
    order = LPF_ORDER_DEFAULT

    lpf_full, lpf_ss = make_lpf_td_and_ss(
        dt=dt,
        clk_freq=clk_freq,
        cutoff_td=cutoff,
        order=order,
        is_input_ss_td=False,
    )

    # Full TD LPF: should keep explicit time axis
    E_filt = lpf_full(E)
    assert E_filt.shape == E.shape
    assert torch.is_complex(E_filt)

    # SS LPF (Mode A + collapse_to_ss=True):
    # now returns period centers for all periods: [1,1,n_periods]
    E_ss = lpf_ss(E)
    assert E_ss.shape == (1, 1, n_periods)
    assert torch.is_complex(E_ss)

    # Recompute period centers from the full TD output
    total_T = E_filt.shape[-1]
    n_periods_eff = total_T // samples_per_period
    center_offset = samples_per_period // 2
    idx = (
        torch.arange(n_periods_eff, device=E_filt.device) * samples_per_period
        + center_offset
    )
    idx = idx.clamp(max=total_T - 1)

    # Expected centers from TD LPF: [1,1,n_periods_eff]
    E_expected = E_filt.index_select(-1, idx)

    # Compare last period center:
    # E_ss last center: [1,1]
    E_ss_last = E_ss[..., -1]
    # E_expected last center: [1,1]
    E_last = E_expected[..., -1]

    assert torch.allclose(E_ss_last, E_last, rtol=1e-5, atol=1e-7)

    L = 4096
    freqs = torch.fft.fftfreq(L, d=dt)
    H = lpf_full._butterworth_H(freqs)
    plot_lpf_frequency_response(
        freqs=freqs,
        H=H,
        cutoff=cutoff,
        filename="butterworth_lpf_frequency_response.png",
    )

    t_end = time.perf_counter()
    print(f"[PASSED] test_butterworth_lpf_standalone in {t_end - t_start:.3f} s")

# ================================================================
# 7) Time-domain 10 GHz inputs: SOA + LPF + collapse checks
# ================================================================

def test_time_domain_10GHz_inputs_with_ase_filter_and_ss_checks():
    t_start = time.perf_counter()

    set_ase_seed(1234)

    f_mod = CLK_FREQ_DEFAULT
    num_periods = N_PERIODS_DEFAULT
    dt = DT_DEFAULT

    T_mod = 1.0 / f_mod
    t0 = 0.0
    t_end = num_periods * T_mod

    pin_fns = make_pin_functions(f_mod, t_end)

    cutoff_freq = LPF_CUTOFF_WIDE
    order = LPF_ORDER_DEFAULT

    samples_per_period = period_samples_per_period(f_mod, dt)

    soa_td = make_soa(
        use_ase=True,
        use_time_domain=True,
        dt=dt,
        clk_freq=f_mod,
        is_input_ss=False,
        collapse_to_ss=False,
    )
    soa_td.set_ase_seed(1234)

    soa_ss = make_soa(
        use_ase=True,
        use_time_domain=False,
        dt=dt,
        clk_freq=f_mod,
        is_input_ss=False,
    )

    n_steps = int(round((t_end - t0) / dt))
    t_grid = torch.linspace(t0, t_end, n_steps + 1)

    lpf_td, lpf_ss = make_lpf_td_and_ss(
        dt=dt,
        clk_freq=f_mod,
        cutoff_td=cutoff_freq,
        order=order,
        is_input_ss_td=False,
    )

    for name, fn in pin_fns.items():
        Pin_td = fn(t_grid)
        Ein = powers_to_field(Pin_td)

        soa_td._params.collapse_to_ss = False
        rng_state = soa_td._rng.get_state()
        Eout_clean, Eout_noisy, Eout_noise, _ = run_td_soa_once(soa_td, Ein)

        Pout_ase = field_to_power(Eout_noisy)
        assert not torch.isnan(Pout_ase).any(), f"NaNs in ASE Pout for {name}"

        # LPF directly on 1D noisy time trace; last dim is time
        E_for_lpf = Eout_noisy
        E_out_filt_td = lpf_td(E_for_lpf)
        Pout_filt_td = field_to_power(E_out_filt_td)

        E_LPF_out = E_out_filt_td.view(-1)

        t_np = t_grid.cpu().numpy()

        # Optional plotting disabled here:
        # plot_real_imag_fields(...)
        # plot_amp_phase(...)

        assert not torch.isnan(Pout_filt_td).any(), f"NaNs in filtered Pout for {name}"
        assert torch.all(Pout_filt_td >= 0)

        # Use period centers instead of period ends
        total_T = E_out_filt_td.shape[-1]
        n_periods_eff = total_T // samples_per_period
        center_offset = samples_per_period // 2
        idx = (
            torch.arange(n_periods_eff, device=E_out_filt_td.device)
            * samples_per_period
            + center_offset
        )
        idx = idx[idx < total_T]
        n_periods_eff = idx.numel()

        # Steady-state LPF on same 1D trace: returns [n_periods_eff]
        E_filt_ss = lpf_ss(E_for_lpf)
        assert E_filt_ss.shape == (n_periods_eff,)

        E_expected = E_out_filt_td.index_select(-1, idx)  # [n_periods_eff]
        # Compare last center sample as SS representative
        E_last = E_expected[-1]
        E_last_ss = E_filt_ss[-1]
        assert torch.allclose(E_last_ss, E_last, rtol=1e-5, atol=1e-7)

        Pout_lpf_ss = field_to_power(E_expected)
        Pout_lpf_ss_np = Pout_lpf_ss.view(-1).cpu().numpy()

        Pout_filt_np = Pout_filt_td.view(-1).cpu().numpy()
        idx_np = idx.cpu().numpy()
        Pout_filt_period_center = Pout_filt_np[idx_np]
        t_period_centers = t_np[idx_np]

        E_det_collapsed, E_noisy_collapsed, E_noise_collapsed = run_td_soa_with_collapse(
            soa_td, Ein, idx, rng_state=rng_state
        )
        assert E_noisy_collapsed.shape[-1] == n_periods_eff
        E_ase_coll_expect = Eout_noisy.index_select(-1, idx)
        assert torch.allclose(E_noisy_collapsed, E_ase_coll_expect, rtol=1e-5, atol=1e-7)

        Pout_soa_collapsed_np = field_to_power(E_noisy_collapsed).view(-1).cpu().numpy()

        num_ss = 22
        sample_times = np.linspace(t0, t_end, num_ss, endpoint=False)
        Pin_np = Pin_td.cpu().numpy()
        sample_idx = time_to_indices(sample_times, dt, len(Pin_np))
        Pin_samples = np.clip(Pin_np[sample_idx], a_min=0.0, a_max=None)

        pin_ss = torch.from_numpy(Pin_samples.astype(np.float32))
        pin_ss_field = powers_to_field(pin_ss)
        y_clean_ss = soa_ss(pin_ss_field)
        Pout_clean_ss_np = field_to_power(y_clean_ss).cpu().numpy()

        Pout_ase_np = Pout_ase.cpu().numpy()
        Pout_lpf_np = Pout_filt_np
        Pout_clean_dyn_np = field_to_power(Eout_clean).cpu().numpy()

        plot_dyn_vs_static(
            t_np=t_np,
            Pin_np=Pin_np,
            Pout_clean_dyn_np=Pout_clean_dyn_np,
            Pout_ase_np=Pout_ase_np,
            Pout_lpf_np=Pout_lpf_np,
            t_period_ends=t_period_centers,              # centers now
            Pout_filt_period_end=Pout_filt_period_center,
            sample_times=sample_times,
            Pout_clean_ss_np=Pout_clean_ss_np,
            Pout_lpf_ss_np=Pout_lpf_ss_np,
            Pout_soa_collapsed_np=Pout_soa_collapsed_np,
            name=name,
            n_periods=n_periods_eff,
        )

    t_end_wall = time.perf_counter()
    print(
        f"[PASSED] test_time_domain_10GHz_inputs_with_ase_filter_and_ss_checks in "
        f"{t_end_wall - t_start:.3f} s"
    )


# ================================================================
# 8) SOA Gradient + shape tests
# ================================================================

def test_gradients_steady_state_shape_and_value():
    set_ase_seed(0)
    soa = make_soa(use_ase=False, use_time_domain=False)

    P_in = torch.tensor([1e-4, 2e-4, 3e-4], dtype=torch.float64)
    x = torch.sqrt(P_in).to(torch.complex128)
    x.requires_grad_(True)

    y = soa(x)
    check_gradients(x, y, "SS")


def test_gradients_time_domain_ss_input_td_ste():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    F = 8
    P_in = torch.linspace(1e-5, 8e-5, F, dtype=torch.float64)
    x = torch.sqrt(P_in).to(torch.complex128)
    x.requires_grad_(True)

    # Use x directly as the SS input
    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=False,
    )

    y = soa_td(x)
    assert y.ndim == 2
    assert y.shape[0] == F

    check_gradients(x, y, "TD-SS")


def test_gradients_time_domain_explicit_time_no_ste():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=False,
        collapse_to_ss=False,
    )

    num_periods = N_PERIODS_DEFAULT
    t0, t_end, t_grid = make_time_grid(num_periods=num_periods, dt=dt, clk_freq=clk_freq)

    fn = ConstantPin(1e-4)
    Pin = fn(t_grid).to(torch.float64)
    x = torch.sqrt(Pin).to(torch.complex128)
    x.requires_grad_(True)

    y = soa_td(x)
    check_gradients(x, y, "TD-explicit")


def test_gradients_soa_plus_lpf_time_domain_ss_input_ste():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    soa_td = make_soa(
        use_ase=False,
        use_time_domain=True,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=False,
    )

    B = 1
    F = 8
    P_in = torch.linspace(1e-5, 8e-5, F, dtype=torch.float64)
    x = torch.sqrt(P_in).to(torch.complex128).view(B, F)
    x.requires_grad_(True)

    y_soa = soa_td(x)

    lpf_td, _ = make_lpf_td_and_ss(
        dt=dt,
        clk_freq=clk_freq,
        cutoff_td=LPF_CUTOFF_DEFAULT,
        order=LPF_ORDER_DEFAULT,
        is_input_ss_td=False,
    )

    y_lpf = lpf_td(y_soa)
    check_gradients(x, y_lpf, "SOA+LPF-TD-SS")


def test_soa_time_domain_batch_as_time_shape():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT
    T = clk_freq * dt

    B = 5
    F = 4

    base_powers = torch.tensor([1e-5, 1e-6, 1e-7, 5e-6, 2e-6], dtype=torch.float32)
    assert base_powers.shape[0] == B
    channel_factors = torch.linspace(1.0, 2.0, F, dtype=torch.float32)
    P_in = base_powers.unsqueeze(1) * channel_factors.unsqueeze(0)

    soa_td, y_noisy = run_soa_td_from_ss(
        P_in,
        use_ase=False,
        collapse_to_ss=False,
        batch_as_time=True,
        dt=dt,
        clk_freq=clk_freq,
    )

    assert y_noisy.ndim == 2
    BT_out, F_out = y_noisy.shape
    assert F_out == F
    assert BT_out > B * T

    Ein_ss = powers_to_field(P_in).to(torch.complex64)
    Ein_ss64 = Ein_ss.to(torch.complex128)
    Ein_ss64.requires_grad_(True)
    y_api = soa_td(Ein_ss64)
    assert y_api.shape == (BT_out, F)
    check_gradients(Ein_ss64, y_api, "TD-batch-as-time")

    soa_td._params.collapse_to_ss = True
    Ein_ss2 = Ein_ss.clone().to(torch.complex64)
    y_det_c, y_noisy_c, y_noise_c, _ = soa_td.solve_fields(Ein_ss2)
    assert y_noisy_c.shape == (B, F)


# ================================================================
# 9) LPF gradient tests
# ================================================================

def test_lpf_gradients_steady_state_bypass():
    dt = DT_DEFAULT

    params = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_DEFAULT,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=False,
        collapse_to_ss=False,
        clk_freq=CLK_FREQ_DEFAULT,
        is_input_ss=False,
    )
    lpf = OpButterworthLPF(params=params)

    x = torch.randn(5, dtype=torch.complex128)
    x.requires_grad_(True)

    y = lpf(x)
    check_gradients(x, y, "LPF-SS")


def test_lpf_gradients_time_domain_explicit_time():
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    num_periods = N_PERIODS_DEFAULT
    t0, t_end, t_grid = make_time_grid(num_periods=num_periods, dt=dt, clk_freq=clk_freq)

    f_sig = 5e9
    x_real = torch.sin(2 * math.pi * f_sig * t_grid)
    x_imag = torch.cos(2 * math.pi * f_sig * t_grid)
    x = torch.complex(x_real, x_imag).to(torch.complex128)
    x = x.view(1, 1, -1)
    x.requires_grad_(True)

    params_td = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_DEFAULT,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=False,
        clk_freq=clk_freq,
        is_input_ss=False,
    )
    lpf_td = OpButterworthLPF(params=params_td)

    y = lpf_td(x)
    check_gradients(x, y, "LPF-TD-explicit")


def test_lpf_gradients_time_domain_ss_input_td():
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    params_td_ss = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_DEFAULT,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=False,
        clk_freq=clk_freq,
        is_input_ss=True,
    )
    lpf_td_ss = OpButterworthLPF(params_td_ss)

    B = 1
    F = 8
    P_in = torch.linspace(1e-4, 8e-4, F, dtype=torch.float64)
    x = torch.sqrt(P_in).to(torch.complex128).view(B, F)
    x.requires_grad_(True)

    y = lpf_td_ss(x)
    assert y.ndim == 3
    check_gradients(x, y, "LPF-TD-SS")


# ================================================================
# 10) Batch tests with plots (no shape logic beyond sanity)
# ================================================================

def test_batch_N_channels_time_domain_soa_lpf_and_collapse():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    T_clk = 1.0 / clk_freq
    n_time = int(round(T_clk / dt))
    t = torch.arange(n_time, dtype=torch.float32) * dt

    B = 5
    F = 2

    base_b = torch.linspace(1.0, 3.0, B, dtype=torch.float32)
    factors_f = torch.tensor([1.0, 4.0], dtype=torch.float32)
    P_base = 1e-5

    P0 = (base_b.unsqueeze(1) * factors_f.unsqueeze(0)) * P_base

    soa_td, E_soa_td = run_soa_td_from_ss(
        P0,
        use_ase=True,
        collapse_to_ss=False,
        batch_as_time=False,
        dt=dt,
        clk_freq=clk_freq,
    )

    _, E_soa_coll = run_soa_td_from_ss(
        P0,
        use_ase=True,
        collapse_to_ss=True,
        batch_as_time=False,
        dt=dt,
        clk_freq=clk_freq,
    )

    lpf_params_td = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_WIDE,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=False,
        clk_freq=clk_freq,
        is_input_ss=False,
    )
    lpf_td = OpButterworthLPF(lpf_params_td)
    E_lpf_td = lpf_td(E_soa_td)

    lpf_params_coll = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_WIDE,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=True,
        clk_freq=clk_freq,
        is_input_ss=False,
    )
    lpf_coll = OpButterworthLPF(lpf_params_coll)
    E_lpf_coll = lpf_coll(E_soa_td)

    t_np = t.cpu().numpy()

    fig, axes = plt.subplots(
        nrows=F,
        ncols=B,
        figsize=(3 * B, 3 * F),
        sharex=False,
        sharey="row",
    )
    if F == 1:
        axes = np.expand_dims(axes, axis=0)
    if B == 1:
        axes = np.expand_dims(axes, axis=1)

    line_keys = {
        "P in (SS)": "Pin",
        "P SOA_out": "Pout_SOA",
        "P LPF_out": "Pout_LPF",
    }

    # collapsed value is per period (one period here), use center of period
    T_clk = 1.0 / clk_freq
    t_center_scalar = T_clk / 2.0

    for f in range(F):
        for b in range(B):
            ax = axes[f, b]

            P_in_ch = P0[b, f].item() * np.ones_like(t_np)
            P_soa_ch = field_to_power(E_soa_td[b, f]).detach().cpu().numpy()
            P_lpf_ch = field_to_power(E_lpf_td[b, f]).detach().cpu().numpy()

            traces = {
                "P in (SS)": P_in_ch,
                "P SOA_out": P_soa_ch,
                "P LPF_out": P_lpf_ch,
            }
            plot_traces(ax, t_np, traces, line_keys)

            P_soa_coll = field_to_power(E_soa_coll[b, f]).item()
            P_lpf_coll_val = field_to_power(E_lpf_coll[b, f]).item()
            marker_series = {
                "SOA_collapsed": P_soa_coll,
                "LPF_collapsed": P_lpf_coll_val,
            }
            marker_keys = {
                "SOA_collapsed": "SOA_collapsed",
                "LPF_collapsed": "LPF_collapsed",
            }

            # use period center as x for collapsed markers
            t_center = np.array([t_center_scalar], dtype=np.float32)

            plot_markers(
                ax,
                t_center,
                marker_series,
                marker_keys,
                label_first_only=True,
                channel_index=f if b == 0 else 1,
            )

            if f == F - 1:
                ax.set_xlabel(PLOT_STYLES["time_label"])
            if b == 0:
                ax.set_ylabel(PLOT_STYLES["power_label"])

            ax.set_title(f"Ch {f}, img {b}")
            ax.grid(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"Batch{B}, F={F}: per-image power vs time (SOA + LPF + collapse)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(PLOT_DIR / f"batch{B}_allchannels_soa_lpf_td_power_grid.png")
    plt.close(fig)


def test_batch_as_time_eight_images_four_channels_with_lpf():
    set_ase_seed(0)
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT

    B = 8
    F = 3

    # ------------------------------------------------------------------
    # Construct per-image powers P_in: [B, F]
    # ------------------------------------------------------------------
    base_b = torch.logspace(-7, -5, B, dtype=torch.float32)
    factors_f = torch.logspace(1.0, 2.0, F, dtype=torch.float32)
    bf_grid_b = torch.arange(B, dtype=torch.float32).unsqueeze(1)
    bf_grid_f = torch.arange(F, dtype=torch.float32).unsqueeze(0)

    P_images = base_b.unsqueeze(1) * factors_f.unsqueeze(0)
    P_images = P_images * (1.0 + 0.05 * bf_grid_b + 0.02 * bf_grid_f)

    # Shuffle images along B
    perm = torch.randperm(B)
    P_in = P_images[perm]  # [B, F]

    # ------------------------------------------------------------------
    # SOA TD from SS, batch_as_time=True (SOA Mode B)
    # ------------------------------------------------------------------
    soa_td, y_noisy_big = run_soa_td_from_ss(
        P_in,
        use_ase=True,
        collapse_to_ss=False,
        batch_as_time=True,
        dt=dt,
        clk_freq=clk_freq,
    )

    assert y_noisy_big.ndim == 2
    B_big, F_out = y_noisy_big.shape
    assert F_out == F

    # Check that long TD trace length matches expected n_time per image
    T_clk = 1.0 / clk_freq
    n_time = int(round(T_clk / dt))
    assert B_big % B == 0
    T_out = B_big // B
    assert T_out == n_time

    # [B_big, F] -> [F, B_big] for plotting as channels x time
    y_noisy_FT = y_noisy_big.transpose(0, 1)  # [F, B_big]

    # ------------------------------------------------------------------
    # SOA TD collapsed (Mode B, collapse_to_ss=True) -> [B, F]
    # ------------------------------------------------------------------
    _, y_noisy_coll = run_soa_td_from_ss(
        P_in,
        use_ase=True,
        collapse_to_ss=True,
        batch_as_time=True,
        dt=dt,
        clk_freq=clk_freq,
    )
    P_soa_coll = field_to_power(y_noisy_coll)  # [B, F]

    # ------------------------------------------------------------------
    # SOA SS reference (no TD) -> [B, F]
    # ------------------------------------------------------------------
    soa_ss = make_soa(
        use_ase=True,
        use_time_domain=False,
        dt=dt,
        clk_freq=clk_freq,
        is_input_ss=True,
        collapse_to_ss=False,
        batch_as_time=False,
    )
    Ein_ss = powers_to_field(P_in).to(torch.complex64)  # [B, F]
    y_ss = soa_ss(Ein_ss)
    P_soa_ss = field_to_power(y_ss)  # [B, F]

    # ------------------------------------------------------------------
    # LPF on explicit batch-as-time TD input: Mode D
    #   E_for_lpf: [B_big, F], time along B_big
    # ------------------------------------------------------------------
    E_for_lpf = y_noisy_big  # [B_big, F]

    lpf_params_td = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_WIDE,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=False,
        clk_freq=clk_freq,
        is_input_ss=False,
        batch_as_time=True,
    )
    lpf_td = OpButterworthLPF(lpf_params_td)

    # Full TD filtered trace: [B_big, F]
    E_lpf_t = lpf_td(E_for_lpf)
    E_lpf_FT = E_lpf_t.transpose(0, 1)  # [F, B_big] for plotting

    # Collapsed LPF: per-period along long time axis (Mode D, collapse_to_ss=True)
    lpf_params_coll = make_lpf_params(
        dt=dt,
        cutoff=LPF_CUTOFF_WIDE,
        order=LPF_ORDER_DEFAULT,
        use_time_domain=True,
        collapse_to_ss=True,
        clk_freq=clk_freq,
        is_input_ss=False,
        batch_as_time=True,
    )
    lpf_coll = OpButterworthLPF(lpf_params_coll)

    E_lpf_coll = lpf_coll(E_for_lpf)            # [n_periods, F]
    P_lpf_coll = field_to_power(E_lpf_coll)     # [n_periods, F]

    # ------------------------------------------------------------------
    # Build common time axis for full long trace and image-center times
    # ------------------------------------------------------------------
    P_in_flat = torch.zeros(F, B * T_out, dtype=torch.float32)
    for b in range(B):
        start = b * T_out
        stop = (b + 1) * T_out
        P_in_flat[:, start:stop] = P_in[b].unsqueeze(1)  # [F, T_out] per image

    t_full = torch.arange(B * T_out, dtype=torch.float32) * dt  # [B_big]
    t_full_np = t_full.cpu().numpy()

    # image centers instead of image ends
    img_center_idx = torch.arange(0, B, dtype=torch.long) * T_out + (T_out // 2)  # [B]
    t_img_centers = (img_center_idx.to(torch.float32) * dt).cpu().numpy()         # [B]

    # ------------------------------------------------------------------
    # Compute LPF period-center times to align with P_lpf_coll
    # ------------------------------------------------------------------
    samples_per_period_f = 1.0 / (clk_freq * dt)
    samples_per_period = int(round(samples_per_period_f))

    idx_lpf = (
        torch.arange(0, B_big // samples_per_period, dtype=torch.long)
        * samples_per_period
        + (samples_per_period // 2)
    )
    idx_lpf = idx_lpf[idx_lpf < B_big]                         # [n_periods]
    t_lpf_centers = (idx_lpf.to(torch.float32) * dt).cpu().numpy()  # [n_periods]

    # Sanity: match n_periods from collapse and index computation
    n_periods_lpf, F_coll = P_lpf_coll.shape
    assert F_coll == F
    assert n_periods_lpf == idx_lpf.numel()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = make_channel_axes(F)

    line_keys = {
        "P in": "Pin",
        "P SOA_out": "Pout_SOA",
        "P LPF_out": "Pout_LPF",
    }
    marker_keys = {
        "SOA collapsed": "SOA_collapsed",
        "LPF collapsed (per period)": "LPF_collapsed",
        "SOA SS": "SOA_SS",
    }

    P_lpf_coll_np = P_lpf_coll.detach().cpu().numpy()  # [n_periods, F]

    for f in range(F):
        ax = axes[f]

        # Full TD traces
        P_in_full = P_in_flat[f].cpu().numpy()                 # [B_big]
        P_soa_full = field_to_power(y_noisy_FT[f]).detach().cpu().numpy()   # [B_big]
        P_lpf_full = field_to_power(E_lpf_FT[f]).detach().cpu().numpy()     # [B_big]

        traces = {
            "P in": P_in_full,
            "P SOA_out": P_soa_full,
            "P LPF_out": P_lpf_full,
        }
        plot_traces(ax, t_full_np, traces, line_keys)

        # Collapsed values
        P_soa_coll_f = P_soa_coll[:, f].cpu().numpy()      # [B]
        P_lpf_coll_f = P_lpf_coll_np[:, f]                 # [n_periods]
        P_soa_ss_f = P_soa_ss[:, f].cpu().numpy()          # [B]

        # SOA collapsed + SOA SS at image-center times
        markers_img = {
            "SOA collapsed": P_soa_coll_f,
            "SOA SS": P_soa_ss_f,
        }
        plot_markers(
            ax,
            t_img_centers,
            markers_img,
            marker_keys,
            label_first_only=True,
            channel_index=f,
        )

        # LPF collapsed at per-period centers along long TD axis
        markers_lpf = {
            "LPF collapsed (per period)": P_lpf_coll_f,
        }
        plot_markers(
            ax,
            t_lpf_centers,
            markers_lpf,
            marker_keys,
            label_first_only=True,
            channel_index=f,
        )

        ax.set_ylabel(PLOT_STYLES["power_label"])
        ax.set_title(f"Channel {f} (B={B} shuffled images)")
        ax.grid(True)

    axes[-1].set_xlabel(PLOT_STYLES["time_label"])
    axes[0].legend()
    fig.suptitle("Batch-as-time: shuffled images, SOA TD vs LPF, collapsed vs SS")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(PLOT_DIR / f"batch_as_time_B{B}_F{F}_images_power_collapsed.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    P_soa_td = field_to_power(y_noisy_FT)
    P_lpf_full_all = field_to_power(E_lpf_FT)
    assert torch.isfinite(P_soa_td).all()
    assert torch.isfinite(P_lpf_full_all).all()
    assert (P_soa_td >= 0).all()
    assert (P_lpf_full_all >= 0).all()


def get_mem_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1e9

def test_batch_as_time_with_channels_and_lpf_perf():
    dt = DT_DEFAULT
    clk_freq = CLK_FREQ_DEFAULT
    T_clk = 1.0 / clk_freq
    n_time = int(round(T_clk / dt))
    
    print(f"DT: {dt:.0e}, CLK: {clk_freq:.0e} Hz, n_time: {n_time}")
    
    # Test configs: vary B and F
    configs = [
        (4, 2),   # Small
        (16, 3),  # Original
        (32, 8),  # Medium
        (64, 16), # Large
    ]
    
    results = []
    
    for B, F in configs:
        print(f"\n{'='*60}")
        print(f"Testing B={B}, F={F}")
        print(f"{'='*60}")
        
        gc.collect()
        
        # Baseline mem
        mem_base = get_mem_usage()
        
        t0 = time.perf_counter()
        
        # ------------------------------------------------------------------
        # Construct P_in: [B, F]
        # ------------------------------------------------------------------
        base_b = torch.logspace(-7, -5, B, dtype=torch.float32)
        factors_f = torch.logspace(1.0, 2.0, F, dtype=torch.float32)
        bf_grid_b = torch.arange(B, dtype=torch.float32).unsqueeze(1)
        bf_grid_f = torch.arange(F, dtype=torch.float32).unsqueeze(0)
        
        P_images = base_b.unsqueeze(1) * factors_f.unsqueeze(0)
        P_images = P_images * (1.0 + 0.05 * bf_grid_b + 0.02 * bf_grid_f)
        perm = torch.randperm(B)
        P_in = P_images[perm]  # [B, F]
        
        # ------------------------------------------------------------------
        # SOA TD full (batch_as_time=True)
        # ------------------------------------------------------------------
        set_ase_seed(0)
        soa_td, y_noisy_big = run_soa_td_from_ss(
            P_in,
            use_ase=True,
            collapse_to_ss=False,
            batch_as_time=True,
            dt=dt,
            clk_freq=clk_freq,
        )
        assert y_noisy_big.ndim == 2
        B_big, F_out = y_noisy_big.shape
        assert F_out == F
        assert B_big % B == 0
        T_out = B_big // B
        assert T_out == n_time
        
        # SOA collapsed
        _, y_noisy_coll = run_soa_td_from_ss(
            P_in,
            use_ase=True,
            collapse_to_ss=True,
            batch_as_time=True,
            dt=dt,
            clk_freq=clk_freq,
        )
        P_soa_coll = field_to_power(y_noisy_coll)
        
        # SOA SS reference
        soa_ss = make_soa(
            use_ase=True,
            use_time_domain=False,
            dt=dt,
            clk_freq=clk_freq,
            is_input_ss=True,
            collapse_to_ss=False,
            batch_as_time=False,
        )
        Ein_ss = powers_to_field(P_in).to(torch.complex64)
        y_ss = soa_ss(Ein_ss)
        P_soa_ss = field_to_power(y_ss)
        
        # ------------------------------------------------------------------
        # LPF full and collapsed on SOA TD output
        # ------------------------------------------------------------------
        E_for_lpf = y_noisy_big  # [B*n_time, F]
        
        lpf_params_td = make_lpf_params(
            dt=dt,
            cutoff=LPF_CUTOFF_WIDE,
            order=LPF_ORDER_DEFAULT,
            use_time_domain=True,
            collapse_to_ss=False,
            clk_freq=clk_freq,
            is_input_ss=False,
            batch_as_time=True,
        )
        lpf_td = OpButterworthLPF(lpf_params_td)
        E_lpf_t = lpf_td(E_for_lpf)
        
        lpf_params_coll = make_lpf_params(
            dt=dt,
            cutoff=LPF_CUTOFF_WIDE,
            order=LPF_ORDER_DEFAULT,
            use_time_domain=True,
            collapse_to_ss=True,
            clk_freq=clk_freq,
            is_input_ss=False,
            batch_as_time=True,
        )
        lpf_coll = OpButterworthLPF(lpf_params_coll)
        E_lpf_coll = lpf_coll(E_for_lpf)
        
        # ------------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------------
        assert E_lpf_coll.shape == (B, F), f"Expected [Batch={B}, F={F}], got {E_lpf_coll.shape}"
        P_lpf_coll = field_to_power(E_lpf_coll).mean(0)  # [F]
        
        rel_err_ss = torch.abs(P_soa_coll - P_soa_ss) / (P_soa_ss + 1e-12)
        rel_err_lpf = torch.abs(P_lpf_coll.unsqueeze(0) - P_soa_coll) / (P_soa_coll + 1e-12)
        
        t_total = time.perf_counter() - t0
        mem_peak = get_mem_usage()
        
        result = {
            'B': B, 'F': F, 'n_time': n_time, 'B_big': B_big,
            'time_s': t_total,
            'mem_gb': mem_peak - mem_base,
            'ss_max_err': rel_err_ss.max().item(),
            'lpf_max_err': rel_err_lpf.max().item(),
        }
        results.append(result)
        
        print(f"Time: {t_total:.3f}s")
        print(f"Peak mem delta: {result['mem_gb']:.2f} GB")
        print(f"SS max rel err: {result['ss_max_err']:.2%}")
        print(f"LPF max rel err: {result['lpf_max_err']:.2%}")
        print(f"Shapes OK: y_noisy_big [{B_big},{F}], E_lpf_coll [{n_time},{F}]")
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"B={r['B']:2d} F={r['F']:2d}: {r['time_s']:6.3f}s | Δmem {r['mem_gb']:5.2f}GB | "
              f"SSerr {r['ss_max_err']:4.1%} | LPFerr {r['lpf_max_err']:4.1%}")
