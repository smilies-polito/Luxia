import math
from copy import deepcopy
from typing import Optional, NamedTuple, Tuple

import torch

from luxia import LxModule
from luxia.params import SoaParams
from luxia.util.log_config import lx_log
from luxia.util.util_functions import (
    flatten_tail,
    unflatten_tail,
)
from luxia.util.util_time_domain_functions import (
    collapse_last_dim,
    expand_ss_to_time,
)


def set_ase_seed(seed: int = 0) -> None:
    """
    Global helper for legacy code: seeds PyTorch global RNGs and
    sets the default SOA CPU generator state.

    For new code prefer Soa.set_ase_seed(seed) on a specific instance.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    Soa._default_cpu_rng = torch.Generator(device="cpu").manual_seed(seed)


class SoaSSResult(NamedTuple):
    phase: torch.Tensor
    P_in: torch.Tensor
    h_ss: torch.Tensor
    G_ss: torch.Tensor
    P_out_phys_ss: torch.Tensor
    field_clean_ss: torch.Tensor
    output_field_ss: torch.Tensor
    noise_ss: torch.Tensor


class _SoaSTETimeDomain(torch.autograd.Function):
    """
    Straight-through estimator wrapper for SOA time-domain outputs.
    """

    @staticmethod
    def forward(
        ctx,
        noisy_td: torch.Tensor,   # [..., F, T]
        noisy_ss: torch.Tensor,   # [..., F]
    ) -> torch.Tensor:
        ctx.save_for_backward(noisy_ss)
        return noisy_td

    @staticmethod
    def backward(
        ctx,
        grad_noisy_td: torch.Tensor,  # [..., F, T]
    ):
        (noisy_ss,) = ctx.saved_tensors

        if grad_noisy_td.ndim < 2:
            raise RuntimeError(
                f"_SoaSTETimeDomain.backward: expected [..., F, T], "
                f"got shape {grad_noisy_td.shape}"
            )
        if grad_noisy_td.shape[:-1] != noisy_ss.shape:
            raise RuntimeError(
                f"_SoaSTETimeDomain.backward: shape mismatch, "
                f"grad_noisy_td[..., T] has {grad_noisy_td.shape[:-1]}, "
                f"noisy_ss has {noisy_ss.shape}"
            )

        grad_noisy_ss = grad_noisy_td.mean(dim=-1)
        return None, grad_noisy_ss


class Soa(LxModule):
    r"""
    Differentiable Semiconductor Optical Amplifier (SOA).
    """

    _H_MIN = -50.0
    _H_MAX = 50.0

    _default_cpu_rng: torch.Generator = torch.Generator(device="cpu").manual_seed(0)

    def __init__(
        self,
        soa_params: Optional[SoaParams] = None,
        debug: bool = False,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__(debug=debug)

        if soa_params is None:
            soa_params = SoaParams()
        self._params = deepcopy(soa_params)

        self._h0 = self._params.g0 * self._params.L  # initial guess for h

        c = 3e8
        self._hnu = 6.62607015e-34 * c / self._params.lam0

        if rng is not None:
            self._rng = rng
        else:
            self._rng = torch.Generator(device="cpu")
            self._rng.manual_seed(int(Soa._default_cpu_rng.initial_seed()))

        self._last_noisy: Optional[torch.Tensor] = None
        self._last_noise: Optional[torch.Tensor] = None
        self._last_clean: Optional[torch.Tensor] = None

        self._inv_P_sat = 1.0 / self._params.P_sat
        self._inv_tau_c = 1.0 / self._params.tau_c

    # -------------------------------------------------------------
    # Small helpers for flags and RNG
    # -------------------------------------------------------------

    def _is_time_domain(self) -> bool:
        return bool(self._params.use_time_domain)

    def _is_input_ss(self) -> bool:
        return bool(self._params.is_input_ss)

    def _collapse_to_ss(self) -> bool:
        return bool(self._params.collapse_to_ss)

    def _batch_as_time(self) -> bool:
        return bool(getattr(self._params, "batch_as_time", False))

    def _ensure_rng_device(self, device: torch.device) -> None:
        if self._rng.device != device:
            seed = int(self._rng.initial_seed())
            self._rng = torch.Generator(device=device).manual_seed(seed)

    # -------------------------------------------------------------
    # Debug helpers (state)
    # -------------------------------------------------------------

    def get_last_noise(self) -> Optional[torch.Tensor]:
        return self._last_noise

    def get_last_clean_field(self) -> Optional[torch.Tensor]:
        return self._last_clean

    def get_last_noisy_field(self) -> Optional[torch.Tensor]:
        return self._last_noisy

    def set_ase_seed(self, seed: int) -> None:
        """
        Set the RNG seed for ASE noise generation in this module only.
        Does not affect global torch RNGs.
        """
        if self._rng.device.type == "cpu":
            self._rng.manual_seed(seed)
        else:
            self._rng = torch.Generator(device=self._rng.device).manual_seed(seed)

    # -------------------------------------------------------------
    # Power and field helpers
    # -------------------------------------------------------------

    def _field_to_power(self, field: torch.Tensor) -> torch.Tensor:
        return field.abs().square()

    def _scale_power_in(self, P: torch.Tensor) -> torch.Tensor:
        return P * self._params.P_ref if self._params.norm_input else P

    def _scale_power_out(self, P: torch.Tensor) -> torch.Tensor:
        return P * self._params.P_ref if self._params.norm_output else P

    def _input_power(self, field: torch.Tensor) -> torch.Tensor:
        return self._scale_power_in(self._field_to_power(field))

    def _output_power(self, field: torch.Tensor) -> torch.Tensor:
        return self._scale_power_out(self._field_to_power(field))

    def _gain_linear(self, h: torch.Tensor) -> torch.Tensor:
        """
        Linear gain G(h) = exp(h).
        """
        return torch.exp(h)

    def _field_out(
        self,
        P_out_phys: torch.Tensor,
        phase_in: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct output field from physical output power and phase.
        """
        P_norm = P_out_phys / self._params.P_ref if self._params.norm_output else P_out_phys
        amp = torch.sqrt(P_norm.clamp(min=0.0))
        phase = phase_in - 0.5 * self._params.alpha * h
        return torch.polar(amp, phase)

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------

    def _debug_log_forward(
        self,
        input_field: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        noise: torch.Tensor,
        ss: Optional[SoaSSResult],
    ) -> None:
        """
        Minimal debug logger for SOA forward().

        Keeps logging lightweight and never touches global torch print options.
        This avoids compatibility issues across torch versions.
        """
        if not self._debug:
            return

        try:
            lx_log.debug(
                "[SOA] forward: "
                f"in={self._field_to_power(input_field[0])} "
                f"TotIn={torch.sum(self._field_to_power(input_field))} "
                f"in={tuple(input_field.shape)} "
                f"clean={tuple(clean.shape)} "
                f"noisy={tuple(noisy.shape)} "
                f"TotOut={torch.sum(self._field_to_power(noisy))} "
                f"noise={tuple(noise.shape)} "
                f"td={self._is_time_domain()} "
                f"input_ss={self._is_input_ss()} "
                f"batch_as_time={self._batch_as_time()} "
                f"collapse={self._collapse_to_ss()}"
            )

            if ss is not None and ss.P_in.numel() > 0:
                p_in0 = float(ss.P_in.reshape(-1)[0].item())
                p_out0 = float(self._output_power(ss.output_field_ss).reshape(-1)[0].item())
                lx_log.debug(
                    f"[SOA] ss sample: P_in[0]={p_in0:.3e}, P_out[0]={p_out0:.3e}"
                )

            if self._is_time_domain() and noisy.numel() > 0:
                p_noisy = self._output_power(noisy.reshape(-1))
                n_show = min(4, p_noisy.numel())
                sample_vals = [float(v.item()) for v in p_noisy[:n_show]]
                lx_log.debug(f"[SOA] td sample powers: {sample_vals}")

        except Exception as e:
            lx_log.debug(f"[SOA] debug logging skipped: {e}")

    # ===============================================
    # Gain equations
    # ================================================================

    @property
    def _g0L(self) -> float:
        return self._params.g0 * self._params.L

    @staticmethod
    def _clamp_h(h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(h, Soa._H_MIN, Soa._H_MAX)

    def _F_gain(self, h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        exp_h = torch.exp(h)
        return (self._g0L - h) - (P * self._inv_P_sat) * (exp_h - 1.0)

    def _dF_gain_dh(self, h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        return -1.0 - (P * self._inv_P_sat) * torch.exp(h)

    def _dh_dt(self, h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        return self._F_gain(h, P) * self._inv_tau_c

    # ================================================================
    # Steady-state solver
    # ================================================================

    def _solve_gain_steady_state(self, P: torch.Tensor) -> torch.Tensor:
        """
        Newton solver for h given input power P.
        """
        h = torch.full_like(P, self._h0)
        max_iters = self._params.newton_iters
        tol = self._params.newton_tol
        eps = 1e-9

        converged = False
        for _ in range(max_iters):
            F = self._F_gain(h, P)
            dF = self._dF_gain_dh(h, P)
            safe_dF = torch.where(
                dF.abs() < eps,
                dF.sign() * eps,
                dF,
            )
            step = F / safe_dF
            h = self._clamp_h(h - step)
            if step.abs().max() < tol:
                converged = True
                break

        if self._debug and not converged:
            lx_log.debug(
                "[SS DEBUG] Newton solver did not fully converge: "
                f"max residual={F.abs().max().item():.3e}, "
                f"max step={step.abs().max().item():.3e}"
            )

        return h

    # ================================================================
    # ASE noise
    # ================================================================

    def _ase_variance(self, G: torch.Tensor) -> torch.Tensor:
        P_ase = self._params.n_sp * (G - 1.0) * self._hnu * self._params.B0
        return P_ase.clamp(min=0.0)

    def _add_ase_noise(
        self,
        field: torch.Tensor,
        G: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._params.use_ase:
            noise = torch.zeros_like(field)
            return field, noise

        sigma2 = self._ase_variance(G) / 2.0
        sigma = torch.sqrt(sigma2)
        while sigma.ndim < field.ndim:
            sigma = sigma.unsqueeze(-1)

        self._ensure_rng_device(field.device)
        gen = self._rng

        noise_r = torch.randn(
            field.shape,
            dtype=field.real.dtype,
            device=field.device,
            generator=gen,
        ) * sigma
        noise_i = torch.randn(
            field.shape,
            dtype=field.real.dtype,
            device=field.device,
            generator=gen,
        ) * sigma
        noise = torch.complex(noise_r, noise_i)
        field_noisy = field + noise
        return field_noisy, noise

    def _noisy_output_field(
        self,
        P_out_phys: torch.Tensor,
        phase_in: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field_clean = self._field_out(P_out_phys, phase_in, h)
        G = self._gain_linear(h)
        field_noisy, noise = self._add_ase_noise(field_clean, G)
        return field_clean, field_noisy, noise

    # ================================================================
    # Steady-state block
    # ================================================================

    def _steady_state_block(self, input_field: torch.Tensor) -> SoaSSResult:
        """
        Compute steady-state output for input_field [..., F] or [..., B, F].
        """
        phase = torch.angle(input_field)
        P_in = self._input_power(input_field)
        h_ss = self._solve_gain_steady_state(P_in)
        G_ss = self._gain_linear(h_ss)
        P_out_phys_ss = P_in * G_ss
        field_clean_ss, output_field_ss, noise_ss = self._noisy_output_field(
            P_out_phys_ss, phase, h_ss
        )
        return SoaSSResult(
            phase=phase,
            P_in=P_in,
            h_ss=h_ss,
            G_ss=G_ss,
            P_out_phys_ss=P_out_phys_ss,
            field_clean_ss=field_clean_ss,
            output_field_ss=output_field_ss,
            noise_ss=noise_ss,
        )

    # ================================================================
    # Helpers
    # ================================================================

    def _view_BT_from_FT(
        self,
        field_ft: torch.Tensor,
        B: int,
        n_time: int,
    ) -> torch.Tensor:
        """
        Helper for debugging: view [F, B*T] as [B, F, T].
        """
        field_f_bt = unflatten_tail(field_ft, (B, n_time))  # [F, B, T]
        return field_f_bt.permute(1, 0, 2)

    def _expand_F_to_FT(self, input_field: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Expand a steady-state input by appending a time axis of length samples_per_period.
        """
        if not hasattr(self._params, "clk_freq"):
            raise AttributeError("Time-domain simulation requested but params.clk_freq is not defined")
        if not hasattr(self._params, "dt"):
            raise AttributeError("Time-domain simulation requested but params.dt is not defined")
        return expand_ss_to_time(
            input_field,
            clk_freq=self._params.clk_freq,
            dt=dt,
        )

    def _prepare_h0(
        self,
        h_init: float | torch.Tensor,
        P_in_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize h_init (scalar or tensor) to spatial shape P_in_full[..., 0].shape.
        """
        h_spatial_shape = P_in_full[..., 0].shape
        if torch.is_tensor(h_init):
            if h_init.shape == h_spatial_shape:
                return self._clamp_h(h_init.detach().to(P_in_full[..., 0]))
            mean_val = float(h_init.detach().mean().item())
        else:
            mean_val = float(h_init)

        return self._clamp_h(
            torch.full(
                h_spatial_shape,
                mean_val,
                device=P_in_full.device,
                dtype=P_in_full.dtype,
            )
        )

    def _collapse_td_center(
        self,
        field_det_td: torch.Tensor,
        field_td: torch.Tensor,
        noise_td: torch.Tensor,
        dt: float,
        clk_freq: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collapse at period centers along last time axis using collapse_last_dim.
        """
        clean_coll = collapse_last_dim(
            field_det_td, mode="per_period_center", dt=dt, clk_freq=clk_freq
        )
        noisy_coll = collapse_last_dim(
            field_td, mode="per_period_center", dt=dt, clk_freq=clk_freq
        )
        noise_coll = collapse_last_dim(
            noise_td, mode="per_period_center", dt=dt, clk_freq=clk_freq
        )
        return clean_coll, noisy_coll, noise_coll

    # ================================================================
    # Unified solve
    # ================================================================

    def solve_fields(
        self,
        input_field: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[SoaSSResult]]:
        """
        Returns (clean_field, noisy_field, noise, ss_result_or_None).

        Shapes:
          - SS only: [..., F]
          - TD, collapse_to_ss=False:
              * Mode A/C: [..., F, T]
              * Mode B:   [B*T, F]
          - TD, collapse_to_ss=True:
              * Mode A/C: [..., F, n_periods]
              * Mode B:   [B, F]
        """
        if not torch.is_complex(input_field):
            raise TypeError("input_field must be a complex tensor (complex64 or complex128)")
        if input_field.dtype not in (torch.complex64, torch.complex128):
            raise TypeError(f"Unsupported complex dtype {input_field.dtype}")

        ss = self._steady_state_block(input_field)

        collapse_to_ss = self._collapse_to_ss()

        # Pure SS mode
        if not self._is_time_domain():
            return ss.field_clean_ss, ss.output_field_ss, ss.noise_ss, ss

        if not hasattr(self._params, "dt"):
            raise AttributeError("Time-domain simulation requested but params.dt is not defined")
        dt = self._params.dt
        clk_freq = self._params.clk_freq

        # Get full TD traces for the selected mode
        if not self._is_input_ss():
            # Mode A: explicit TD [...,F,T]
            field_det_td, field_td, noise_td = self._solve_td_explicit_time(
                input_field, ss, dt
            )
        elif self._batch_as_time():
            # Mode B: SS + batch-as-time
            # TD output:
            #   - collapse_to_ss=False: [B*T, F] (after transpose)
            #   - collapse_to_ss=True:  [B, F]   (one value per batch)
            field_det_td, field_td, noise_td = self._solve_td_ss_batch_as_time(
                input_field, ss, dt
            )  # [F, B*T]

            if not collapse_to_ss:
                # [F, B*T] -> [B*T, F]
                return (
                    field_det_td.transpose(0, 1),
                    field_td.transpose(0, 1),
                    noise_td.transpose(0, 1),
                    ss,
                )

            # Collapse along time, then map periods back to batches
            clean_coll, noisy_coll, noise_coll = self._collapse_td_center(
                field_det_td, field_td, noise_td, dt, clk_freq
            )  # [F, n_periods]

            n_periods = clean_coll.shape[-1]
            B = input_field.shape[0]
            if n_periods < B:
                raise ValueError(
                    f"Mode B collapse: n_periods={n_periods} < B={B}"
                )

            clean_coll = clean_coll[:, :B].transpose(0, 1)  # [B, F]
            noisy_coll = noisy_coll[:, :B].transpose(0, 1)
            noise_coll = noise_coll[:, :B].transpose(0, 1)
            return clean_coll, noisy_coll, noise_coll, ss

        else:
            # Mode C: SS per batch [...,F,T]
            field_det_td, field_td, noise_td = self._solve_td_ss_per_batch(
                input_field, ss, dt
            )

        # No collapse (Modes A/C): just return TD
        if not collapse_to_ss:
            return field_det_td, field_td, noise_td, ss

        # Collapse at period centers (Modes A/C)
        clean_coll, noisy_coll, noise_coll = self._collapse_td_center(
            field_det_td, field_td, noise_td, dt, clk_freq
        )

        # Special case: SS input with no explicit batch (Mode C, input [F])
        # should return [F], matching the steady-state API and tests.
        if self._is_input_ss() and (not self._batch_as_time()) and input_field.ndim == 1:
            clean_coll = clean_coll[..., -1]      # [F]
            noisy_coll = noisy_coll[..., -1]
            noise_coll = noise_coll[..., -1]

        return clean_coll, noisy_coll, noise_coll, ss

    # ================================================================
    # TD strategies (always return full TD)
    # ================================================================

    def _solve_td_explicit_time(
        self,
        input_field: torch.Tensor,
        ss: SoaSSResult,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mode A: is_input_ss == False.

        input_field: [..., F, T] (time is last axis).
        """
        field_td, field_det_td, noise_td = self.simulate(
            input_field, dt=dt, h_init=ss.h_ss
        )
        return field_det_td, field_td, noise_td

    def _solve_td_ss_per_batch(
        self,
        input_field: torch.Tensor,
        ss: SoaSSResult,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mode C: is_input_ss == True and batch_as_time == False.

        input_field: [..., F]
        """
        field_td_in = self._expand_F_to_FT(input_field, dt)  # [..., F, T]

        field_td_out, field_det_td_out, noise_td_out = self.simulate(
            field_td_in, dt=dt, h_init=ss.h_ss
        )
        return field_det_td_out, field_td_out, noise_td_out

    def _solve_td_ss_batch_as_time(
        self,
        input_field: torch.Tensor,
        ss: SoaSSResult,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mode B: is_input_ss == True and batch_as_time == True.

        Internal TD output: [F, B*T] (long time axis).
        """
        if input_field.ndim != 2:
            raise ValueError(
                "batch_as_time=True expects input_field of shape [B, F]"
            )

        B, F = input_field.shape
        field_bt = self._expand_F_to_FT(input_field, dt)  # [B, F, T]
        _, _, n_time = field_bt.shape

        # [B, F, T] -> [F, B*T] (long time axis)
        field_ft = flatten_tail(field_bt.permute(1, 0, 2), 2)  # [F, B*T]

        h_init_channels = ss.h_ss.mean(dim=0)  # [F]
        h_init = h_init_channels.unsqueeze(0)  # [1, F]

        field_td_out, field_clean_td_out, noise_td_out = self.simulate(
            field_ft, dt=dt, h_init=h_init
        )

        # Always return full TD along long time axis [F,B*T]
        return field_clean_td_out, field_td_out, noise_td_out

    # ================================================================
    # Forward
    # ================================================================

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Main module forward.
        """
        clean, noisy, noise, ss = self.solve_fields(input_field)

        self._last_clean = clean
        self._last_noisy = noisy
        self._last_noise = noise

        self._debug_log_forward(input_field, clean, noisy, noise, ss)

        if (not self._is_time_domain()) or self._collapse_to_ss():
            return noisy

        if self._is_input_ss() and (not self._batch_as_time()):
            noisy_ss = ss.output_field_ss  # [..., F]
            noisy_ste = _SoaSTETimeDomain.apply(noisy, noisy_ss)
            self._last_noisy = noisy_ste
            return noisy_ste

        return noisy

    # ================================================================
    # Time-domain simulation (per-channel RK4)
    # ================================================================

    def simulate(
        self,
        input_field: torch.Tensor,
        dt: float,
        h_init: float | torch.Tensor = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run time-domain RK4 simulation.

        input_field: complex [..., F, T], T>=2.
        """
        if not torch.is_complex(input_field):
            raise TypeError("input_field must be a complex tensor")

        T = input_field.shape[-1]
        if T < 2:
            raise ValueError("input_field must have at least 2 time samples")

        dt_t = torch.as_tensor(
            dt,
            dtype=input_field.real.dtype,
            device=input_field.device,
        )

        P_in_full = self._input_power(input_field)  # [..., F, T]
        h0 = self._prepare_h0(h_init, P_in_full)

        h = torch.empty_like(P_in_full)
        h[..., 0] = h0

        field_out = torch.empty_like(input_field)
        field_clean_out = torch.empty_like(input_field)
        noise_out = torch.empty_like(input_field)

        # Initial phase: only first time sample needed
        phase_state = torch.angle(input_field[..., 0])

        P_in0 = P_in_full[..., 0]
        G0 = self._gain_linear(h0)
        P_det0 = P_in0 * G0

        phase0 = phase_state
        while phase0.ndim < P_det0.ndim:
            phase0 = phase0.unsqueeze(-1)

        field_clean0, field0_noisy, noise0 = self._noisy_output_field(
            P_det0, phase0, h0
        )
        field_clean_out[..., 0] = field_clean0
        field_out[..., 0] = field0_noisy
        noise_out[..., 0] = noise0

        for i in range(T - 1):
            h_i = h[..., i]
            P_i = P_in_full[..., i]
            P_i1 = P_in_full[..., i + 1]

            k1 = self._dh_dt(h_i, P_i)
            P_half = 0.5 * (P_i + P_i1)
            k2 = self._dh_dt(h_i + 0.5 * dt_t * k1, P_half)
            k3 = self._dh_dt(h_i + 0.5 * dt_t * k2, P_half)
            k4 = self._dh_dt(h_i + dt_t * k3, P_i1)

            h_next = self._clamp_h(h_i + (dt_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
            h[..., i + 1] = h_next.detach()

            G_next = self._gain_linear(h_next)
            P_det = P_i1 * G_next

            phase = phase_state
            while phase.ndim < P_det.ndim:
                phase = phase.unsqueeze(-1)

            field_clean, field_noisy, noise = self._noisy_output_field(
                P_det, phase, h_next
            )
            field_clean_out[..., i + 1] = field_clean
            field_out[..., i + 1] = field_noisy
            noise_out[..., i + 1] = noise

        return field_out, field_clean_out, noise_out

    def set_params(self, soa_params: SoaParams) -> None:
        """
        Update SOA parameters and recompute derived quantities.

        This is used at test time to inject non-ideal hardware configs
        (e.g. enable ASE noise, change gain, etc.).
        """
        from copy import deepcopy
        self._params = deepcopy(soa_params)

        # Recompute derived quantities that depend on params
        self._h0 = self._params.g0 * self._params.L
        c = 3e8
        self._hnu = 6.62607015e-34 * c / self._params.lam0
        self._inv_P_sat = 1.0 / self._params.P_sat
        self._inv_tau_c = 1.0 / self._params.tau_c

    def get_params(self) -> SoaParams:
        """Return the current SOA parameters."""
        return self._params
