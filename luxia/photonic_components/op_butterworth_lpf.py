from copy import deepcopy
import torch
from luxia import LxModule

from luxia.params.butterworth_lpf_params import ButterworthLPFParams
from luxia.util.util_functions import flatten_tail, unflatten_tail
from luxia.util.util_time_domain_functions import (
    expand_ss_to_time,
    collapse_last_dim,
)
from luxia.util.log_config import lx_log


class _LPFSTE(torch.autograd.Function):
    r"""
    Straight-through estimator for the LPF.
    """

    @staticmethod
    def forward(ctx, E_out: torch.Tensor) -> torch.Tensor:
        return E_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class OpButterworthLPF(LxModule):
    r"""
    Frequency-domain Butterworth low-pass filter on complex envelopes.

    Modes (mirrors SOA + an extra D):

    - Mode A: use_time_domain=True, is_input_ss=False, batch_as_time=False
    - Mode B: use_time_domain=True, is_input_ss=True, batch_as_time=True
    - Mode C: use_time_domain=True, is_input_ss=True, batch_as_time=False
    - Mode D: use_time_domain=True, is_input_ss=False, batch_as_time=True
    """

    def __init__(self, params: ButterworthLPFParams, debug: bool = False) -> None:
        super().__init__(debug)
        self._p = deepcopy(params)

        # --- cutoff freq (learnable, log-space) ---
        cutoff_freq = float(self._p.cutoff_freq)
        if cutoff_freq <= 0.0:
            raise ValueError("cutoff_freq must be > 0.")

        cutoff_init = max(cutoff_freq, 1e-9)
        log_cutoff_init = torch.log(torch.tensor(cutoff_init))

        if self._is_trainable_cutoff:
            self.log_cutoff_freq = torch.nn.Parameter(log_cutoff_init)
        else:
            self.register_buffer("log_cutoff_freq", log_cutoff_init)

    # -------------------------------------------------------------
    # Flags (mirror SOA naming)
    # -------------------------------------------------------------

    @property
    def _is_time_domain(self) -> bool:
        return bool(self._p.use_time_domain)

    @property
    def _is_input_ss(self) -> bool:
        return bool(getattr(self._p, "is_input_ss", False))

    @property
    def _batch_as_time(self) -> bool:
        return bool(getattr(self._p, "batch_as_time", False))

    @property
    def _collapse_to_ss(self) -> bool:
        return bool(getattr(self._p, "collapse_to_ss", False))

    @property
    def _is_trainable_cutoff(self) -> bool:
        return bool(getattr(self._p, "trainable_cutoff", False))

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _power(self, E: torch.Tensor) -> torch.Tensor:
        return E.abs().square()

    def _expand_F_to_FT(self, E_in: torch.Tensor, dt: float) -> torch.Tensor:
        if not hasattr(self._p, "clk_freq"):
            raise AttributeError(
                "Time-domain LPF requested but params.clk_freq is not defined"
            )
        return expand_ss_to_time(E_in, clk_freq=self._p.clk_freq, dt=dt)

    # -------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------
    def _debug_log_forward(
        self,
        E_in: torch.Tensor,
        E_out: torch.Tensor,
        E_td: torch.Tensor | None,
    ) -> None:
        """
        Minimal debug logger for LPF forward().

        Keeps logging lightweight and avoids torch print-option APIs,
        which are not consistently available across torch versions.
        """
        if not self._debug:
            return

        try:
            msg = (
                "[LPF] forward: "
                f"in={tuple(E_in.shape)} "
                f"out={tuple(E_out.shape)} "
                f"td={tuple(E_td.shape) if E_td is not None else None} "
                f"time_domain={self._is_time_domain} "
                f"is_input_ss={self._is_input_ss} "
                f"batch_as_time={self._batch_as_time} "
                f"collapse_to_ss={self._collapse_to_ss} "
                f"is_trainable_cutoff={self._is_trainable_cutoff}"
            )

            if self._is_trainable_cutoff:
                cutoff_hz = self._cutoff_freq()
                msg += (
                    f" trainable_cutoff_value_Hz={cutoff_hz.item():.6e}"
                    f" trainable_cutoff_value_GHz={cutoff_hz.item() / 1e9:.6f}"

                )
            lx_log.debug(msg)
            if E_out.numel() > 0:
                P_out = self._power(E_out.reshape(-1))
                n_show = min(4, P_out.numel())
                out_vals = [float(v.item()) for v in P_out[:n_show]]
                lx_log.debug(f"[LPF] output sample powers: {out_vals}")
            else:
                lx_log.debug("[LPF] output sample powers: None")

            if E_td is not None and E_td.numel() > 0:
                P_td = self._power(E_td.reshape(-1))
                n_show = min(4, P_td.numel())
                td_vals = [float(v.item()) for v in P_td[:n_show]]
                lx_log.debug(f"[LPF] td sample powers: {td_vals}")
            else:
                lx_log.debug("[LPF] td sample powers: None")

        except Exception as e:
            lx_log.debug(f"[LPF] debug logging skipped: {e}")

    # -------------------------------------------------------------
    # Core filter (always last dim)
    # -------------------------------------------------------------
    def _cutoff_freq(self) -> torch.Tensor:
        # returns scalar tensor, always > 0
        return self.log_cutoff_freq.exp()

    def _butterworth_H(self, freqs: torch.Tensor) -> torch.Tensor:
        cutoff = self._cutoff_freq()
        norm = freqs.abs() / cutoff
        x = norm.pow(2 * self._p.order)
        return 1.0 / torch.sqrt(1.0 + x)

    def H_freq_response(
        self,
        L: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if L < 1:
            raise ValueError(f"L must be >= 1, got {L!r}")
        freqs = torch.fft.fftfreq(L, d=self._p.dt, device=device, dtype=dtype).reshape(
            *([1] * 2), L
        )
        return self._butterworth_H(freqs)

    def _filter_time_axis(self, E_in: torch.Tensor) -> torch.Tensor:
        """
        FFT-based LPF along the last axis (time).
        E_in is complex, last axis is time.
        """
        if not torch.is_complex(E_in):
            raise TypeError("E_in must be a complex tensor")

        T = E_in.shape[-1]
        if T < 1:
            raise ValueError("Last axis (time) must have at least 1 sample")

        E_fft = torch.fft.fft(E_in, dim=-1)

        freqs = torch.fft.fftfreq(
            T,
            d=self._p.dt,
            device=E_in.device,
            dtype=E_in.real.dtype,
        ).view(*([1] * (E_in.ndim - 1)), T)

        H = self._butterworth_H(freqs)
        if H.shape != E_fft.shape:
            H = H.expand_as(E_fft)

        E_fft_filt = E_fft * H
        return torch.fft.ifft(E_fft_filt, dim=-1)

    # -------------------------------------------------------------
    # TD sub-solvers (mirror SOA structure)
    # -------------------------------------------------------------

    def _solve_td_explicit_time(
        self,
        E_in: torch.Tensor,
        dt: float,
        clk_freq: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mode A: is_input_ss == False and batch_as_time == False.
        E_in: [..., F, T] or generic [..., T] (time last).
        Returns (E_out_or_ss, E_td_for_debug).
        """
        E_td = self._filter_time_axis(E_in)
        if not self._collapse_to_ss:
            return E_td, E_td

        # Collapse along last (time) axis to period centers
        E_coll = collapse_last_dim(
            E_td, mode="per_period_center", dt=dt, clk_freq=clk_freq
        )  # [..., n_periods]

        # 1D input [T] -> always return [n_periods]
        if E_in.ndim == 1:
            E_out = E_coll  # [n_periods]
            return E_out, E_td

        # Higher‑D input [...,F,T] -> [...,F,n_periods] or [...,F]
        n_periods = E_coll.shape[-1]
        if n_periods == 1:
            E_out = E_coll[..., 0]   # squeeze to [...,F]
        else:
            E_out = E_coll           # keep [...,F,n_periods]

        return E_out, E_td

    def _solve_td_ss_per_batch(
        self,
        E_in: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mode C: is_input_ss == True and batch_as_time == False.
        E_in: [B,F] or [...,F].
        Returns (E_out_or_td, E_td_for_debug).
        """
        if E_in.ndim != 2:
            raise ValueError(f"LPF Mode C expects [B, F], got {E_in.shape}")

        E_td_in = self._expand_F_to_FT(E_in, dt)  # [B,F,T]
        E_td = self._filter_time_axis(E_td_in)    # [B,F,T]

        if not self._collapse_to_ss:
            return E_td, E_td

        # collapse to last time sample (mirrors Soa 'last' for SS)
        E_out = collapse_last_dim(E_td, mode="last")  # [B,F]
        return E_out, E_td

    def _solve_td_ss_batch_as_time(
        self,
        E_in: torch.Tensor,
        dt: float,
        clk_freq: float,
        n_time: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mode B: is_input_ss == True and batch_as_time == True.
        E_in: [B,F].
        Returns (E_out_or_td, E_td_for_debug) with:
        - collapse_to_ss=False: [B*T,F]
        - collapse_to_ss=True: [B,F]
        """
        if E_in.ndim != 2:
            raise ValueError(f"LPF Mode B expects [B, F], got {E_in.shape}")

        B, F = E_in.shape
        E_bft = self._expand_F_to_FT(E_in, dt)  # [B,F,T]
        _, _, T = E_bft.shape

        if T != n_time:
            # keep behavior consistent with original: just use T as it is
            n_time = T

        # batch-as-time: [B,F,T] -> [F,B,T] -> [F,B*T]
        E_ft = flatten_tail(E_bft.permute(1, 0, 2), 2)  # [F,B*T]
        E_ft_filt = self._filter_time_axis(E_ft)        # [F,B*T]

        if not self._collapse_to_ss:
            E_out = E_ft_filt.transpose(0, 1)  # [B*T,F]
            return E_out, E_ft_filt

        E_ft_coll = collapse_last_dim(
            E_ft_filt, mode="per_period_center", dt=dt, clk_freq=clk_freq
        )  # [F,n_periods]
        n_periods = E_ft_coll.shape[-1]
        if n_periods < B:
            raise ValueError(f"Mode B collapse: n_periods={n_periods} < B={B}")
        E_ft_coll = E_ft_coll[:, :B]        # [F,B]
        E_out = E_ft_coll.transpose(0, 1)   # [B,F]
        return E_out, E_ft_filt

    def _solve_td_batch_as_time_explicit(
            self,
            E_in: torch.Tensor,
            dt: float,
            clk_freq: float,
            n_time: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Mode D: is_input_ss == False and batch_as_time == True.
            E_in: [B_big,F] with B_big multiple of n_time.
            Returns (E_out_or_td, E_td_for_debug).
            """
            if E_in.ndim != 2:
                raise ValueError(f"LPF Mode D expects [B_big, F], got {E_in.shape}")

            B_big, F = E_in.shape
            if B_big % n_time != 0:
                raise ValueError(f"B_big={B_big} not multiple of n_time={n_time}")
            B = B_big // n_time

            # Channels-first for per-channel time filter
            E_ft = E_in.transpose(0, 1)            # [F,B_big]
            E_ft_filt = self._filter_time_axis(E_ft)  # [F,B_big]

            if not self._collapse_to_ss:
                E_out = E_ft_filt.transpose(0, 1)  # [B_big,F]
                return E_out, E_ft_filt

            # Collapse per-period centers along the long time axis
            E_coll_ft = collapse_last_dim(
                E_ft_filt, mode="per_period_center", dt=dt, clk_freq=clk_freq
            )  # [F,n_periods]

            n_periods = E_coll_ft.shape[-1]
            if n_periods < B:
                raise ValueError(f"Mode D collapse: n_periods={n_periods} < B={B}")

            E_coll_ft = E_coll_ft[:, :B]      # [F,B]
            E_coll = E_coll_ft.transpose(0, 1)  # [B,F]

            return E_coll, E_ft_filt
    # -------------------------------------------------------------
    # Unified solve (modes A, B, C, D)
    # -------------------------------------------------------------

    def _solve_fields(
        self,
        E_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns (E_out, E_td_for_debug_or_None).

        Shapes (mirrors Soa):
        - SS-only: [B,F] or [B_big,F], no explicit T.
        - TD, collapse_to_ss=False:
          * Mode A/C: [...,F,T]
          * Mode B/D: [B*T,F] or [B_big,F]
        - TD, collapse_to_ss=True:
          * Mode A/C: [...,F,n_periods]
          * Mode B/D: [B,F]
        """
        if not self._is_time_domain:
            E_out = E_in
            return E_out, None

        if (not hasattr(self._p, "clk_freq")) or (not hasattr(self._p, "dt")):
            raise AttributeError(
                "Time-domain LPF requested but params.clk_freq/dt is not fully defined"
            )

        clk_freq = self._p.clk_freq
        dt = self._p.dt
        T_clk = 1.0 / clk_freq
        n_time = int(round(T_clk / dt))

        # Mode A: explicit time, arbitrary leading dims, last dim is time
        if (not self._is_input_ss) and (not self._batch_as_time):
            E_out, E_td = self._solve_td_explicit_time(E_in, dt, clk_freq)
            return E_out, E_td

        # Mode C: SS per batch [B,F]
        if self._is_input_ss and (not self._batch_as_time):
            E_out, E_td = self._solve_td_ss_per_batch(E_in, dt)
            return E_out, E_td

        # Mode B: SS batch-as-time [B,F]
        if self._is_input_ss and self._batch_as_time:
            E_out, E_td = self._solve_td_ss_batch_as_time(E_in, dt, clk_freq, n_time)
            return E_out, E_td

        # Mode D: explicit batch-as-time [B_big,F]
        if (not self._is_input_ss) and self._batch_as_time:
            E_out, E_td = self._solve_td_batch_as_time_explicit(
                E_in, dt, clk_freq, n_time
            )
            return E_out, E_td

        raise RuntimeError(
            "Unsupported LPF configuration: use_time_domain=True but "
            f"is_input_ss={self._is_input_ss}, batch_as_time={self._batch_as_time}"
        )

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------

    def forward(self, E_in: torch.Tensor) -> torch.Tensor:
        if not self._is_time_domain:
            E_out = E_in
            self._debug_log_forward(E_in, E_out, None)
            if getattr(self._p, "use_ste", False):
                E_out = _LPFSTE.apply(E_out)
            return E_out

        E_out, E_td = self._solve_fields(E_in)
        self._debug_log_forward(E_in, E_out, E_td)

        if getattr(self._p, "use_ste", False):
            E_out = _LPFSTE.apply(E_out)

        return E_out
