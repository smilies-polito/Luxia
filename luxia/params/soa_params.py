from dataclasses import dataclass


@dataclass
class SoaParams:
    """
    Semiconductor Optical Amplifier (SOA) model parameters.

    Attributes
    ----------
    alpha:
        Linewidth enhancement (Henry) factor, linking carrier-induced gain
        changes to refractive index changes and thus phase/amplitude coupling.
    g0:
        Small-signal material gain coefficient (per unit length), setting the
        unsaturated gain available in the active region.
    L:
        Active device length (m) over which optical power experiences
        amplification inside the SOA.
    P_sat:
        Saturation output power (W) at which the gain drops by roughly 3 dB
        because of carrier depletion (onset of gain saturation).
    P_ref:
        Reference optical power (W) used for internal normalization or
        scaling of powers in the model (often chosen as 1 mW).
    norm_input:
        If True, input optical power is normalized (e.g. by P_ref) before
        applying the SOA gain / saturation equations.
    norm_output:
        If True, the model renormalizes the output back to physical units
        after internal normalized computations.
    newton_iters:
        Maximum number of Newton–Raphson iterations used by the solver to
        find the steady-state carrier density / gain.
    newton_tol:
        Convergence tolerance for the Newton solver; iteration stops when
        successive updates fall below this threshold.
    use_ase:
        If True, include Amplified Spontaneous Emission (ASE) noise
        contribution in the SOA output.
    n_sp:
        Spontaneous emission factor n_sp, scaling ASE power above the
        quantum-limited spontaneous emission level.
    B0:
        Optical noise bandwidth (Hz) over which ASE is integrated, usually
        tied to receiver / filter bandwidth.
    tau_c:
        Carrier lifetime (s), controlling dynamic response, i.e. how quickly
        gain recovers and how the SOA behaves under high-speed modulation.
    lam0:
        Central optical wavelength (m) at which the photon energy hν is
        computed (e.g. 1550e-9 for 1550 nm).

    Unified steady-state / time-domain behavior
    -------------------------------------------
    use_time_domain:
        If False (default), the SOA forward method computes a purely
        steady-state response, mapping an input field of shape [..., F] to an
        output field of the same shape [..., F].

        If True, the forward method performs a time-domain simulation whose
        behavior depends on is_input_ss, batch_as_time, and collapse_to_ss:

        * is_input_ss == True and batch_as_time == False:
            - Input to forward(): complex tensor of shape [..., F], where F is
              a set of independent “features” (e.g. pixels, channels).
            - Internally, each feature is held constant over one clock period
              T_clk = 1 / clk_freq, and the integrator uses dt to determine
              the number of time steps n_time ≈ T_clk / dt.
            - If collapse_to_ss is False, forward() returns a time-resolved
              output of shape [..., F, T], where T = n_time.
            - If collapse_to_ss is True, forward() collapses the time axis back
              to a steady-state-like output of shape [..., F] by taking the
              last time sample for each feature.

        * is_input_ss == True and batch_as_time == True:
            - Input to forward(): complex tensor of shape [B, F] (batch of B
              images / samples, each with F features).
            - Each of the B inputs is held constant over one period, and these
              B periods are concatenated into one long time sequence of length
              B * n_time per feature, so the carrier h evolves *across* images.
            - If collapse_to_ss is False, forward() returns [B, F, T], a view
              of that long sequence split back into B images with T samples
              each.
            - If collapse_to_ss is True, forward() collapses to [B, F] by
              taking the last time sample of each image/feature.

        * is_input_ss == False:
            - Input to forward(): complex tensor of shape [..., F, T], where
              the last axis T is a time dimension of consecutive samples for
              each spatial / frequency channel.
            - The simulator integrates h(t) over these T samples using dt.
            - If collapse_to_ss is False, forward() returns the full
              time-resolved output of shape [..., F, T].
            - If collapse_to_ss is True, forward() collapses the time dimension
              to a steady-state-like output, typically by picking one
              representative sample per clock period (e.g. the last sample at
              each period end). The resulting shape is [..., F, N_periods].

    dt:
        Time step (s) used for the internal time-domain integrator when
        use_time_domain is True. Must satisfy dt << 1/clk_freq for stable
        integration and at least 2 samples per period.
    clk_freq:
        Clock frequency (Hz). When use_time_domain is True, this defines
        the clock period T_clk = 1 / clk_freq, which is used to decide how
        many time samples belong to one period and, when collapse_to_ss is
        True, how to pick representative samples.
    is_input_ss:
        If True (default), the last axis of the input to forward() is treated
        as a collection of independent steady-state operating points (features
        / pixels) with no explicit time dependence.
        If False, the last axis is interpreted as a time dimension of a
        sequence for each spatial / frequency channel.
    collapse_to_ss:
        If True and use_time_domain is True, the time-domain output is
        collapsed back to a steady-state-like tensor by taking one value per
        clock period (e.g. the last time step of each period). The resulting
        shape is [..., F] (SS input) or [..., F, N_periods] (TD input).
    batch_as_time:
        If True and use_time_domain=True and is_input_ss=True, a batch
        dimension [B, F] is treated as a sequence of B periods for each
        feature, so the carrier dynamics evolve across images/samples in
        the batch instead of treating each batch element independently.
    """
    alpha: float = 5.0
    g0: float = 6075.0
    L: float = 500e-6
    P_sat: float = 0.0211
    P_ref: float = 1e-3
    norm_input: bool = False
    norm_output: bool = False

    newton_iters: int = 8
    newton_tol: float = 1e-5

    use_ase: bool = False
    n_sp: float = 2.0
    B0: float = 100e9

    tau_c: float = 300e-12
    lam0: float = 1550e-9

    # Unified steady-state / time-domain behavior
    use_time_domain: bool = False
    dt: float = 1e-12
    clk_freq: float = 10e9
    is_input_ss: bool = True
    collapse_to_ss: bool = False
    batch_as_time: bool = False
