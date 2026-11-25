from dataclasses import dataclass


@dataclass
class ButterworthLPFParams:
    dt: float
    cutoff_freq: float = 25e9
    trainable_cutoff: bool = False
    order: int = 2
    use_time_domain: bool = False
    collapse_to_ss: bool = False
    clk_freq: float = 10e9
    # If True, LPF uses straight-through estimator for gradients:
    # forward = full filtered signal, backward = identity grad.
    use_ste: bool = False
    # If True, input is treated as steady-state [B, F] and a single
    # clock period is synthesized per feature (SS -> TD modes),
    # unless batch_as_time=True, in which case batch is treated as time.
    # If False, input is an explicit time sequence [..., T].
    is_input_ss: bool = False
    # If True and is_input_ss == True, treat batch dimension as time
    # (batch-as-time mode). Input [B,F] is interpreted as F channels
    # with B time samples.
    batch_as_time: bool = False

