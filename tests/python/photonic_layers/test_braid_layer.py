"""
Tests for BraidLayerParams bindings and BraidLayer (C++).

BraidLayer is passive — it has no heaters/phase-shifters, so no trainable parameters.
"""
import math
import pickle

import pytest
import torch

from luxia_lib import (
    BraidLayer,
    BraidLayerParams, BraidBlockParams, CrossParams,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _block_params(n: int) -> BraidBlockParams:
    return BraidBlockParams(
        number_of_blocks=n,
        cross_params=CrossParams(number_of_splitters=n),
    )


def _params(n: int = 4) -> BraidLayerParams:
    nb = (n - 1) // 2
    return BraidLayerParams(
        number_of_inputs=n,
        is_even_layer=False,
        braid_block_params=_block_params(nb),
    )


@pytest.fixture
def layer():
    return BraidLayer(_params(4))


# ---------------------------------------------------------------------------
# BraidLayerParams bindings (C++)
# ---------------------------------------------------------------------------

def test_braid_layer_params_is_nn_module():
    assert isinstance(BraidLayerParams(), torch.nn.Module)


def test_braid_layer_params_default():
    p = BraidLayerParams()
    assert isinstance(p, torch.nn.Module)


def test_braid_layer_params_full():
    p = _params(4)
    assert p.number_of_inputs == 4
    assert p.is_even_layer is False


def test_braid_layer_params_repr():
    assert len(repr(BraidLayerParams())) > 0


def test_braid_layer_params_forward_raises():
    with pytest.raises(TypeError):
        BraidLayerParams().forward(torch.zeros(2))


def test_braid_layer_params_pickle():
    p = BraidLayerParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, BraidLayerParams)


# ---------------------------------------------------------------------------
# BraidLayer forward / backward
# ---------------------------------------------------------------------------

def test_braid_layer_construction():
    m = BraidLayer(_params(4))
    assert isinstance(m, torch.nn.Module)


def test_braid_layer_forward_shape(layer):
    x = torch.rand(4, dtype=torch.complex128)
    out = layer.forward(x)
    assert out.shape == x.shape
    assert torch.is_complex(out)


def test_braid_layer_backward(layer):
    x = torch.rand(4, dtype=torch.complex128).requires_grad_(True)
    out = layer.forward(x)
    out.real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_braid_layer_power_conservation(layer):
    x = torch.rand(4, dtype=torch.complex128)
    out = layer.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-4)


def test_braid_layer_is_passive(layer):
    """BraidLayer has no heaters — no trainable parameters."""
    assert len(list(layer.parameters())) == 0


# ---------------------------------------------------------------------------
# Various sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_braid_layer_various_sizes(n):
    m = BraidLayer(_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = m.forward(x)
    assert out.shape == (n,)


# ---------------------------------------------------------------------------
# torch.nn.Module API
# ---------------------------------------------------------------------------

def test_braid_layer_nn_api(layer):
    layer.train()
    layer.eval()


# ---------------------------------------------------------------------------
# Half-cross boundary waveguide behavior
# ---------------------------------------------------------------------------

_IL_DB = 3.0    # 3 dB insertion loss
_CT_DB = -20.0  # -20 dB crosstalk (small but non-zero)


def _params_with_half_cross(n: int,
                             il_dB: float = _IL_DB,
                             ct_dB: float = _CT_DB) -> BraidLayerParams:
    hcp = CrossParams(
        number_of_splitters=1,
        insertion_loss=torch.tensor([il_dB]),
        crosstalk=torch.tensor([ct_dB]),
    )
    nb = (n - 1) // 2
    return BraidLayerParams(
        number_of_inputs=n,
        is_even_layer=False,
        braid_block_params=_block_params(nb),
        half_cross_params=hcp,
    )


def _expected_half_cross_coeff(il_dB: float, ct_dB: float) -> float:
    """cross_plus * loss_factor = sqrt(alpha) * 10^(-il_dB/20)."""
    ct_linear = 10 ** (ct_dB / 10)
    alpha = ct_linear / (1 + ct_linear)
    return math.sqrt(alpha) * 10 ** (-il_dB / 20)


def test_braid_layer_half_cross_params_exposed():
    p = _params_with_half_cross(4)
    assert p.half_cross_params is not None


def test_braid_layer_half_cross_top_boundary_coefficient():
    """output[0] = cross_plus * loss_factor * input[0] when only the top
    boundary waveguide is active."""
    layer = BraidLayer(_params_with_half_cross(4))
    expected = _expected_half_cross_coeff(_IL_DB, _CT_DB)

    x = torch.zeros(4, dtype=torch.complex128)
    x[0] = 1.0
    out = layer.forward(x)

    assert torch.allclose(out[0].abs(),
                          torch.tensor(expected, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(out[1:].abs().sum(),
                          torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_braid_layer_half_cross_bottom_boundary_coefficient():
    """output[-1] = cross_plus * loss_factor * input[-1] when only the bottom
    boundary waveguide is active (even-input layer, so bottom edge exists)."""
    layer = BraidLayer(_params_with_half_cross(4))
    expected = _expected_half_cross_coeff(_IL_DB, _CT_DB)

    x = torch.zeros(4, dtype=torch.complex128)
    x[-1] = 1.0
    out = layer.forward(x)

    assert torch.allclose(out[-1].abs(),
                          torch.tensor(expected, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(out[:-1].abs().sum(),
                          torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_braid_layer_half_cross_attenuates_boundary():
    """With insertion loss the boundary power is reduced; without half_cross
    the boundary is identity (power preserved)."""
    layer_hc = BraidLayer(_params_with_half_cross(4))
    layer_id = BraidLayer(_params(4))

    x = torch.zeros(4, dtype=torch.complex128)
    x[0] = 1.0

    out_hc = layer_hc.forward(x)
    out_id = layer_id.forward(x)

    # Default (no half_cross) passes the boundary straight through.
    assert torch.allclose(out_id[0].abs(),
                          torch.tensor(1.0, dtype=torch.float64), atol=1e-10)
    # Half-cross attenuates it.
    assert out_hc[0].abs() < out_id[0].abs()


def test_braid_layer_half_cross_internal_unaffected():
    """half_cross_params only touches the boundary waveguides; the internal
    Cross block output is identical with and without it."""
    layer_hc = BraidLayer(_params_with_half_cross(4))
    layer_id = BraidLayer(_params(4))

    # Activate only the two middle waveguides that feed the internal Cross.
    x = torch.zeros(4, dtype=torch.complex128)
    x[1] = 1.0
    x[2] = 1.0

    out_hc = layer_hc.forward(x)
    out_id = layer_id.forward(x)

    assert torch.allclose(out_hc[1:3], out_id[1:3], atol=1e-10)
    # Boundaries were zero in → zero out regardless of half_cross coefficient.
    assert torch.allclose(out_hc[0].abs(),
                          torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(out_hc[-1].abs(),
                          torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


def test_braid_layer_half_cross_backward():
    """Gradient flows through the half-cross boundary path."""
    layer = BraidLayer(_params_with_half_cross(4))

    x = torch.rand(4, dtype=torch.complex128).requires_grad_(True)
    layer.forward(x).real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
