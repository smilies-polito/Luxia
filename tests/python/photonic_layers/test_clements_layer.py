"""
Tests for ClementsLayerParams bindings and ClementsLayer (C++).
"""
import pickle

import pytest
import torch

from luxia_lib import (
    ClementsLayer,
    ClementsLayerParams, ClementsBlockParams, HeaterParams, MmiParams,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _block_params(n: int) -> ClementsBlockParams:
    hp = [HeaterParams(number_of_heaters=n), HeaterParams(number_of_heaters=n)]
    mmi = [MmiParams(number_of_splitters=n), MmiParams(number_of_splitters=n)]
    return ClementsBlockParams(number_of_blocks=n, heater_params=hp, mmi_params=mmi)


def _params(n: int = 4, even: bool = True) -> ClementsLayerParams:
    nb = n // 2 if even else (n - 1) // 2
    return ClementsLayerParams(
        number_of_inputs=n,
        is_even_layer=even,
        clements_blk_params=_block_params(nb),
    )


@pytest.fixture
def layer():
    return ClementsLayer(_params(4, True))


# ---------------------------------------------------------------------------
# ClementsLayerParams bindings (C++)
# ---------------------------------------------------------------------------

def test_clements_layer_params_is_nn_module():
    assert isinstance(ClementsLayerParams(), torch.nn.Module)


def test_clements_layer_params_default():
    p = ClementsLayerParams()
    assert isinstance(p, torch.nn.Module)


def test_clements_layer_params_full():
    p = _params(4, True)
    assert p.number_of_inputs == 4
    assert p.is_even_layer is True


def test_clements_layer_params_repr():
    assert len(repr(ClementsLayerParams())) > 0


def test_clements_layer_params_forward_raises():
    with pytest.raises(TypeError):
        ClementsLayerParams().forward(torch.zeros(2))


def test_clements_layer_params_pickle():
    p = _params(4, True)
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, ClementsLayerParams)


# ---------------------------------------------------------------------------
# ClementsLayer forward / backward
# ---------------------------------------------------------------------------

def test_clements_layer_construction():
    m = ClementsLayer(_params(4, True))
    assert isinstance(m, torch.nn.Module)


def test_clements_layer_forward_shape(layer):
    x = torch.rand(4, dtype=torch.complex128)
    out = layer.forward(x)
    assert out.shape == x.shape
    assert torch.is_complex(out)


def test_clements_layer_backward(layer):
    x = torch.rand(4, dtype=torch.complex128).requires_grad_(True)
    out = layer.forward(x)
    out.real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_clements_layer_power_conservation(layer):
    x = torch.rand(4, dtype=torch.complex128)
    out = layer.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-4)


def test_clements_layer_parameters_trainable(layer):
    assert len(list(layer.parameters())) > 0


def test_clements_layer_phase_shifts_roundtrip(layer):
    orig = layer.get_phase_shifts()
    new_phases = torch.zeros_like(orig)
    layer.set_phase_shifts(new_phases)
    assert torch.allclose(layer.get_phase_shifts(), new_phases, atol=1e-6)


# ---------------------------------------------------------------------------
# Various configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,even", [(4, True), (4, False), (6, True), (6, False)])
def test_clements_layer_various_configs(n, even):
    m = ClementsLayer(_params(n, even))
    x = torch.rand(n, dtype=torch.complex128)
    out = m.forward(x)
    assert out.shape == (n,)


# ---------------------------------------------------------------------------
# torch.nn.Module API
# ---------------------------------------------------------------------------

def test_clements_layer_nn_api(layer):
    layer.train()
    layer.eval()
    assert len(list(layer.parameters())) > 0
