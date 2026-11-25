"""
Tests for FldzhyanBlockParams bindings and FldzhyanBlock (C++).
"""
import pickle

import pytest
import torch

from luxia_lib import (
    FldzhyanBlock,
    FldzhyanBlockParams, HeaterParams, MmiParams,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _params(n: int = 4) -> FldzhyanBlockParams:
    return FldzhyanBlockParams(
        number_of_blocks=n,
        heater_params=HeaterParams(number_of_heaters=n),
        mmi_params=MmiParams(number_of_splitters=n),
    )


@pytest.fixture
def block():
    return FldzhyanBlock(_params(4))


# ---------------------------------------------------------------------------
# FldzhyanBlockParams bindings (C++)
# ---------------------------------------------------------------------------

def test_fldzhyan_block_params_is_nn_module():
    assert isinstance(FldzhyanBlockParams(), torch.nn.Module)


def test_fldzhyan_block_params_default():
    p = FldzhyanBlockParams()
    assert p.number_of_blocks >= 1


def test_fldzhyan_block_params_full():
    p = _params(4)
    assert p.number_of_blocks == 4


def test_fldzhyan_block_params_repr():
    assert len(repr(FldzhyanBlockParams())) > 0


def test_fldzhyan_block_params_forward_raises():
    with pytest.raises(TypeError):
        FldzhyanBlockParams().forward(torch.zeros(2))


def test_fldzhyan_block_params_pickle():
    p = FldzhyanBlockParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, FldzhyanBlockParams)


# ---------------------------------------------------------------------------
# FldzhyanBlock forward / backward
# ---------------------------------------------------------------------------

def test_fldzhyan_block_construction():
    m = FldzhyanBlock(_params(4))
    assert isinstance(m, torch.nn.Module)


def test_fldzhyan_block_forward_shape(block):
    portA = torch.rand(4, dtype=torch.complex128)
    portB = torch.rand(4, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    assert torch.is_complex(outA) and torch.is_complex(outB)
    assert outA.shape == portA.shape
    assert outB.shape == portB.shape


def test_fldzhyan_block_backward(block):
    portA = torch.rand(4, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(4, dtype=torch.complex128).requires_grad_(True)
    outA, outB = block.forward(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portB.grad is not None
    assert portA.grad.abs().sum() > 0
    assert portB.grad.abs().sum() > 0


def test_fldzhyan_block_power_conservation(block):
    portA = torch.rand(4, dtype=torch.complex128)
    portB = torch.rand(4, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    p_in = (portA.abs() ** 2).sum() + (portB.abs() ** 2).sum()
    p_out = (outA.abs() ** 2).sum() + (outB.abs() ** 2).sum()
    assert torch.isclose(p_in, p_out, rtol=1e-4)


def test_fldzhyan_block_parameters_trainable(block):
    assert len(list(block.parameters())) > 0


def test_fldzhyan_block_get_params(block):
    assert block.get_params() is not None


def test_fldzhyan_block_phase_shift_roundtrip(block):
    orig = block.get_heaters_phase_shift()
    new_phases = torch.ones_like(orig) * 0.5
    block.set_heaters_phase_shift(new_phases)
    retrieved = block.get_heaters_phase_shift()
    assert torch.allclose(retrieved, new_phases, atol=1e-6)


# ---------------------------------------------------------------------------
# Various sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 6])
def test_fldzhyan_block_various_sizes(n):
    m = FldzhyanBlock(_params(n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = m.forward(portA, portB)
    assert outA.shape == (n,)


# ---------------------------------------------------------------------------
# torch.nn.Module API
# ---------------------------------------------------------------------------

def test_fldzhyan_block_nn_api(block):
    block.train()
    block.eval()
    assert len(list(block.parameters())) > 0
