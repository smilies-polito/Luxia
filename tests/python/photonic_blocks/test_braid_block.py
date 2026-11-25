"""
Tests for BraidBlockParams bindings and BraidBlock (C++).
"""
import pickle

import pytest
import torch

from luxia_lib import (
    BraidBlock,
    BraidBlockParams, CrossParams,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _params(n: int = 2) -> BraidBlockParams:
    return BraidBlockParams(
        number_of_blocks=n,
        cross_params=CrossParams(number_of_splitters=n),
    )


@pytest.fixture
def block():
    return BraidBlock(_params(2))


# ---------------------------------------------------------------------------
# BraidBlockParams bindings (C++)
# ---------------------------------------------------------------------------

def test_braid_block_params_is_nn_module():
    assert isinstance(BraidBlockParams(), torch.nn.Module)


def test_braid_block_params_default():
    p = BraidBlockParams()
    assert p.number_of_blocks >= 1


def test_braid_block_params_full():
    p = _params(3)
    assert p.number_of_blocks == 3


def test_braid_block_params_repr():
    assert len(repr(BraidBlockParams())) > 0


def test_braid_block_params_forward_raises():
    with pytest.raises(TypeError):
        BraidBlockParams().forward(torch.zeros(2))


def test_braid_block_params_pickle():
    p = BraidBlockParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, BraidBlockParams)


# ---------------------------------------------------------------------------
# BraidBlock forward / backward
# ---------------------------------------------------------------------------

def test_braid_block_construction():
    m = BraidBlock(_params(2))
    assert isinstance(m, torch.nn.Module)


def test_braid_block_forward_shape(block):
    portA = torch.rand(2, dtype=torch.complex128)
    portB = torch.rand(2, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    assert torch.is_complex(outA) and torch.is_complex(outB)
    assert outA.shape == portA.shape
    assert outB.shape == portB.shape


def test_braid_block_backward(block):
    portA = torch.rand(2, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(2, dtype=torch.complex128).requires_grad_(True)
    outA, outB = block.forward(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portA.grad.abs().sum() > 0


def test_braid_block_power_conservation(block):
    portA = torch.rand(2, dtype=torch.complex128)
    portB = torch.rand(2, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    p_in = (portA.abs() ** 2).sum() + (portB.abs() ** 2).sum()
    p_out = (outA.abs() ** 2).sum() + (outB.abs() ** 2).sum()
    assert torch.isclose(p_in, p_out, rtol=1e-4)


def test_braid_block_get_params(block):
    assert block.get_params() is not None


# ---------------------------------------------------------------------------
# Various sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 4])
def test_braid_block_various_sizes(n):
    m = BraidBlock(_params(n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = m.forward(portA, portB)
    assert outA.shape == (n,)


# ---------------------------------------------------------------------------
# torch.nn.Module API
# ---------------------------------------------------------------------------

def test_braid_block_nn_api(block):
    block.train()
    block.eval()
