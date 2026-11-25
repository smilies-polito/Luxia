"""
Tests for ClementsBlockParams bindings and ClementsBlock (Python + C++).
"""
import pickle
import timeit

import pytest
import torch

import luxia_lib
from luxia_lib import (
    ClementsBlock as CppClementsBlock,
    ClementsBlockParams as CppClementsBlockParams,
)
from luxia.photonic_meshes.clements import ClementsBlock as PyClementsBlock
from luxia.params import (
    ClementsBlockParams, HeaterParams, MmiParams, DacParams,
)
from luxia.types import round_t


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _params(n: int = 8) -> ClementsBlockParams:
    """Build ClementsBlockParams using Python params (compatible with both Py and C++)."""
    hp = [HeaterParams(number_of_heaters=n), HeaterParams(number_of_heaters=n)]
    mmi = [MmiParams(number_of_splitters=n), MmiParams(number_of_splitters=n)]
    return ClementsBlockParams(number_of_blocks=n, heater_params=hp, mmi_params=mmi)


@pytest.fixture
def small_params():
    return _params(8)


@pytest.fixture
def large_params():
    dac = DacParams(number_of_bits=8)
    hp = [HeaterParams(number_of_heaters=2048, dac_params=dac),
          HeaterParams(number_of_heaters=2048, dac_params=dac)]
    mmi = [MmiParams(number_of_splitters=2048), MmiParams(number_of_splitters=2048)]
    return ClementsBlockParams(number_of_blocks=2048, heater_params=hp, mmi_params=mmi)


@pytest.fixture(params=["python", "cpp"])
def impl(request):
    return request.param


def _make(impl_name, params):
    return PyClementsBlock(params) if impl_name == "python" else CppClementsBlock(params)


# ---------------------------------------------------------------------------
# ClementsBlockParams bindings (C++)
# ---------------------------------------------------------------------------

def test_clements_block_params_is_nn_module():
    assert isinstance(CppClementsBlockParams(), torch.nn.Module)


def test_clements_block_params_default():
    p = CppClementsBlockParams()
    assert p.number_of_blocks >= 1


def test_clements_block_params_full():
    from luxia_lib import HeaterParams as CppHP, MmiParams as CppMmi
    hp = [CppHP(number_of_heaters=4), CppHP(number_of_heaters=4)]
    mmi = [CppMmi(number_of_splitters=4), CppMmi(number_of_splitters=4)]
    p = CppClementsBlockParams(number_of_blocks=4, heater_params=hp, mmi_params=mmi)
    assert p.number_of_blocks == 4


def test_clements_block_params_repr():
    assert len(repr(CppClementsBlockParams())) > 0


def test_clements_block_params_forward_raises():
    with pytest.raises(TypeError):
        CppClementsBlockParams().forward(torch.zeros(2))


def test_clements_block_params_pickle():
    p = CppClementsBlockParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, CppClementsBlockParams)


# ---------------------------------------------------------------------------
# ClementsBlock forward / backward
# ---------------------------------------------------------------------------

def test_clements_block_forward_shape(impl, small_params):
    block = _make(impl, small_params)
    n = small_params.number_of_blocks
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    assert torch.is_complex(outA) and torch.is_complex(outB)
    assert outA.shape == portA.shape
    assert outB.shape == portB.shape


def test_clements_block_backward(impl, small_params):
    block = _make(impl, small_params)
    n = small_params.number_of_blocks
    portA = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    outA, outB = block.forward(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portB.grad is not None
    assert portA.grad.abs().sum() > 0
    assert portB.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clements_block_dtype(impl, small_params, dtype):
    block = _make(impl, small_params)
    n = small_params.number_of_blocks
    portA = torch.rand(n, dtype=dtype) + 0j
    portB = torch.rand(n, dtype=dtype) + 0j
    outA, outB = block.forward(portA, portB)
    expected = torch.complex64 if dtype == torch.float32 else torch.complex128
    assert outA.dtype == expected and outB.dtype == expected


def test_clements_block_parameters_trainable(impl, small_params):
    block = _make(impl, small_params)
    assert len(list(block.parameters())) > 0


# ---------------------------------------------------------------------------
# Various sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 8])
def test_clements_block_various_sizes(impl, n):
    block = _make(impl, _params(n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = block.forward(portA, portB)
    assert outA.shape == (n,)


# ---------------------------------------------------------------------------
# torch.nn.Module API
# ---------------------------------------------------------------------------

def test_clements_block_nn_api(impl, small_params):
    block = _make(impl, small_params)
    block.train()
    block.eval()
    assert len(list(block.parameters())) > 0


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def test_clements_block_speed(large_params):
    py_block = PyClementsBlock(large_params)
    cpp_block = CppClementsBlock(large_params)
    n = large_params.number_of_blocks
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    n_iters = 50

    t_py = timeit.timeit(lambda: py_block.forward(portA, portB), number=n_iters)
    t_cpp = timeit.timeit(lambda: cpp_block.forward(portA, portB), number=n_iters)
    print(f"\nClementsBlock Python={t_py:.3f}s C++={t_cpp:.3f}s ({n_iters} iters, n={n})")
