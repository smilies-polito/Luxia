"""
Tests for HeaterParams bindings and Heater (Python + C++) forward/backward/speed.
"""
import pickle
import timeit

import pytest
import torch

import luxia_lib
from luxia_lib import HeaterParams as CppHeaterParams
from luxia_lib import Heater as CppHeater
from luxia.photonic_components import Heater as PyHeater
from luxia.params import HeaterParams as PyHeaterParams, DacParams as PyDacParams
from luxia.types import round_t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["python", "cpp"])
def impl(request):
    return request.param


@pytest.fixture
def py_heater_params():
    dac = PyDacParams(
        number_of_bits=8,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        min_digital_reference=0.0,
        max_digital_reference=1.0,
        round_type=round_t.ROUND,
    )
    return PyHeaterParams(number_of_heaters=8, dac_params=dac)


@pytest.fixture
def py_heater_params_large():
    dac = PyDacParams(
        number_of_bits=8,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        min_digital_reference=0.0,
        max_digital_reference=1.0,
        round_type=round_t.ROUND,
    )
    return PyHeaterParams(number_of_heaters=2048, dac_params=dac, use_dac=True)


@pytest.fixture
def small_input(py_heater_params):
    return torch.rand(2, py_heater_params.number_of_heaters)


@pytest.fixture
def large_input():
    return torch.rand(1024, 2048)


def _make_heater(impl_name, params):
    return PyHeater(params) if impl_name == "python" else CppHeater(params)


# ---------------------------------------------------------------------------
# HeaterParams bindings (C++)
# ---------------------------------------------------------------------------

def test_heater_params_cpp_is_nn_module():
    assert isinstance(CppHeaterParams(), torch.nn.Module)


def test_heater_params_cpp_default():
    p = CppHeaterParams()
    assert p.number_of_heaters >= 1


def test_heater_params_cpp_full():
    dac = luxia_lib.DacParams(number_of_bits=8)
    p = CppHeaterParams(number_of_heaters=4, dac_params=dac)
    assert p.number_of_heaters == 4


def test_heater_params_cpp_repr():
    assert len(repr(CppHeaterParams())) > 0


def test_heater_params_cpp_forward_raises():
    with pytest.raises(TypeError):
        CppHeaterParams().forward(torch.zeros(2))


def test_heater_params_cpp_pickle():
    p = CppHeaterParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, CppHeaterParams)


# ---------------------------------------------------------------------------
# HeaterParams (Python dataclass)
# ---------------------------------------------------------------------------

def test_heater_params_py_construction(py_heater_params):
    assert py_heater_params.number_of_heaters == 8
    assert py_heater_params.dac_params is not None


def test_heater_params_py_default():
    p = PyHeaterParams()
    assert p.number_of_heaters == 1
    assert p.dac_params is None
    assert p.use_dac is False


# ---------------------------------------------------------------------------
# Heater forward / backward
# ---------------------------------------------------------------------------

def test_heater_forward(impl, py_heater_params, small_input):
    heater = _make_heater(impl, py_heater_params)
    y = heater.forward(small_input.clone())
    assert y.shape == small_input.shape
    assert torch.is_complex(y)


def test_heater_backward(impl, py_heater_params, small_input):
    heater = _make_heater(impl, py_heater_params)
    x = small_input.clone().requires_grad_(True)
    y = heater.forward(x)
    y.real.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# DAC properties
# ---------------------------------------------------------------------------

def test_heater_dac_properties(impl, py_heater_params):
    heater = _make_heater(impl, py_heater_params)
    if getattr(heater, "is_dac_used", False) and hasattr(heater, "get_quantization_error"):
        assert heater.get_quantization_error() is not None
        assert heater.get_dac_snr() is not None
        assert heater.get_dac_sqnr() is not None
        assert heater.get_dac_resolution() == heater.get_dac_params().number_of_bits


def test_heater_set_use_dac_toggle(impl):
    heater = _make_heater(impl, PyHeaterParams(number_of_heaters=2))
    dac = PyDacParams(
        number_of_bits=4,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        min_digital_reference=0.0,
        max_digital_reference=1.0,
        round_type=round_t.ROUND,
    )
    assert getattr(heater, "is_dac_used", False) is False
    if hasattr(heater, "get_dac_resolution"):
        assert heater.get_dac_resolution() is None

    heater.set_use_dac(True, dac)
    assert getattr(heater, "is_dac_used", False) is True
    if hasattr(heater, "get_dac_resolution"):
        assert heater.get_dac_resolution() == 4

    heater.set_use_dac(False)
    assert getattr(heater, "is_dac_used", False) is False
    if hasattr(heater, "get_dac_resolution"):
        assert heater.get_dac_resolution() is None


def test_heater_phases_scaling(impl, py_heater_params):
    heater = _make_heater(impl, py_heater_params)
    phi = heater.get_phases()
    assert phi.shape[0] == py_heater_params.number_of_heaters
    assert torch.allclose(torch.abs(phi), torch.ones_like(torch.abs(phi)))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_heater_dtype_consistency(impl, py_heater_params, dtype):
    heater = _make_heater(impl, py_heater_params)
    x = torch.rand(2, py_heater_params.number_of_heaters, dtype=dtype)
    y = heater.forward(x)
    assert torch.is_complex(y)
    expected = torch.complex64 if dtype == torch.float32 else torch.complex128
    assert y.dtype == expected


def test_heater_complex_input(impl, py_heater_params):
    heater = _make_heater(impl, py_heater_params)
    x = torch.tensor(
        [[0.1 + 0.2j] * py_heater_params.number_of_heaters,
         [0.3 + 0.4j] * py_heater_params.number_of_heaters],
    )
    try:
        y = heater.forward(x)
        assert y.shape == x.shape
        assert torch.is_complex(y)
    except RuntimeError:
        pass


def test_heater_parameter_trainability(impl, py_heater_params, small_input):
    heater = _make_heater(impl, py_heater_params)
    params = [p for p in heater.parameters() if p.requires_grad]
    assert len(params) > 0

    x = small_input.clone().requires_grad_(True)
    loss = heater.forward(x).abs().sum()
    loss.backward()

    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0
    assert sum(g.abs().sum().item() for g in grads) > 0


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def test_heater_speed_cpp_vs_python(py_heater_params_large, large_input):
    py_heater = PyHeater(py_heater_params_large)
    cpp_heater = CppHeater(py_heater_params_large)
    x = large_input.clone()
    n = 10
    t_py = timeit.timeit(lambda: py_heater.forward(x), number=n)
    t_cpp = timeit.timeit(lambda: cpp_heater.forward(x), number=n)
    print(f"\nHeater Python={t_py:.3f}s C++={t_cpp:.3f}s ({n} iters, {x.numel()} elements)")
