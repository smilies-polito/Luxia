"""
Tests for AdcParams bindings and Adc (Python + C++) forward/backward/speed.
"""
import pickle
import timeit

import pytest
import torch

import luxia_lib
from luxia_lib import AdcParams, Adc as AdcCpp
from luxia.io_components.adc import Adc as AdcPy
from luxia.params.adc_params import AdcParams as PyAdcParams
from luxia.types import round_t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adc_params():
    return AdcParams(
        number_of_bits=4,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        round_type=luxia_lib.lx_round_t.FLOOR,
    )


@pytest.fixture
def py_adc_params():
    return PyAdcParams(
        number_of_bits=4,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        round_type=round_t.FLOOR,
    )


@pytest.fixture
def batch_tensor():
    return torch.rand(10_000)


# ---------------------------------------------------------------------------
# AdcParams bindings (C++)
# ---------------------------------------------------------------------------

def test_adc_params_is_nn_module():
    assert isinstance(AdcParams(), torch.nn.Module)


def test_adc_params_default():
    p = AdcParams()
    assert p.number_of_bits > 0


def test_adc_params_full():
    p = AdcParams(
        number_of_bits=6,
        min_analog_reference=-1.0,
        max_analog_reference=1.0,
        round_type=luxia_lib.lx_round_t.CEIL,
    )
    assert p.number_of_bits == 6
    assert p.min_analog_reference == pytest.approx(-1.0)
    assert p.max_analog_reference == pytest.approx(1.0)
    assert p.round_type == luxia_lib.lx_round_t.CEIL


def test_adc_params_property_roundtrip():
    p = AdcParams()
    p.number_of_bits = 10
    assert p.number_of_bits == 10
    p.min_analog_reference = -0.5
    assert p.min_analog_reference == pytest.approx(-0.5)
    p.max_analog_reference = 0.5
    assert p.max_analog_reference == pytest.approx(0.5)
    p.round_type = luxia_lib.lx_round_t.CEIL
    assert p.round_type == luxia_lib.lx_round_t.CEIL


def test_adc_params_repr():
    assert len(repr(AdcParams())) > 0


def test_adc_params_forward_raises():
    with pytest.raises(TypeError):
        AdcParams().forward(torch.zeros(2))


def test_adc_params_pickle():
    p = AdcParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, AdcParams)


# ---------------------------------------------------------------------------
# AdcParams (Python dataclass)
# ---------------------------------------------------------------------------

def test_py_adc_params_construction(py_adc_params):
    assert py_adc_params.number_of_bits == 4
    assert py_adc_params.min_analog_reference == pytest.approx(0.0)
    assert py_adc_params.max_analog_reference == pytest.approx(1.0)
    assert py_adc_params.round_type == round_t.FLOOR


# ---------------------------------------------------------------------------
# Adc construction / module attributes
# ---------------------------------------------------------------------------

def test_adc_cpp_module_attributes():
    assert hasattr(luxia_lib, "AdcParams")
    assert hasattr(luxia_lib, "Adc")
    assert hasattr(luxia_lib.lx_round_t, "ROUND")
    assert hasattr(luxia_lib.lx_round_t, "FLOOR")
    assert hasattr(luxia_lib.lx_round_t, "CEIL")


def test_adc_cpp_creation(adc_params):
    adc = AdcCpp(adc_params)
    adc.set_debug_mode(False)
    assert adc.get_debug_mode() is False
    adc.set_debug_mode(True)
    assert adc.get_debug_mode() is True


def test_adc_py_creation(py_adc_params):
    adc = AdcPy(py_adc_params)
    assert isinstance(adc, torch.nn.Module)


# ---------------------------------------------------------------------------
# Forward (CPU + CUDA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adc_forward_device(impl, device, adc_params, py_adc_params):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if impl == "python":
        adc = AdcPy(py_adc_params)
    else:
        adc = AdcCpp(adc_params)

    x = torch.linspace(0.0, 1.0, steps=10, device=device)
    out = adc.forward(x)

    assert torch.is_tensor(out)
    assert out.device.type == device
    assert out.shape == x.shape


@pytest.mark.parametrize("impl", ["python", "cpp"])
def test_adc_forward_values_in_range(impl, adc_params, py_adc_params):
    adc = AdcPy(py_adc_params) if impl == "python" else AdcCpp(adc_params)
    x = torch.linspace(0.0, 1.0, steps=16)
    out = adc.forward(x)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adc_backward_device(impl, device, adc_params, py_adc_params):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if impl == "python":
        adc = AdcPy(py_adc_params)
    else:
        adc = AdcCpp(adc_params)

    x = torch.linspace(0.0, 1.0, steps=10, device=device, requires_grad=True)
    adc.forward(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Python and C++ output range sanity
# ---------------------------------------------------------------------------

def test_python_and_cpp_outputs_in_range(adc_params, py_adc_params):
    """Both Python and C++ ADC produce normalized outputs in [0, 1].

    Float32 rounding at quantization boundaries means exact equality between
    the two implementations is not guaranteed.
    """
    adc_py = AdcPy(py_adc_params)
    adc_cpp = AdcCpp(adc_params)
    x = torch.linspace(0.0, 1.0, steps=16)
    out_py = adc_py.forward(x.clone())
    out_cpp = adc_cpp.forward(x.clone())
    assert out_py.min() >= 0.0 - 1e-6 and out_py.max() <= 1.0 + 1e-6
    assert out_cpp.min() >= 0.0 - 1e-6 and out_cpp.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_adc_edge_empty_tensor(adc_params, py_adc_params):
    empty = torch.tensor([])
    out_py = AdcPy(py_adc_params).forward(empty)
    assert out_py.shape == empty.shape

    out_cpp = AdcCpp(adc_params).forward(empty)
    assert out_cpp.shape == empty.shape


def test_adc_edge_2d_tensor(adc_params, py_adc_params):
    t = torch.tensor([[0.1, 0.5], [0.7, 0.9]])
    assert AdcPy(py_adc_params).forward(t).shape == t.shape
    assert AdcCpp(adc_params).forward(t).shape == t.shape


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_adc_cpp_metrics(adc_params):
    adc = AdcCpp(adc_params)
    adc.forward(torch.linspace(0.0, 1.0, steps=16))
    assert adc.get_snr() > 0
    assert adc.get_sqnr() > 0
    assert torch.is_tensor(adc.get_quantization_error())
    assert adc.get_quantization_error().shape[0] == 16


def test_adc_py_metrics(py_adc_params):
    adc = AdcPy(py_adc_params)
    adc.forward(torch.linspace(0.0, 1.0, steps=16))
    assert float(adc.snr) > 0
    assert float(adc.sqnr) > 0
    assert torch.is_tensor(adc.quantization_error)
    assert adc.quantization_error.shape[0] == 16


# ---------------------------------------------------------------------------
# Params get/set
# ---------------------------------------------------------------------------

def test_adc_cpp_get_params(adc_params):
    adc = AdcCpp(adc_params)
    p = adc.get_params()
    assert p.number_of_bits == adc_params.number_of_bits


def test_adc_cpp_set_params(adc_params):
    adc = AdcCpp(adc_params)
    new_p = AdcParams(number_of_bits=6)
    adc.set_params(new_p)
    assert adc.get_params().number_of_bits == 6


def test_adc_py_set_params(py_adc_params):
    adc = AdcPy(py_adc_params)
    new_p = PyAdcParams(number_of_bits=6)
    adc.set_params(new_p)
    assert adc.params.number_of_bits == 6


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
def test_adc_torch_save_load(tmp_path, impl, adc_params, py_adc_params):
    adc = AdcPy(py_adc_params) if impl == "python" else AdcCpp(adc_params)
    x = torch.linspace(0.0, 1.0, steps=10)
    out_before = adc(x)
    path = tmp_path / f"adc_{impl}.pt"
    torch.save(adc, path)
    adc_loaded = torch.load(path, weights_only=False)
    assert torch.allclose(out_before, adc_loaded(x), atol=1e-6)


def test_adc_cpp_pickle_roundtrip(adc_params):
    adc = AdcCpp(adc_params)
    adc2 = pickle.loads(pickle.dumps(adc))
    p1, p2 = adc.get_params(), adc2.get_params()
    assert p1.number_of_bits == p2.number_of_bits
    assert p1.round_type == p2.round_type
    x = torch.linspace(0.0, 1.0, steps=10)
    assert torch.allclose(adc(x), adc2(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def test_adc_speed_cpp_vs_python(adc_params, py_adc_params, batch_tensor):
    adc_py = AdcPy(py_adc_params)
    adc_cpp = AdcCpp(adc_params)
    x = batch_tensor
    n = 100
    t_py = timeit.timeit(lambda: adc_py.forward(x), number=n)
    t_cpp = timeit.timeit(lambda: adc_cpp.forward(x), number=n)
    print(f"\nADC Python={t_py:.3f}s C++={t_cpp:.3f}s ({n} iters, {x.numel()} elements)")
    assert t_cpp < t_py * 2
