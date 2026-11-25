"""
Tests for DacParams bindings and Dac (Python + C++) forward/backward/speed.
"""
import pickle
import timeit

import pytest
import torch

import luxia_lib
from luxia_lib import DacParams
from luxia.io_components.dac import Dac as DacPy
from luxia.types import round_t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dac_params():
    return DacParams(
        number_of_bits=3,
        min_analog_reference=0.0,
        max_analog_reference=1.0,
        min_digital_reference=0.0,
        max_digital_reference=1.0,
        round_type=luxia_lib.lx_round_t.ROUND,
    )


@pytest.fixture
def batch_tensor():
    return torch.rand(10_000, requires_grad=True)


# ---------------------------------------------------------------------------
# DacParams bindings
# ---------------------------------------------------------------------------

def test_dac_params_is_nn_module():
    assert isinstance(DacParams(), torch.nn.Module)


def test_dac_params_default():
    p = DacParams()
    assert p.number_of_bits > 0


def test_dac_params_full():
    p = DacParams(
        number_of_bits=4,
        min_analog_reference=0.0,
        max_analog_reference=2.0,
        min_digital_reference=0.0,
        max_digital_reference=1.0,
        round_type=luxia_lib.lx_round_t.FLOOR,
    )
    assert p.number_of_bits == 4
    assert p.max_analog_reference == pytest.approx(2.0)
    assert p.round_type == luxia_lib.lx_round_t.FLOOR


def test_dac_params_repr():
    assert len(repr(DacParams())) > 0


def test_dac_params_forward_raises():
    with pytest.raises(TypeError):
        DacParams().forward(torch.zeros(2))


def test_dac_params_pickle():
    p = DacParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, DacParams)


# ---------------------------------------------------------------------------
# Dac construction / module import
# ---------------------------------------------------------------------------

def test_dac_cpp_module_attributes():
    assert hasattr(luxia_lib, "DacParams")
    assert hasattr(luxia_lib, "Dac")
    assert hasattr(luxia_lib.lx_round_t, "ROUND")
    assert hasattr(luxia_lib.lx_round_t, "FLOOR")


def test_dac_cpp_creation(dac_params):
    dac = luxia_lib.Dac(dac_params)
    dac.set_debug_mode(False)
    assert dac.get_debug_mode() is False
    dac.set_debug_mode(True)
    assert dac.get_debug_mode() is True


# ---------------------------------------------------------------------------
# Forward (CPU + CUDA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dac_forward_device(impl, device, dac_params):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dac = DacPy(dac_params) if impl == "python" else luxia_lib.Dac(dac_params)
    x = torch.linspace(0.0, 1.0, steps=10, device=device, requires_grad=True)
    out = dac.forward(x)

    assert torch.is_tensor(out)
    assert out.device.type == device
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dac_backward_device(impl, device, dac_params):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dac = DacPy(dac_params) if impl == "python" else luxia_lib.Dac(dac_params)
    x = torch.linspace(0.0, 1.0, steps=10, device=device, requires_grad=True)
    dac.forward(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Python vs C++ equivalence
# ---------------------------------------------------------------------------

def test_python_and_cpp_outputs_in_range(dac_params):
    """Both Python and C++ DAC produce normalized outputs in [0, 1].

    Float32 rounding at quantization boundaries means exact equality between
    the two implementations is not guaranteed.
    """
    dac_py = DacPy(dac_params)
    dac_cpp = luxia_lib.Dac(dac_params)
    x = torch.linspace(0.0, 1.0, steps=10)
    out_py = dac_py.forward(x.clone())
    out_cpp = dac_cpp.forward(x.clone())
    assert out_py.min() >= 0.0 - 1e-6 and out_py.max() <= 1.0 + 1e-6
    assert out_cpp.min() >= 0.0 - 1e-6 and out_cpp.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_dac_edge_empty_tensor(dac_params):
    empty = torch.tensor([])
    out = DacPy(dac_params).forward(empty)
    assert out.shape == empty.shape

    out_cpp = luxia_lib.Dac(dac_params).forward(empty)
    assert out_cpp.shape == empty.shape


def test_dac_edge_out_of_range_raises(dac_params):
    invalid = torch.tensor([-0.1, 1.1])
    with pytest.raises(ValueError, match="Input values must be between 0 and 1"):
        DacPy(dac_params).forward(invalid)
    with pytest.raises(RuntimeError):
        luxia_lib.Dac(dac_params).forward(invalid)


def test_dac_edge_2d_tensor(dac_params):
    t = torch.tensor([[0.1, 0.5], [0.7, 0.9]])
    assert DacPy(dac_params).forward(t).shape == t.shape
    assert luxia_lib.Dac(dac_params).forward(t).shape == t.shape


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_dac_cpp_metrics(dac_params):
    dac = luxia_lib.Dac(dac_params)
    dac.forward(torch.tensor([0.5]))
    assert dac.get_snr() > 0
    assert dac.get_sqnr() > 0
    assert torch.is_tensor(dac.get_quantization_error())


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ["python", "cpp"])
def test_dac_torch_save_load(tmp_path, impl, dac_params):
    dac_cls = DacPy if impl == "python" else luxia_lib.Dac
    dac = dac_cls(dac_params)
    x = torch.linspace(0.0, 1.0, steps=10)
    out_before = dac(x)
    path = tmp_path / f"dac_{impl}.pt"
    torch.save(dac, path)
    dac_loaded = torch.load(path, weights_only=False)
    assert torch.allclose(out_before, dac_loaded(x), atol=1e-6)


def test_dac_cpp_pickle_roundtrip(dac_params):
    dac = luxia_lib.Dac(dac_params)
    dac2 = pickle.loads(pickle.dumps(dac))
    p1, p2 = dac.get_params(), dac2.get_params()
    assert p1.number_of_bits == p2.number_of_bits
    assert p1.round_type == p2.round_type
    x = torch.linspace(0.0, 1.0, steps=10)
    assert torch.allclose(dac(x), dac2(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def test_dac_speed_cpp_vs_python(dac_params, batch_tensor):
    dac_py = DacPy(dac_params)
    dac_cpp = luxia_lib.Dac(dac_params)
    x = batch_tensor.detach()
    n = 100
    t_py = timeit.timeit(lambda: dac_py.forward(x), number=n)
    t_cpp = timeit.timeit(lambda: dac_cpp.forward(x), number=n)
    print(f"\nDAC Python={t_py:.3f}s C++={t_cpp:.3f}s ({n} iters, {x.numel()} elements)")
    assert t_cpp < t_py * 2
