"""
Tests for MmiParams bindings and Mmi forward behavior.

MMI convention:
  - insertion_loss=None      → ideal 50/50 (mC = 1/√2)
  - insertion_loss=tensor(v) → mC = sqrt(v); ideal when v=0.5
    Total output power = 2*v for unit input on one port.
  - imbalance=tensor(δ)      → asymmetric split between ports.
"""
import pickle

import pytest
import torch

import luxia_lib
from luxia_lib import MmiParams as CppMmiParams
from luxia.photonic_components.beamsplitters.mmi import Mmi
from luxia.params.bs_params import MmiParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_portA(n: int = 1):
    return (
        torch.ones(n, dtype=torch.complex128),
        torch.zeros(n, dtype=torch.complex128),
    )


def _power(t: torch.Tensor) -> torch.Tensor:
    return (t.abs() ** 2).sum()


# ---------------------------------------------------------------------------
# MmiParams bindings (C++)
# ---------------------------------------------------------------------------

def test_mmi_params_cpp_is_nn_module():
    assert isinstance(CppMmiParams(), torch.nn.Module)


def test_mmi_params_cpp_default():
    p = CppMmiParams()
    assert p.number_of_splitters >= 1


def test_mmi_params_cpp_full():
    p = CppMmiParams(number_of_splitters=4)
    assert p.number_of_splitters == 4


def test_mmi_params_cpp_property_roundtrip():
    p = CppMmiParams()
    p.number_of_splitters = 7
    assert p.number_of_splitters == 7


def test_mmi_params_cpp_tensor_members():
    loss = torch.tensor([0.1, 0.1, 0.1])
    p = CppMmiParams(number_of_splitters=3, insertion_loss=loss)
    assert p.insertion_loss.shape == loss.shape


def test_mmi_params_cpp_repr():
    assert len(repr(CppMmiParams())) > 0


def test_mmi_params_cpp_forward_raises():
    with pytest.raises(TypeError):
        CppMmiParams().forward(torch.zeros(2))


def test_mmi_params_cpp_pickle():
    p = CppMmiParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, CppMmiParams)


def test_mmi_params_cpp_pickle_preserves_tensors():
    loss = torch.ones(4) * 0.05
    p = CppMmiParams(number_of_splitters=4, insertion_loss=loss)
    q = pickle.loads(pickle.dumps(p))
    assert torch.allclose(q.insertion_loss, p.insertion_loss)


# ---------------------------------------------------------------------------
# MmiParams (Python dataclass)
# ---------------------------------------------------------------------------

def test_mmi_params_py_default():
    p = MmiParams()
    assert p.number_of_splitters == 1
    assert p.insertion_loss is None
    assert p.imbalance is None


def test_mmi_params_py_full():
    loss = torch.tensor([0.4])
    imb = torch.tensor([0.05])
    p = MmiParams(number_of_splitters=2, insertion_loss=loss, imbalance=imb)
    assert p.number_of_splitters == 2
    assert torch.allclose(p.insertion_loss, loss)
    assert torch.allclose(p.imbalance, imb)


# ---------------------------------------------------------------------------
# Mmi — ideal (no imperfections)
# ---------------------------------------------------------------------------

def test_mmi_ideal_output_shape():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA, portB = _unit_portA()
    outA, outB = mmi(portA, portB)
    assert outA.shape == portA.shape
    assert outB.shape == portB.shape


def test_mmi_ideal_power_conservation():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128)
    portB = torch.rand(1, dtype=torch.complex128)
    p_in = _power(portA) + _power(portB)
    outA, outB = mmi(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_in, p_out, rtol=1e-6)


def test_mmi_ideal_equal_splitting():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA, portB = _unit_portA()
    outA, outB = mmi(portA, portB)
    assert torch.isclose(_power(outA), _power(outB), rtol=1e-6)


def test_mmi_ideal_phase_quadrature():
    """Ideal MMI output has 90° phase shift (j) on the cross-coupled port."""
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = mmi(portA, portB)
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    assert abs(outA[0].real - inv_sqrt2) < 1e-7
    assert abs(outA[0].imag) < 1e-7
    assert abs(outB[0].real) < 1e-7
    assert abs(outB[0].imag - inv_sqrt2) < 1e-7


# ---------------------------------------------------------------------------
# Mmi — insertion loss
# ---------------------------------------------------------------------------

def test_mmi_insertion_loss_reduces_power():
    """insertion_loss=0.4 gives total output power 0.8 for unit input."""
    mmi_lossy = Mmi(MmiParams(number_of_splitters=1, insertion_loss=torch.tensor([0.4])))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    oA, oB = mmi_lossy(portA, portB)
    p_out = _power(oA) + _power(oB)
    assert torch.isclose(p_out, torch.tensor(0.8, dtype=torch.float64), rtol=1e-6)


def test_mmi_lossless_insertion_loss_conserves_power():
    """insertion_loss=0.5 is equivalent to ideal lossless."""
    mmi = Mmi(MmiParams(number_of_splitters=1, insertion_loss=torch.tensor([0.5])))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = mmi(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_out, torch.tensor(1.0, dtype=torch.float64), rtol=1e-6)


# ---------------------------------------------------------------------------
# Mmi — imbalance
# ---------------------------------------------------------------------------

def test_mmi_imbalance_unequal_ports():
    """Positive imbalance causes port 0 to carry more power than port 1."""
    mmi = Mmi(MmiParams(number_of_splitters=1, imbalance=torch.tensor([0.1])))
    portA, portB = _unit_portA()
    outA, outB = mmi(portA, portB)
    assert _power(outA) > _power(outB)


def test_mmi_zero_imbalance_equal_ports():
    """imbalance=0 gives equal splitting."""
    mmi = Mmi(MmiParams(number_of_splitters=1, imbalance=torch.tensor([0.0])))
    portA, portB = _unit_portA()
    outA, outB = mmi(portA, portB)
    assert torch.isclose(_power(outA), _power(outB), rtol=1e-6)


# ---------------------------------------------------------------------------
# Mmi — gradient propagation
# ---------------------------------------------------------------------------

def test_mmi_gradients_through_portA():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = mmi(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None


def test_mmi_gradients_through_portB():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA = torch.zeros(1, dtype=torch.complex128)
    portB = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    outA, outB = mmi(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portB.grad is not None


def test_mmi_gradients_both_ports():
    mmi = Mmi(MmiParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    outA, outB = mmi(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portB.grad is not None


# ---------------------------------------------------------------------------
# Mmi — multiple splitters (vectorised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 8])
def test_mmi_vectorised_shape(n):
    mmi = Mmi(MmiParams(number_of_splitters=n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = mmi(portA, portB)
    assert outA.shape == (n,)
    assert outB.shape == (n,)


@pytest.mark.parametrize("n", [2, 4])
def test_mmi_vectorised_power_conservation(n):
    mmi = Mmi(MmiParams(number_of_splitters=n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    p_in = _power(portA) + _power(portB)
    outA, outB = mmi(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_in, p_out, rtol=1e-6)
