"""
Tests for CrossParams bindings and Cross forward behavior.

Cross convention:
  - crosstalk=None      → ideal crossing: input A → output B, input B → output A
  - crosstalk=tensor(ε) → ε fraction of power leaks to the straight-through port
    Transfer matrix: [[sqrt(ε), sqrt(1-ε)], [sqrt(1-ε), sqrt(ε)]] — real symmetric,
    not unitary. Power conserved per column (single-port input) but not for general
    two-port inputs.
"""
import pickle

import pytest
import torch

import luxia_lib
from luxia_lib import CrossParams as CppCrossParams
from luxia.photonic_components.beamsplitters.cross import Cross
from luxia.params.bs_params import CrossParams


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
# CrossParams bindings (C++)
# ---------------------------------------------------------------------------

def test_cross_params_cpp_is_nn_module():
    assert isinstance(CppCrossParams(), torch.nn.Module)


def test_cross_params_cpp_default():
    p = CppCrossParams()
    assert p.number_of_splitters >= 1


def test_cross_params_cpp_full():
    p = CppCrossParams(number_of_splitters=2)
    assert p.number_of_splitters == 2


def test_cross_params_cpp_tensor_members():
    ct = torch.tensor([-30.0, -30.0])
    p = CppCrossParams(number_of_splitters=2, crosstalk=ct)
    assert p.crosstalk.shape == ct.shape


def test_cross_params_cpp_repr():
    assert len(repr(CppCrossParams())) > 0


def test_cross_params_cpp_forward_raises():
    with pytest.raises(TypeError):
        CppCrossParams().forward(torch.zeros(2))


def test_cross_params_cpp_pickle():
    p = CppCrossParams()
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, CppCrossParams)


def test_cross_params_cpp_pickle_preserves_tensors():
    ct = torch.tensor([-25.0])
    p = CppCrossParams(number_of_splitters=1, crosstalk=ct)
    q = pickle.loads(pickle.dumps(p))
    assert torch.allclose(q.crosstalk, p.crosstalk)


# ---------------------------------------------------------------------------
# CrossParams (Python dataclass)
# ---------------------------------------------------------------------------

def test_cross_params_py_default():
    p = CrossParams()
    assert p.number_of_splitters == 1
    assert p.insertion_loss is None
    assert p.crosstalk is None


def test_cross_params_py_full():
    ct = torch.tensor([0.05])
    p = CrossParams(number_of_splitters=1, crosstalk=ct)
    assert p.number_of_splitters == 1
    assert torch.allclose(p.crosstalk, ct)


# ---------------------------------------------------------------------------
# Cross — ideal (no imperfections)
# ---------------------------------------------------------------------------

def test_cross_ideal_output_shape():
    cross = Cross(CrossParams(number_of_splitters=1))
    portA, portB = _unit_portA()
    outA, outB = cross(portA, portB)
    assert outA.shape == portA.shape
    assert outB.shape == portB.shape


def test_cross_ideal_pure_crossing():
    """Ideal cross routes input on port A entirely to output port B."""
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    assert torch.isclose(_power(outA), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
    assert torch.isclose(_power(outB), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_cross_ideal_power_conservation():
    """Ideal cross conserves total optical power."""
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128)
    portB = torch.rand(1, dtype=torch.complex128)
    p_in = _power(portA) + _power(portB)
    outA, outB = cross(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_in, p_out, rtol=1e-6)


def test_cross_ideal_swaps_ports():
    """Ideal cross swaps portA and portB (both directions)."""
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.zeros(1, dtype=torch.complex128)
    portB = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    assert torch.isclose(_power(outA), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)
    assert torch.isclose(_power(outB), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# Cross — crosstalk
# ---------------------------------------------------------------------------

def test_cross_crosstalk_leaks_to_wrong_port():
    """Non-zero crosstalk causes power to appear on the straight-through port."""
    cross_ideal = Cross(CrossParams(number_of_splitters=1))
    cross_ct = Cross(CrossParams(number_of_splitters=1, crosstalk=torch.tensor([0.1])))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    oA_i, _ = cross_ideal(portA, portB)
    oA_ct, _ = cross_ct(portA, portB)
    assert _power(oA_ct) > _power(oA_i)


def test_cross_crosstalk_power_at_wrong_port():
    """crosstalk=ε means exactly ε fraction of input power on the wrong port."""
    epsilon = 0.1
    cross = Cross(CrossParams(number_of_splitters=1, crosstalk=torch.tensor([epsilon])))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    assert torch.isclose(_power(outA), torch.tensor(epsilon, dtype=torch.float64), rtol=1e-6)
    assert torch.isclose(_power(outB), torch.tensor(1.0 - epsilon, dtype=torch.float64), rtol=1e-6)


def test_cross_zero_crosstalk_is_ideal():
    """crosstalk=0.0 is equivalent to ideal (no leakage)."""
    cross = Cross(CrossParams(number_of_splitters=1, crosstalk=torch.tensor([0.0])))
    portA = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    assert torch.isclose(_power(outA), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
    assert torch.isclose(_power(outB), torch.tensor(1.0, dtype=torch.float64), rtol=1e-6)


def test_cross_crosstalk_single_port_power_conservation():
    """Power conserved for single-port input even with crosstalk.

    The transfer matrix is real-symmetric but not unitary, so power conservation
    holds per-column (single-port) but not for general two-port inputs.
    """
    cross = Cross(CrossParams(number_of_splitters=1, crosstalk=torch.tensor([0.05])))
    portA = torch.rand(1, dtype=torch.complex128)
    portB = torch.zeros(1, dtype=torch.complex128)
    p_in = _power(portA)
    outA, outB = cross(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_in, p_out, rtol=1e-6)


# ---------------------------------------------------------------------------
# Cross — gradient propagation
# ---------------------------------------------------------------------------

def test_cross_gradients_through_portA():
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    portB = torch.zeros(1, dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None


def test_cross_gradients_through_portB():
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.zeros(1, dtype=torch.complex128)
    portB = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    outA, outB = cross(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portB.grad is not None


def test_cross_gradients_both_ports():
    cross = Cross(CrossParams(number_of_splitters=1))
    portA = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    outA, outB = cross(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portB.grad is not None


def test_cross_gradients_with_crosstalk():
    cross = Cross(CrossParams(number_of_splitters=1, crosstalk=torch.tensor([0.05])))
    portA = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    portB = torch.rand(1, dtype=torch.complex128).requires_grad_(True)
    outA, outB = cross(portA, portB)
    (outA.real.sum() + outB.real.sum()).backward()
    assert portA.grad is not None
    assert portB.grad is not None


# ---------------------------------------------------------------------------
# Cross — multiple splitters (vectorised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 8])
def test_cross_vectorised_shape(n):
    cross = Cross(CrossParams(number_of_splitters=n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    outA, outB = cross(portA, portB)
    assert outA.shape == (n,)
    assert outB.shape == (n,)


@pytest.mark.parametrize("n", [2, 4])
def test_cross_vectorised_power_conservation(n):
    cross = Cross(CrossParams(number_of_splitters=n))
    portA = torch.rand(n, dtype=torch.complex128)
    portB = torch.rand(n, dtype=torch.complex128)
    p_in = _power(portA) + _power(portB)
    outA, outB = cross(portA, portB)
    p_out = _power(outA) + _power(outB)
    assert torch.isclose(p_in, p_out, rtol=1e-6)
