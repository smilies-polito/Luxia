"""
Tests for BraidMeshParams bindings and BraidMesh (C++).

BraidMesh topology: n_layers = 2*n_inputs - 3.
Even-indexed layers use ClementsLayerParams (with heaters);
odd-indexed layers use BraidLayerParams (passive).
"""
import pickle

import pytest
import torch

import luxia_lib
from luxia_lib import BraidMesh, BraidMeshParams
from utils.mesh_builders import make_braid_mesh_params, make_braid_mesh_py


# ---------------------------------------------------------------------------
# BraidMeshParams bindings (C++)
# ---------------------------------------------------------------------------

def test_braid_mesh_params_is_nn_module():
    assert isinstance(BraidMeshParams(), torch.nn.Module)


def test_braid_mesh_params_default():
    p = BraidMeshParams()
    assert isinstance(p, torch.nn.Module)


def test_braid_mesh_params_from_n_inputs():
    p = BraidMeshParams(number_of_inputs=4)
    assert p.number_of_inputs == 4


def test_braid_mesh_params_full():
    p = make_braid_mesh_params(4)
    assert p.number_of_layers == 2 * 4 - 3


def test_braid_mesh_params_repr():
    assert len(repr(BraidMeshParams(number_of_inputs=4))) > 0


def test_braid_mesh_params_forward_raises():
    with pytest.raises(TypeError):
        BraidMeshParams().forward(torch.zeros(2))


def test_braid_mesh_params_pickle():
    p = BraidMeshParams(number_of_inputs=4)
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, BraidMeshParams)


# ---------------------------------------------------------------------------
# BraidMesh — construction / layer count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 6])
def test_braid_mesh_layer_count(n):
    params = make_braid_mesh_params(n)
    assert params.number_of_layers == 2 * n - 3


# ---------------------------------------------------------------------------
# BraidMesh — forward / backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 6])
def test_braid_mesh_forward_shape(n):
    mesh = BraidMesh(make_braid_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert out.shape == x.shape
    assert torch.is_complex(out)


@pytest.mark.parametrize("n", [4, 6])
def test_braid_mesh_matmul_forward(n):
    mesh = BraidMesh(make_braid_mesh_params(n, is_matmul=True))
    x = torch.rand(n, dtype=torch.complex128)
    assert mesh.matmul_forward(x).shape == x.shape


@pytest.mark.parametrize("n", [4, 6])
def test_braid_mesh_backward(n):
    mesh = BraidMesh(make_braid_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    mesh.forward(x).real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Power conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_braid_mesh_power_conservation(n):
    mesh = BraidMesh(make_braid_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# Self-consistency: two instances synced to same phases → same output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_braid_mesh_self_consistency(n):
    """Two BraidMesh instances with the same params produce the same output
    after syncing their phase states via get/set_layer_phase_shifts.
    """
    params = make_braid_mesh_params(n)
    mesh_a = BraidMesh(params)
    mesh_b = BraidMesh(params)

    phases = mesh_a.get_layer_phase_shifts()
    mesh_b.set_phase_shifts(phases)

    x = torch.rand(n, dtype=torch.complex128)
    assert torch.allclose(mesh_a.forward(x), mesh_b.forward(x), atol=1e-10)


# ---------------------------------------------------------------------------
# CUDA device placement
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n", [4, 6])
def test_braid_mesh_cuda_device(n):
    mesh = BraidMesh(make_braid_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128, device="cuda")
    out = mesh.forward(x)
    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# C++ vs Python forward equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_braid_cpp_vs_python_forward(n):
    """C++ BraidMesh and Python BraidedMesh produce the same output when given
    the same phase configuration."""
    torch.manual_seed(42)
    cpp_mesh = BraidMesh(make_braid_mesh_params(n))
    py_mesh = make_braid_mesh_py(n)

    random_phases = [torch.rand_like(p) * 2 * torch.pi
                     for p in cpp_mesh.get_layer_phase_shifts()]
    cpp_mesh.set_phase_shifts(random_phases)
    py_mesh.set_phase_shifts(cpp_mesh.get_layer_phase_shifts())

    x = torch.rand(n, dtype=torch.complex128)
    cpp_out = cpp_mesh.forward(x)
    py_out = py_mesh.forward(x.unsqueeze(0)).squeeze(0)

    assert torch.allclose(cpp_out, py_out, atol=1e-10)
