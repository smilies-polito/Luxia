"""
Tests for FldzhyanMeshParams bindings and FldzhyanMesh (C++).
"""
import pickle

import pytest
import torch

import luxia_lib
from luxia_lib import FldzhyanMesh, FldzhyanMeshParams
from luxia.photonic_meshes.fldzhyan.fldzhyan_mesh import FldzhyanMesh as PyFldzhyanMesh
from utils.mesh_builders import make_fldzhyan_mesh_params, make_fldzhyan_mesh_py


# ---------------------------------------------------------------------------
# FldzhyanMeshParams bindings (C++)
# ---------------------------------------------------------------------------

def test_fldzhyan_mesh_params_is_nn_module():
    assert isinstance(FldzhyanMeshParams(), torch.nn.Module)


def test_fldzhyan_mesh_params_default():
    p = FldzhyanMeshParams()
    assert isinstance(p, torch.nn.Module)


def test_fldzhyan_mesh_params_from_n_inputs():
    p = FldzhyanMeshParams(number_of_inputs=4)
    assert p.number_of_inputs == 4


def test_fldzhyan_mesh_params_full():
    p = make_fldzhyan_mesh_params(4)
    assert p.number_of_layers == 8


def test_fldzhyan_mesh_params_repr():
    assert len(repr(FldzhyanMeshParams(number_of_inputs=4))) > 0


def test_fldzhyan_mesh_params_forward_raises():
    with pytest.raises(TypeError):
        FldzhyanMeshParams().forward(torch.zeros(2))


def test_fldzhyan_mesh_params_pickle():
    p = FldzhyanMeshParams(number_of_inputs=4)
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, FldzhyanMeshParams)


# ---------------------------------------------------------------------------
# FldzhyanMesh — construction / layer count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_layer_count(n):
    params = make_fldzhyan_mesh_params(n)
    assert params.number_of_layers == 2 * n


# ---------------------------------------------------------------------------
# FldzhyanMesh — forward / backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_forward_shape(n):
    mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert out.shape == x.shape
    assert torch.is_complex(out)


@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_matmul_forward(n):
    mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n, is_matmul=True))
    x = torch.rand(n, dtype=torch.complex128)
    assert mesh.matmul_forward(x).shape == x.shape


@pytest.mark.parametrize("n", [4])
def test_fldzhyan_mesh_backward(n):
    mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    mesh.forward(x).real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Power conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_power_conservation(n):
    mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# Self-consistency: two instances with same phases produce same output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_self_consistency(n):
    params = make_fldzhyan_mesh_params(n)
    mesh_a = FldzhyanMesh(params)
    mesh_b = FldzhyanMesh(params)

    torch.manual_seed(13)
    phases = [torch.rand_like(p) * 2 * torch.pi for p in mesh_a.get_layer_phase_shifts()]
    mesh_a.set_phase_shifts(phases)
    mesh_b.set_phase_shifts(phases)

    x = torch.rand(n, dtype=torch.complex128)
    assert torch.allclose(mesh_a.forward(x), mesh_b.forward(x), atol=1e-10)


@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_different_phases_differ(n):
    """Different phase configurations produce different outputs."""
    params = make_fldzhyan_mesh_params(n)
    mesh_a = FldzhyanMesh(params)
    mesh_b = FldzhyanMesh(params)

    phases_a = [torch.zeros_like(p) for p in mesh_a.get_layer_phase_shifts()]
    phases_b = [torch.ones_like(p) for p in mesh_a.get_layer_phase_shifts()]
    mesh_a.set_phase_shifts(phases_a)
    mesh_b.set_phase_shifts(phases_b)

    x = torch.rand(n, dtype=torch.complex128)
    assert not torch.allclose(mesh_a.forward(x), mesh_b.forward(x), atol=1e-6)


# ---------------------------------------------------------------------------
# CUDA device placement
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_mesh_cuda_device(n):
    mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128, device="cuda")
    out = mesh.forward(x)
    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# C++ vs Python forward equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_fldzhyan_cpp_vs_python_forward(n):
    """C++ FldzhyanMesh and Python FldzhyanMesh produce the same output when
    given the same phase configuration."""
    torch.manual_seed(42)
    cpp_mesh = FldzhyanMesh(make_fldzhyan_mesh_params(n))
    py_mesh = make_fldzhyan_mesh_py(n)

    random_phases = [torch.rand_like(p) * 2 * torch.pi
                     for p in cpp_mesh.get_layer_phase_shifts()]
    cpp_mesh.set_phase_shifts(random_phases)
    py_mesh.set_phase_shifts(cpp_mesh.get_layer_phase_shifts())

    x = torch.rand(n, dtype=torch.complex128)
    cpp_out = cpp_mesh.forward(x)
    py_out = py_mesh.forward(x.unsqueeze(0)).squeeze(0)

    assert torch.allclose(cpp_out, py_out, atol=1e-10)
