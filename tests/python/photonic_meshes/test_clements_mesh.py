"""
Tests for ClementsMeshParams bindings and ClementsMesh (C++ + Python).

Known pre-existing issue: Python ClementsLayer.set_phase_shifts() does not
exist, so tests that sync phases from C++ → Python mesh are marked xfail.
"""
import pickle
import time

import pytest
import torch

import luxia_lib
from luxia_lib import ClementsMesh, ClementsMeshParams
from luxia.photonic_meshes.clements.clements_mesh import ClementsMesh as PyClementsMesh
from utils.mesh_builders import make_clements_mesh_params, make_clements_mesh_params_py


# ---------------------------------------------------------------------------
# ClementsMeshParams bindings (C++)
# ---------------------------------------------------------------------------

def test_clements_mesh_params_is_nn_module():
    assert isinstance(ClementsMeshParams(), torch.nn.Module)


def test_clements_mesh_params_default():
    p = ClementsMeshParams()
    assert isinstance(p, torch.nn.Module)


def test_clements_mesh_params_from_n_inputs():
    p = ClementsMeshParams(number_of_inputs=4)
    assert p.number_of_inputs == 4


def test_clements_mesh_params_full():
    p = make_clements_mesh_params(4)
    assert p.number_of_layers == 4


def test_clements_mesh_params_repr():
    assert len(repr(ClementsMeshParams(number_of_inputs=4))) > 0


def test_clements_mesh_params_forward_raises():
    with pytest.raises(TypeError):
        ClementsMeshParams().forward(torch.zeros(2))


def test_clements_mesh_params_pickle():
    p = ClementsMeshParams(number_of_inputs=4)
    q = pickle.loads(pickle.dumps(p))
    assert isinstance(q, ClementsMeshParams)


# ---------------------------------------------------------------------------
# ClementsMesh (C++) — construction / layer count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8])
def test_clements_mesh_layer_count(n):
    params = make_clements_mesh_params(n)
    assert params.number_of_layers == n


# ---------------------------------------------------------------------------
# ClementsMesh (C++) — forward / backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_clements_mesh_forward_shape(n):
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert out.shape == x.shape
    assert torch.is_complex(out)


@pytest.mark.parametrize("n", [4, 6])
def test_clements_mesh_matmul_forward(n):
    mesh = ClementsMesh(make_clements_mesh_params(n, is_matmul=True))
    x = torch.rand(n, dtype=torch.complex128)
    assert mesh.matmul_forward(x).shape == x.shape


@pytest.mark.parametrize("n", [4])
def test_clements_mesh_backward(n):
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    mesh.forward(x).real.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Power conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_clements_mesh_power_conservation(n):
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    out = mesh.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# Phase shift roundtrip
# ---------------------------------------------------------------------------

def test_clements_mesh_phase_shift_roundtrip():
    mesh = ClementsMesh(make_clements_mesh_params(4))
    phases = [torch.zeros_like(p) for p in mesh.get_layer_phase_shifts()]
    mesh.set_phase_shifts(phases)
    for retrieved, expected in zip(mesh.get_layer_phase_shifts(), phases):
        assert torch.allclose(retrieved, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Insertion loss reduces power
# ---------------------------------------------------------------------------

def test_clements_mesh_insertion_loss_reduces_power():
    """Mesh with insertion_loss=0.4 on every MMI outputs less power than ideal."""
    from luxia_lib import ClementsBlockParams, ClementsLayerParams, HeaterParams, MmiParams

    n = 4
    ideal_params = make_clements_mesh_params(n)

    n_layers = n
    layer_params = []
    for i in range(n_layers):
        is_even = (i % 2 == 0)
        n_blocks = n // 2 if is_even else (n - 1) // 2
        loss_t = torch.tensor([0.4])
        blk = ClementsBlockParams(
            number_of_blocks=n_blocks,
            heater_params=[HeaterParams(number_of_heaters=n_blocks),
                           HeaterParams(number_of_heaters=n_blocks)],
            mmi_params=[MmiParams(number_of_splitters=n_blocks, insertion_loss=loss_t),
                        MmiParams(number_of_splitters=n_blocks, insertion_loss=loss_t)],
        )
        layer_params.append(ClementsLayerParams(
            number_of_inputs=n, is_even_layer=is_even, clements_blk_params=blk))
    lossy_params = ClementsMeshParams(
        number_of_inputs=n, number_of_layers=n_layers, layer_params=layer_params)

    mesh_ideal = ClementsMesh(ideal_params)
    mesh_lossy = ClementsMesh(lossy_params)

    torch.manual_seed(7)
    phases = [torch.rand_like(p) * 2 * torch.pi for p in mesh_ideal.get_layer_phase_shifts()]
    mesh_ideal.set_phase_shifts(phases)
    mesh_lossy.set_phase_shifts(phases)

    x = torch.rand(n, dtype=torch.complex128)
    p_in = (x.abs() ** 2).sum()

    p_out_ideal = (mesh_ideal.forward(x).abs() ** 2).sum()
    p_out_lossy = (mesh_lossy.forward(x).abs() ** 2).sum()

    assert torch.isclose(p_out_ideal, p_in, rtol=1e-5)
    assert p_out_lossy < p_out_ideal


# ---------------------------------------------------------------------------
# Gradient non-triviality (C++)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6])
def test_clements_mesh_gradient_non_trivial(n):
    torch.manual_seed(99)
    mesh = ClementsMesh(make_clements_mesh_params(n))

    x1 = torch.rand(n, dtype=torch.complex128).requires_grad_(True)
    mesh.forward(x1).real.sum().backward()
    grad1 = x1.grad.clone()

    assert grad1 is not None and grad1.shape == (n,)
    assert grad1.abs().sum() > 0

    mesh.set_phase_shifts([torch.rand_like(p) * 2 * torch.pi
                           for p in mesh.get_layer_phase_shifts()])
    x2 = x1.detach().requires_grad_(True)
    mesh.forward(x2).real.sum().backward()
    grad2 = x2.grad

    assert not torch.allclose(grad1, grad2, atol=1e-6)


# ---------------------------------------------------------------------------
# CUDA device placement
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n", [4, 6])
def test_clements_mesh_cuda_device(n):
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128, device="cuda")
    out = mesh.forward(x)
    assert out.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clements_mesh_cuda_power_conservation():
    n = 4
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128, device="cuda")
    out = mesh.forward(x)
    assert torch.isclose((x.abs() ** 2).sum(), (out.abs() ** 2).sum(), rtol=1e-5)


# ---------------------------------------------------------------------------
# C++ vs Python ClementsMesh
# Note: requires Python ClementsLayer.set_phase_shifts() — pre-existing bug.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 6, 8])
def test_clements_cpp_vs_python_forward(n):
    torch.manual_seed(42)
    cpp_mesh = ClementsMesh(make_clements_mesh_params(n))
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n))

    random_phases = [torch.rand_like(p) * 2 * torch.pi
                     for p in cpp_mesh.get_layer_phase_shifts()]
    cpp_mesh.set_phase_shifts(random_phases)
    py_mesh.set_phase_shifts(cpp_mesh.get_layer_phase_shifts())

    x = torch.rand(n, dtype=torch.complex128)
    cpp_fwd = cpp_mesh.forward(x)
    py_fwd = py_mesh.forward(x.unsqueeze(0)).squeeze(0)

    assert torch.allclose(cpp_fwd, py_fwd, atol=1e-10)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_clements_cpp_vs_python_matmul_forward(n):
    torch.manual_seed(42)
    cpp_mesh = ClementsMesh(make_clements_mesh_params(n, is_matmul=True))
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n, is_matmul=True))

    cpp_mesh.set_phase_shifts([torch.rand_like(p) * 2 * torch.pi
                               for p in cpp_mesh.get_layer_phase_shifts()])
    py_mesh.set_phase_shifts(cpp_mesh.get_layer_phase_shifts())

    x = torch.rand(n, dtype=torch.complex128)
    assert torch.allclose(cpp_mesh.matmul_forward(x),
                          py_mesh.matmul_forward(x.unsqueeze(0)).squeeze(0), atol=1e-10)


# ---------------------------------------------------------------------------
# Batched PyClementsMesh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,n", [(4, 4), (8, 6)])
def test_py_clements_batched_forward_shape(B, n):
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n))
    x = torch.rand(B, n, dtype=torch.complex128)
    out = py_mesh.forward(x)
    assert out.shape == (B, n)
    assert torch.is_complex(out)


@pytest.mark.parametrize("B,n", [(4, 4), (8, 6)])
def test_py_clements_batched_backward(B, n):
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n))
    x = torch.rand(B, n, dtype=torch.complex128).requires_grad_(True)
    py_mesh.forward(x).real.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("B,n", [(4, 4)])
def test_py_clements_batched_power_conservation(B, n):
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n))
    x = torch.rand(B, n, dtype=torch.complex128)
    out = py_mesh.forward(x)
    assert torch.allclose((x.abs() ** 2).sum(-1), (out.abs() ** 2).sum(-1), rtol=1e-5)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,reps", [(4, 100), (8, 100), (16, 50)])
def test_clements_mesh_forward_speed(n, reps, capsys):
    mesh = ClementsMesh(make_clements_mesh_params(n))
    x = torch.rand(n, dtype=torch.complex128)
    for _ in range(5):
        mesh.forward(x)

    t0 = time.perf_counter()
    for _ in range(reps):
        mesh.forward(x)
    avg_us = (time.perf_counter() - t0) / reps * 1e6

    with capsys.disabled():
        print(f"\n[speed] ClementsMesh n={n}: {avg_us:.1f} µs/call")
    assert avg_us < 10_000


@pytest.mark.parametrize("n,reps", [(4, 100), (8, 100)])
def test_cpp_vs_python_speed(n, reps, capsys):
    cpp_mesh = ClementsMesh(make_clements_mesh_params(n))
    py_mesh = PyClementsMesh(make_clements_mesh_params_py(n))
    x_1d = torch.rand(n, dtype=torch.complex128)
    x_2d = x_1d.unsqueeze(0)

    for _ in range(5):
        cpp_mesh.forward(x_1d)
        py_mesh.forward(x_2d)

    t0 = time.perf_counter()
    for _ in range(reps):
        cpp_mesh.forward(x_1d)
    t_cpp = (time.perf_counter() - t0) / reps * 1e6

    t0 = time.perf_counter()
    for _ in range(reps):
        py_mesh.forward(x_2d)
    t_py = (time.perf_counter() - t0) / reps * 1e6

    with capsys.disabled():
        print(f"\n[speed] n={n}: C++={t_cpp:.1f} µs, Python={t_py:.1f} µs")
