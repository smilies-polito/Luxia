"""
Shared parameter builders used by photonic_meshes/ and photonic_layers/ tests.

C++ param builders (make_*) return luxia_lib params suitable for C++ meshes.
Python param builders (make_*_py) return luxia.params dataclasses for PyClementsMesh.
"""
from luxia_lib import (
    HeaterParams,
    MmiParams,
    CrossParams,
    ClementsBlockParams,
    ClementsLayerParams,
    ClementsMeshParams,
    FldzhyanBlockParams,
    FldzhyanLayerParams,
    FldzhyanMeshParams,
    BraidBlockParams,
    BraidLayerParams,
    BraidMeshParams,
)
from luxia.params import (
    HeaterParams as PyHeaterParams,
    MmiParams as PyMmiParams,
    ClementsBlockParams as PyClementsBlockParams,
    ClementsLayerParams as PyClementsLayerParams,
    ClementsMeshParams as PyClementsMeshParams,
)
from luxia.photonic_meshes.clements.clements_mesh import (
    ClementsMesh as PyClementsMesh,
)
from luxia.photonic_meshes.braided.braided_mesh import BraidedMesh as PyBraidedMesh
from luxia.photonic_meshes.fldzhyan.fldzhyan_mesh import FldzhyanMesh as PyFldzhyanMesh


# ---------------------------------------------------------------------------
# Clements
# ---------------------------------------------------------------------------

def make_clements_layer_params(n_inputs: int, is_even_layer: bool) -> ClementsLayerParams:
    n_blocks = n_inputs // 2 if is_even_layer else (n_inputs - 1) // 2
    blk = ClementsBlockParams(
        number_of_blocks=n_blocks,
        heater_params=[HeaterParams(number_of_heaters=n_blocks),
                       HeaterParams(number_of_heaters=n_blocks)],
        mmi_params=[MmiParams(number_of_splitters=n_blocks),
                    MmiParams(number_of_splitters=n_blocks)],
    )
    return ClementsLayerParams(
        number_of_inputs=n_inputs,
        is_even_layer=is_even_layer,
        clements_blk_params=blk,
    )


def make_clements_mesh_params(n_inputs: int, is_matmul: bool = False) -> ClementsMeshParams:
    n_layers = n_inputs
    layer_params = [make_clements_layer_params(n_inputs, i % 2 == 0) for i in range(n_layers)]
    return ClementsMeshParams(
        number_of_inputs=n_inputs,
        number_of_layers=n_layers,
        layer_params=layer_params,
        is_matmul=is_matmul,
    )


# ---------------------------------------------------------------------------
# Fldzhyan
# ---------------------------------------------------------------------------

def make_fldzhyan_layer_params(n_inputs: int, is_even_layer: bool) -> FldzhyanLayerParams:
    n_blocks = n_inputs // 2 if is_even_layer else (n_inputs - 1) // 2
    blk = FldzhyanBlockParams(
        number_of_blocks=n_blocks,
        heater_params=HeaterParams(number_of_heaters=n_blocks),
        mmi_params=MmiParams(number_of_splitters=n_blocks),
    )
    return FldzhyanLayerParams(
        number_of_inputs=n_inputs,
        is_even_layer=is_even_layer,
        fldzhyan_blk_params=blk,
    )


def make_fldzhyan_mesh_params(n_inputs: int, is_matmul: bool = False) -> FldzhyanMeshParams:
    n_layers = 2 * n_inputs
    layer_params = [make_fldzhyan_layer_params(n_inputs, i % 2 == 0) for i in range(n_layers)]
    return FldzhyanMeshParams(
        number_of_inputs=n_inputs,
        number_of_layers=n_layers,
        layer_params=layer_params,
        is_matmul=is_matmul,
    )


# ---------------------------------------------------------------------------
# Braid
# ---------------------------------------------------------------------------

def make_braid_layer_params(n_inputs: int, layer_index: int):
    is_even = (layer_index % 2 == 0)
    if is_even:
        n_blocks = n_inputs // 2
        blk = ClementsBlockParams(
            number_of_blocks=n_blocks,
            heater_params=[HeaterParams(number_of_heaters=n_blocks),
                           HeaterParams(number_of_heaters=n_blocks)],
            mmi_params=[MmiParams(number_of_splitters=n_blocks),
                        MmiParams(number_of_splitters=n_blocks)],
        )
        return ClementsLayerParams(
            number_of_inputs=n_inputs,
            is_even_layer=True,
            clements_blk_params=blk,
        )
    else:
        n_blocks = (n_inputs - 1) // 2
        return BraidLayerParams(
            number_of_inputs=n_inputs,
            is_even_layer=False,
            braid_block_params=BraidBlockParams(number_of_blocks=n_blocks),
        )


def make_braid_mesh_params(n_inputs: int, is_matmul: bool = False) -> BraidMeshParams:
    n_layers = 2 * n_inputs - 3
    layer_params = [make_braid_layer_params(n_inputs, i) for i in range(n_layers)]
    return BraidMeshParams(
        number_of_inputs=n_inputs,
        number_of_layers=n_layers,
        layer_params=layer_params,
        is_matmul=is_matmul,
    )


# ---------------------------------------------------------------------------
# Python-params Clements builders (for PyClementsMesh)
# ---------------------------------------------------------------------------

def make_clements_layer_params_py(n_inputs: int, is_even_layer: bool) -> PyClementsLayerParams:
    n_blocks = n_inputs // 2 if is_even_layer else (n_inputs - 1) // 2
    blk = PyClementsBlockParams(
        number_of_blocks=n_blocks,
        heater_params=[PyHeaterParams(number_of_heaters=n_blocks),
                       PyHeaterParams(number_of_heaters=n_blocks)],
        mmi_params=[PyMmiParams(number_of_splitters=n_blocks),
                    PyMmiParams(number_of_splitters=n_blocks)],
    )
    return PyClementsLayerParams(
        number_of_inputs=n_inputs,
        is_even_layer=is_even_layer,
        clements_blk_params=blk,
    )


def make_clements_mesh_params_py(n_inputs: int, is_matmul: bool = False) -> PyClementsMeshParams:
    n_layers = n_inputs
    layer_params = [make_clements_layer_params_py(n_inputs, i % 2 == 0) for i in range(n_layers)]
    return PyClementsMeshParams(
        number_of_inputs=n_inputs,
        number_of_layers=n_layers,
        layer_params=layer_params,
        is_matmul=is_matmul,
    )


# ---------------------------------------------------------------------------
# Python BraidedMesh builder
# ---------------------------------------------------------------------------

def make_fldzhyan_mesh_py(n_inputs: int, is_matmul: bool = False) -> PyFldzhyanMesh:
    n_layers = 2 * n_inputs
    return PyFldzhyanMesh(
        n_layers=n_layers,
        n_inputs=n_inputs,
        is_matmul=is_matmul,
    )


def make_braid_mesh_py(n_inputs: int, is_matmul: bool = False) -> PyBraidedMesh:
    n_layers = 2 * n_inputs - 3
    return PyBraidedMesh(
        n_inputs=n_inputs,
        n_layers=n_layers,
        is_matmul=is_matmul,
    )


# ---------------------------------------------------------------------------
# Phase sync helper (C++ ClementsMesh → Python ClementsMesh)
# ---------------------------------------------------------------------------

def sync_phases(cpp_mesh, py_mesh: PyClementsMesh) -> None:
    """Copy phase shifts from a C++ ClementsMesh into every layer of a Python one."""
    cpp_phases = cpp_mesh.get_layer_phase_shifts()
    for layer, phases in zip(py_mesh.get_photonic_layers(), cpp_phases):
        layer.set_phase_shifts(phases)
