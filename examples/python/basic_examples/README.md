# Basic examples

Small, self-contained scripts that each demonstrate one piece of the LuxIA
Python API. Run any of them from the repository root with the venv active:

```bash
python examples/python/basic_examples/<script>.py
```

Scripts that import `luxia_lib` require the compiled extension; install it
first with `python -m pip install --no-build-isolation -e .`.

## IO components

| Script | What it shows |
|---|---|
| `io_example.py` | Instantiate `ElectricalToOptical` and `OpticalToElectrical` |
| `pdb_example.py` | Custom `OpticalToElectrical` subclass adding a trainable diagonal layer and bias on top of `IdealPhotodetector` |

## Photonic blocks and layers

| Script | What it shows |
|---|---|
| `block_example.py` | Build a custom `PhotonicBlock` from a `Heater` and an `Mmi`; run a single forward pass |
| `multiple_block_example.py` | Same pattern with configurable block count (`n_blocks`) |
| `layer_example.py` | Wrap a `ClementsBlock` in a `PhotonicLayer` |
| `tf_matrix_example.py` | Retrieve the transfer-function matrix of a custom block by combining heater phases and MMI matrices |

## Mesh topologies

| Script | What it shows |
|---|---|
| `mesh_example.py` | Assemble a `PhotonicMesh` from a list of `FldzhyanLayer` objects |
| `clements_example.py` | Clements mesh forward pass via the `luxia_lib` C++ backend (`requires luxia_lib`) |
| `fldzhyan_example.py` | Fldzhyan mesh forward pass via the `luxia_lib` C++ backend (`requires luxia_lib`) |

## Full accelerators (no training)

| Script | What it shows |
|---|---|
| `clements_bell_example.py` | End-to-end forward pass: `ElectricalToOptical` → `ClementsBellMesh` → `OpticalToElectrical` → `LxAccelerator` |
| `fldzhyan_bell_example.py` | Same pipeline with a `FldzhyanBellMesh`; note that Bell architectures require at least 3 layers |
| `braided_example.py` | Same pipeline with a `BraidedMesh` |
