# Custom mesh

Shows how to define a new photonic mesh topology by subclassing the three
LuxIA building blocks: `PhotonicBlock`, `PhotonicLayer`, and `PhotonicMesh`.

## `my_ph_mesh.py`

Implements `MyCostumBlock → MyCostumeLayer → MyCostumeMesh`, which
reproduces the Fldzhyan architecture from scratch as a worked example of the
extension pattern:

- **`MyCostumBlock`** — subclasses `PhotonicBlock`; wires a `Heater` into an
  `Mmi` and exposes `get_tf_matrix()` by multiplying the heater phase-diagonal
  with the MMI transfer matrix.
- **`MyCostumeLayer`** — subclasses `PhotonicLayer`; computes the block count
  for even/odd layers and passes per-layer MMI impairment tensors down to
  `MyCostumBlock`.
- **`MyCostumeMesh`** — subclasses `PhotonicMesh`; stacks alternating even/odd
  layers and forwards optional per-layer insertion-loss and imbalance lists.

The ASCII diagrams in the class docstrings show the waveguide connectivity for
both even and odd input counts.

## Using the custom mesh

Import `MyCostumeMesh` and pass it to any `LxAccelerator` the same way a
built-in mesh would be used. The accelerator wrapper lives in
`examples/python/accelerators/my_ph_mesh_accelerator.py`.
