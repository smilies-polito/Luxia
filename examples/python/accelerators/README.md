# Accelerator classes

Reusable `LxAccelerator` subclasses used by the trainable accelerator example
and the cross-platform weight-transfer scripts. These are library modules, not
standalone scripts — import them from other code rather than running them
directly.

## Classes

### `ClementsAccelerator` (`clements_accelerator.py`)

Clements-mesh classifier that mirrors the C++ `PhotonicAccelerator` exactly:

```
DAC → ElectricalToOptical → ClementsMesh → IdealPhotodetector → ADC
```

8-bit DAC on normalised `[0, 1]` inputs; 12-bit ADC with an upper reference
equal to `n_inputs` (power is conserved by a unitary mesh). Class scores are
the first `n_classes` output channels, matching the C++ convention.

### `FullClementsAccelerator` (`full_clements_accelerator.py`)

Extended version of `ClementsAccelerator` with configurable quantisation at
every stage (DAC, ADC, heater DACs) and inspection methods for SQNR, SNR, and
quantisation error per component. Adds `LaserModulator` between the DAC and
each mesh.

### `FldzhyanAccelerator` (`fldzhyan_accelerator.py`)

Fldzhyan-mesh classifier using `OE_PDB` (photodetector with trainable diagonal
gain and bias) as the optical-to-electrical converter.

### `FldzhyanBellAccelerator` (`fldzhyan_bell_accelerator.py`)

Fldzhyan Bell architecture with optional per-layer MMI insertion-loss and
imbalance tensors for non-ideal device modelling.

### `MyPhMeshAccelerator` (`my_ph_mesh_accelerator.py`)

Accelerator wrapping the custom mesh from `examples/python/custom_mesh/`.
Demonstrates how to drop a user-defined mesh topology into the standard
accelerator template.
