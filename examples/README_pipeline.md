# Photonic pipeline examples

Examples are grouped by language under `examples/python/` and `examples/cpp/`.
The shared Iris dataset lives in `examples/data/iris/iris.data` (downloaded
on first use; excluded from Git by `.gitignore`).

---

## Python API exploration

Short, self-contained scripts for exploring individual LuxIA components and
mesh topologies without running a full training pipeline.

- [Basic examples](python/basic_examples/) — IO components, custom blocks,
  mesh topologies, and minimal accelerator forward passes
- [Accelerator classes](python/accelerators/) — reusable `LxAccelerator`
  subclasses (Clements, Fldzhyan, Fldzhyan Bell, full configurable version)
- [Custom mesh](python/custom_mesh/) — step-by-step walkthrough of defining a
  new topology by subclassing `PhotonicBlock`, `PhotonicLayer`, and
  `PhotonicMesh`

---

## Basic accelerator

Runs a random input through the full signal chain: DAC → laser modulator →
one-layer Clements mesh → ideal photodetector → ADC, then prints the gradient
of the summed ADC output with respect to the input.

**Python**

```bash
python examples/python/basic_accelerator/run.py
```

**C++ with LuxiaTensor** (pure C++, no PyTorch dependency)

```bash
bash examples/cpp/basic_accelerator/run.sh
```

**C++ with LibTorch** (requires PyTorch in the active venv)

```bash
bash examples/cpp/basic_accelerator/run_libtorch.sh
```

---

## Trainable Clements accelerator — Iris classification

Trains a four-layer Clements mesh to classify Iris flowers from four features
into three species. The only trainable parameters are the heater phase shifts.
Weights are saved in the portable `LUXIA_WEIGHTS_V1` CSV format and can be
exchanged between Python and C++.

### Python

```bash
# Train and save weights
python examples/python/trainable_accelerator/train.py

# Evaluate (defaults to Python-trained weights)
python examples/python/trainable_accelerator/evaluate.py

# Evaluate with weights from another source
python examples/python/trainable_accelerator/evaluate.py path/to/weights.csv
```

### C++

Training requires the LibTorch backend; evaluation uses the native LuxiaTensor
backend. Both scripts build the example automatically.

```bash
# Train (LibTorch backend, requires PyTorch in active venv)
bash examples/cpp/trainable_accelerator/run_train.sh

# Evaluate (LuxiaTensor backend, no PyTorch dependency)
bash examples/cpp/trainable_accelerator/run_inference.sh
```

→ [Detailed README](cpp/trainable_accelerator/README.md): signal chain, file
map, configuration constants, and weight format.

---

## Cross-platform weight transfer

Demonstrates that `LUXIA_WEIGHTS_V1` files produced by either backend load
and produce identical results in the other. Runs all four combinations:

- C++ weights → C++ evaluate
- Python weights → C++ evaluate
- C++ weights → Python evaluate
- Python weights → Python evaluate

```bash
bash examples/cross_platform/run.sh
```

Requires trained weights from both backends. Run the Python and C++ training
steps above first, then run this script.

---

## C++ smoke test

Verifies that the LibTorch C++ extension links and executes correctly:

```bash
bash examples/cpp/torch_smoke_test/run_torch_smoke_test.sh
```
