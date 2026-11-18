"""
Python training → C++ inference cross-platform example (aligned split).

Trains a Clements-mesh photonic classifier on the Iris dataset using PyTorch
autograd, then saves the learned heater phases in LUXIA_WEIGHTS_V1 CSV format
so that the C++ evaluate binary can load and test them without any Python
dependency.

Data split mirrors examples/cpp/trainable_accelerator exactly:
  - Source: data/iris/iris.data (same file used by C++ train/evaluate)
  - Normalization: fixed per-feature min/max matching C++ IrisDataset constants
  - Split: stratified 70/15/15 with seed=42

Signal chain (identical to examples/cpp/trainable_accelerator):
    DAC (8-bit) -> ElectricalToOptical -> ClementsMesh -> |E|^2 -> ADC (12-bit)
"""

import csv
import math
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from accelerators.clements_accelerator import ClementsAccelerator

# ---------------------------------------------------------------------------
# Config — mirrors examples/cpp/trainable_accelerator/config.hh
# ---------------------------------------------------------------------------
N_INPUTS     = 4
N_LAYERS     = 4
N_CLASSES    = 3
EPOCHS       = 250
BATCH_SIZE   = 16
LR           = 0.01
SPLIT_SEED   = 42
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15

_DIR = os.path.dirname(os.path.abspath(__file__))
IRIS_PATH    = os.path.join(_DIR, '../../data/iris/iris.data')
WEIGHTS_PATH = os.path.join(_DIR, 'trained_weights.csv')

# Fixed per-feature min/max matching C++ IrisDataset::make_sample
_MINIMA = [4.3, 2.0, 1.0, 0.1]
_MAXIMA = [7.9, 4.4, 6.9, 2.5]
_LABEL_MAP = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_iris_by_class(path):
    by_class = {0: [], 1: [], 2: []}
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 5 or not row[0].strip():
                continue
            feat = [(float(row[i]) - _MINIMA[i]) / (_MAXIMA[i] - _MINIMA[i])
                    for i in range(4)]
            label = _LABEL_MAP[row[4].strip()]
            by_class[label].append(feat)
    return by_class


def make_stratified_split(by_class, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Stratified split matching C++ make_stratified_splits strategy.
    Uses Python random.Random (MT19937) with shared state across classes.
    Per class: floor(n * train_frac) train, floor(n * val_frac) val, rest test.
    """
    rng = random.Random(seed)
    train_X, train_y = [], []
    val_X,   val_y   = [], []
    test_X,  test_y  = [], []
    for label, samples in sorted(by_class.items()):
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(math.floor(n * train_frac))
        n_val   = int(math.floor(n * val_frac))
        for feat in shuffled[:n_train]:
            train_X.append(feat); train_y.append(label)
        for feat in shuffled[n_train:n_train + n_val]:
            val_X.append(feat); val_y.append(label)
        for feat in shuffled[n_train + n_val:]:
            test_X.append(feat); test_y.append(label)
    return (torch.tensor(train_X), torch.tensor(train_y, dtype=torch.long),
            torch.tensor(val_X),   torch.tensor(val_y,   dtype=torch.long),
            torch.tensor(test_X),  torch.tensor(test_y,  dtype=torch.long))


# ---------------------------------------------------------------------------
# Reproducibility + data
# ---------------------------------------------------------------------------
torch.manual_seed(SPLIT_SEED)

by_class = load_iris_by_class(IRIS_PATH)
train_X, train_y, val_X, val_y, test_X, test_y = make_stratified_split(
    by_class, TRAIN_FRAC, VAL_FRAC, SPLIT_SEED)

n_train, n_val, n_test = len(train_X), len(val_X), len(test_X)
print(f"Split: {n_train} train / {n_val} val / {n_test} test")
print(f"  train classes: {[(train_y == c).sum().item() for c in range(3)]}")
print(f"  val   classes: {[(val_y   == c).sum().item() for c in range(3)]}")
print(f"  test  classes: {[(test_y  == c).sum().item() for c in range(3)]}")

train_loader = DataLoader(TensorDataset(train_X, train_y),
                          batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model     = ClementsAccelerator(n_inputs=N_INPUTS, n_layers=N_LAYERS,
                                n_classes=N_CLASSES, n_networks=1)
loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nTraining: batch={BATCH_SIZE}  lr={LR}  epochs={EPOCHS}  "
      f"params={sum(p.numel() for p in model.parameters())}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(xb)
        train_correct += (out.argmax(1) == yb).sum().item()

    if epoch % 25 == 0:
        model.eval()
        with torch.no_grad():
            val_out     = model(val_X)
            val_correct = (val_out.argmax(1) == val_y).sum().item()
        print(f"epoch {epoch:3d}/{EPOCHS}"
              f"  train loss {train_loss/n_train:.5f}"
              f"  acc {train_correct/n_train:.2%}"
              f"  |  val acc {val_correct/n_val:.2%}")

# ---------------------------------------------------------------------------
# Test accuracy (same split as C++ evaluate)
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_out     = model(test_X)
    test_correct = (test_out.argmax(1) == test_y).sum().item()
print(f"\nIris test accuracy (Python eval): "
      f"{test_correct}/{n_test} ({100.0 * test_correct / n_test:.4f}%)")

# ---------------------------------------------------------------------------
# Save weights — LUXIA_WEIGHTS_V1 format read by C++ evaluate
# ---------------------------------------------------------------------------
mesh    = model._ph_model['lx_module_2']   # DAC=0, E2O=1, Mesh=2
weights = mesh.get_layer_phase_shifts()
max_n   = max(w.numel() for w in weights)

with open(WEIGHTS_PATH, 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['LUXIA_WEIGHTS_V1', len(weights), max_n])
    for w in weights:
        flat = w.detach().cpu().contiguous().reshape(-1).tolist()
        n    = len(flat)
        writer.writerow([n] + flat + [0.0] * (max_n - n))

print(f"Saved weights to {WEIGHTS_PATH}")
