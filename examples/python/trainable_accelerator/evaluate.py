"""
Python evaluation of a trained weights file on the aligned Iris test split.

Loads a LUXIA_WEIGHTS_V1 CSV (from either Python or C++ training), builds
a ClementsAccelerator with the same signal chain as C++ evaluate, and reports
test accuracy on the same stratified 70/15/15 test split used by train.py.

Usage:
    python evaluate.py [weights_file]

Default: trained_weights.csv (Python-trained weights in this directory).
C++ weights: python evaluate.py ../../cpp/trainable_accelerator/trained_weights.csv
"""

import csv
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from accelerators.clements_accelerator import ClementsAccelerator

# ---------------------------------------------------------------------------
# Config — must match train.py and config.hh exactly
# ---------------------------------------------------------------------------
N_INPUTS   = 4
N_LAYERS   = 4
N_CLASSES  = 3
SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

_DIR = os.path.dirname(os.path.abspath(__file__))
IRIS_PATH = os.path.join(_DIR, '../../data/iris/iris.data')

_MINIMA   = [4.3, 2.0, 1.0, 0.1]
_MAXIMA   = [7.9, 4.4, 6.9, 2.5]
_LABEL_MAP = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
_LABEL_NAMES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# ---------------------------------------------------------------------------
# Shared split helpers (identical to train.py)
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


def make_test_split(by_class, train_frac=0.70, val_frac=0.15, seed=42):
    rng = random.Random(seed)
    test_X, test_y = [], []
    for label, samples in sorted(by_class.items()):
        shuffled = samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(math.floor(n * train_frac))
        n_val   = int(math.floor(n * val_frac))
        for feat in shuffled[n_train + n_val:]:
            test_X.append(feat); test_y.append(label)
    return (torch.tensor(test_X),
            torch.tensor(test_y, dtype=torch.long))


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_weights_csv(path):
    """Return list of float tensors, one per layer (flattened)."""
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0] == 'LUXIA_WEIGHTS_V1', f"Unknown format: {header[0]}"
        n_layers = int(header[1])
        weights = []
        for _ in range(n_layers):
            row   = next(reader)
            count = int(row[0])
            vals  = [float(v) for v in row[1:count + 1]]
            weights.append(torch.tensor(vals, dtype=torch.float32))
    return weights


def set_mesh_weights(mesh, weight_list):
    layers = mesh._ph_layers
    assert len(weight_list) == len(layers), (
        f"Layer count mismatch: file has {len(weight_list)}, "
        f"mesh has {len(layers)}")
    for wrapper, flat_w in zip(layers, weight_list):
        current  = wrapper._cpp_layer.get_phase_shifts()
        reshaped = flat_w.reshape(current.shape)
        wrapper._cpp_layer.set_phase_shifts(reshaped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    weights_file = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_DIR, 'trained_weights.csv')
    weights_file = os.path.abspath(weights_file)

    print(f"Weights : {weights_file}")
    print(f"Dataset : {os.path.abspath(IRIS_PATH)}")

    by_class = load_iris_by_class(IRIS_PATH)
    test_X, test_y = make_test_split(by_class, TRAIN_FRAC, VAL_FRAC, SPLIT_SEED)
    n_test = len(test_X)
    print(f"Test set: {n_test} samples  "
          f"classes={[(test_y == c).sum().item() for c in range(3)]}")

    model = ClementsAccelerator(n_inputs=N_INPUTS, n_layers=N_LAYERS,
                                n_classes=N_CLASSES, n_networks=1)
    mesh  = model._ph_model['lx_module_2']

    weights = load_weights_csv(weights_file)
    set_mesh_weights(mesh, weights)

    model.eval()
    with torch.no_grad():
        out     = model(test_X)
        preds   = out.argmax(1)
        correct = (preds == test_y).sum().item()

    print(f"\nIris test accuracy (Python eval): "
          f"{correct}/{n_test} ({100.0 * correct / n_test:.4f}%)")
    print("\nPer-sample results:")
    for i, (pred, true) in enumerate(zip(preds.tolist(), test_y.tolist())):
        mark = '[correct]' if pred == true else '[wrong]  '
        print(f"  sample {i:2d}: predicted {_LABEL_NAMES[pred]:20s}  "
              f"expected {_LABEL_NAMES[true]:20s}  {mark}")


if __name__ == '__main__':
    main()
