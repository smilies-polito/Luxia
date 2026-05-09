import os
import csv
import random
import time

import numpy as np
import torch

from util.init_args import init_args
from util.init_model import init_model
from util.utils import (
    load_dataset,
    expand_data_features,
    generate_dataloader,
    get_device,
)


# ==========================================================
# Utils
# ==========================================================

def save_results(filename, results):
    print("[*] Saving results...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        headers = results[0].keys()
        writer.writerow(headers)
        for row in results:
            writer.writerow(row.values())


def rel_error(x, y):
    return (
        torch.norm(x - y)
        / (torch.norm(y) + 1e-12)
    ).item()


def compute_gradient_error(model, reference_model):
    errors = []

    for (_, p), (_, p_ref) in zip(
        model.named_parameters(),
        reference_model.named_parameters(),
    ):
        if p.grad is None or p_ref.grad is None:
            continue

        err = (
            torch.norm(p.grad - p_ref.grad)
            / (torch.norm(p_ref.grad) + 1e-12)
        )

        errors.append(err.item())

    return {
        "avg": float(np.mean(errors)) if errors else 0.0,
        "max": float(np.max(errors)) if errors else 0.0,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_step(model, data, labels, loss_fn, device):
    model.zero_grad(set_to_none=True)

    synchronize(device)
    t0 = time.perf_counter()

    out = model(data)

    synchronize(device)
    t1 = time.perf_counter()

    loss = loss_fn(out, labels)
    loss.backward()

    synchronize(device)
    t2 = time.perf_counter()

    return {
        "out": out,
        "forward_time_s": t1 - t0,
        "backward_time_s": t2 - t1,
        "total_time_s": t2 - t0,
    }


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    args = init_args()

    seed_everything(args.seed)

    args.device = get_device(args.device, args.verbose)

    window_size = getattr(args, "window_size", 1)

    # ======================================================
    # Data
    # ======================================================

    data_train, labels_train, data_test, labels_test = load_dataset(
        args.dataset,
        args.verbose,
    )

    n_features = data_train.shape[-1]

    data_train, data_test = expand_data_features(
        data_train,
        data_test,
        n_features,
        args.n_inputs,
        args.verbose,
    )

    datal_train, _, _ = generate_dataloader(
        val_frac=args.val_frac,
        n_features=args.n_inputs,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        data_train=data_train,
        labels_train=labels_train,
        data_test=data_test,
        labels_test=labels_test,
        verbose=args.verbose,
    )

    n_classes = len(torch.unique(labels_train))

    # single batch
    data_batch, labels_batch = next(iter(datal_train))
    data_batch = data_batch.to(args.device)
    labels_batch = labels_batch.squeeze().to(args.device)

    loss_fn = torch.nn.CrossEntropyLoss()

    # ======================================================
    # Reference model (always matmul)
    # ======================================================

    seed_everything(args.seed)

    model_ref = init_model(
        name=args.model,
        device=args.device,
        n_inputs=args.n_inputs,
        n_classes=n_classes,
        n_layers=args.n_layers,
        n_networks=args.n_networks,
        is_matmul=True,
        window_size=args.n_layers,
    )

    # ======================================================
    # Test model
    # ======================================================

    seed_everything(args.seed)

    model_test = init_model(
        name=args.model,
        device=args.device,
        n_inputs=args.n_inputs,
        n_classes=n_classes,
        n_layers=args.n_layers,
        n_networks=args.n_networks,
        is_matmul=args.is_matmul,
        window_size=window_size,
    )

    # ======================================================
    # Run
    # ======================================================

    ref_run = run_step(
        model=model_ref,
        data=data_batch,
        labels=labels_batch,
        loss_fn=loss_fn,
        device=args.device,
    )

    test_run = run_step(
        model=model_test,
        data=data_batch,
        labels=labels_batch,
        loss_fn=loss_fn,
        device=args.device,
    )

    # ======================================================
    # Errors
    # ======================================================

    forward_error = rel_error(
        test_run["out"].detach(),
        ref_run["out"].detach(),
    )

    grad_error = compute_gradient_error(
        model_test,
        model_ref,
    )

    print()
    print("Numerical errors")
    print(f"Forward error:  {forward_error:.6e}")
    print(f"Gradient error: {grad_error['avg']:.6e}")

    print()
    print("Reference (matmul)")
    print(f"Forward:  {ref_run['forward_time_s']:.6f} s")
    print(f"Backward: {ref_run['backward_time_s']:.6f} s")
    print(f"Total:    {ref_run['total_time_s']:.6f} s")

    print()
    print("Test")
    print(f"Forward:  {test_run['forward_time_s']:.6f} s")
    print(f"Backward: {test_run['backward_time_s']:.6f} s")
    print(f"Total:    {test_run['total_time_s']:.6f} s")

    # ======================================================
    # Save
    # ======================================================

    results = [{
        "model": args.model,
        "dataset": args.dataset,
        "device": str(args.device),
        "seed": args.seed,
        "n_networks": args.n_networks,
        "n_inputs": args.n_inputs,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "window_size": window_size,
        "is_matmul": int(bool(args.is_matmul)),

        "ref_forward_time_s": ref_run["forward_time_s"],
        "ref_backward_time_s": ref_run["backward_time_s"],
        "ref_total_time_s": ref_run["total_time_s"],

        "forward_time_s": test_run["forward_time_s"],
        "backward_time_s": test_run["backward_time_s"],
        "total_time_s": test_run["total_time_s"],

        "forward_error": forward_error,
        "avg_grad_error": grad_error["avg"],
        "max_grad_error": grad_error["max"],
    }]

    filename = os.path.join(
        args.exp_dir,
        f"numerical_error_{args.model}.csv",
    )

    save_results(filename, results)

    print("[*] Done.")
