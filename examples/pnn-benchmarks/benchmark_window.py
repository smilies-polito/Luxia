#!/usr/bin/env python3
import os
import csv
import gc
import time
import random
import threading

import numpy as np
import psutil
import torch

from util.init_args import init_args
from util.init_model import init_model
from util.mem_usage import get_gpu_mem_usage
from util.utils import (
    load_dataset,
    expand_data_features,
    generate_dataloader,
    get_device,
)


# ==========================================================
# CSV
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


# ==========================================================
# Device utils
# ==========================================================

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_gpu_used_mem_mb(device, verbose=False):
    """
    Returns 'used' GPU memory (MB) for the current CUDA device, or 0.0 on CPU.
    """
    if device.type != "cuda":
        return 0.0

    gpu_mem = get_gpu_mem_usage(verbose)

    if isinstance(gpu_mem, tuple):
        if len(gpu_mem) == 3:
            _, used_mb, _ = gpu_mem
            return float(used_mb)
        if len(gpu_mem) == 2:
            free_mb, total_mb = gpu_mem
            return float(total_mb - free_mb)

    if isinstance(gpu_mem, (int, float)):
        return float(gpu_mem)

    raise TypeError(
        f"Unsupported return type from get_gpu_mem_usage: {type(gpu_mem)}"
    )


def get_process_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ==========================================================
# Unified CPU + GPU peak monitor
# ==========================================================

class ResourceMonitor:
    """
    Monitors peak CPU RSS (MB) and, if device is CUDA, peak GPU 'used' memory (MB)
    during a benchmark execution window.
    """
    def __init__(self, device, interval=0.005):
        self.device = device
        self.interval = interval

        self._running = False
        self._thread = None

        # CPU
        self._process = psutil.Process(os.getpid())
        self._cpu_peak_rss_bytes = 0

        # GPU
        self._gpu_peak_used_mb = 0.0

    def _loop(self):
        while self._running:
            try:
                # CPU peak RSS
                rss = self._process.memory_info().rss
                if rss > self._cpu_peak_rss_bytes:
                    self._cpu_peak_rss_bytes = rss

                # GPU peak used
                if self.device.type == "cuda":
                    gpu_mem = get_gpu_mem_usage(verbose=False)
                    if isinstance(gpu_mem, tuple) and len(gpu_mem) >= 2:
                        used_mb = float(gpu_mem[1])
                        if used_mb > self._gpu_peak_used_mb:
                            self._gpu_peak_used_mb = used_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._cpu_peak_rss_bytes = self._process.memory_info().rss
        self._gpu_peak_used_mb = 0.0

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()

    @property
    def cpu_peak_mb(self):
        return self._cpu_peak_rss_bytes / (1024 ** 2)

    @property
    def gpu_peak_mb(self):
        # 0.0 if no CUDA or nothing recorded
        return self._gpu_peak_used_mb


# ==========================================================
# Benchmark
# ==========================================================

def benchmark_model(model, data_batch, labels_batch, loss_fn, device, verbose=False):
    """
    Benchmark exactly one fresh-process model execution.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.zero_grad(set_to_none=True)

    cpu_before_mb = get_process_rss_mb()
    gpu_before_mb = get_gpu_used_mem_mb(device, verbose=verbose)

    monitor = ResourceMonitor(device=device, interval=0.002)
    monitor.start()

    sync_device(device)
    t0 = time.time()

    out = model(data_batch)

    sync_device(device)
    t1 = time.time()

    loss = loss_fn(out, labels_batch)

    sync_device(device)
    t2 = time.time()

    loss.backward()

    sync_device(device)
    t3 = time.time()

    monitor.stop()

    cpu_after_mb = get_process_rss_mb()
    gpu_after_mb = get_gpu_used_mem_mb(device, verbose=verbose)

    cpu_peak_mb = monitor.cpu_peak_mb
    gpu_peak_mb = monitor.gpu_peak_mb

    # "Consumption" = peak - initial
    cpu_mem_consumption_mb = max(0.0, cpu_peak_mb - cpu_before_mb)
    gpu_mem_consumption_mb = max(0.0, gpu_peak_mb - gpu_before_mb)

    return {
        "output_shape": tuple(out.shape),
        "loss_value": float(loss.detach().cpu().item()),

        "forward_time": t1 - t0,
        "backward_time": t3 - t2,
        "total_time": t3 - t0,

        "cpu_mem_before_mb": cpu_before_mb,
        "cpu_mem_after_mb": cpu_after_mb,
        "cpu_mem_peak_mb": cpu_peak_mb,
        "cpu_mem_consumption_mb": cpu_mem_consumption_mb,

        "gpu_mem_before_mb": gpu_before_mb,
        "gpu_mem_after_mb": gpu_after_mb,
        "gpu_mem_peak_mb": gpu_peak_mb,
        "gpu_mem_consumption_mb": gpu_mem_consumption_mb,
    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    args = init_args()

    if not hasattr(args, "mode"):
        raise ValueError(
            "benchmark_mode.py requires a --mode argument with one of: ref, full, slice"
        )

    if args.mode not in {"ref", "full", "slice"}:
        raise ValueError("--mode must be one of: ref, full, slice")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.device = get_device(args.device, args.verbose)

    # full/slice use float32; ref uses float64
    if args.mode == "ref":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    print(f"[*] Benchmark mode: {args.mode}")
    print(f"[*] Default float dtype set to: {torch.get_default_dtype()}")

    print("[*] Loading dataset...")
    data_train, labels_train, data_test, labels_test = load_dataset(
        args.dataset,
        args.verbose
    )

    n_features = data_train.shape[-1]

    data_train, data_test = expand_data_features(
        data_train,
        data_test,
        n_features,
        args.n_inputs,
        args.verbose
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
        verbose=args.verbose
    )

    n_classes = len(torch.unique(labels_train))

    data_batch, labels_batch = next(iter(datal_train))

    if args.mode == "ref":
        run_device = torch.device("cpu")
        data_batch = data_batch.to(run_device, dtype=torch.float64)
        labels_batch = labels_batch.squeeze().to(run_device)
        is_matmul = True
    elif args.mode == "full":
        run_device = args.device
        data_batch = data_batch.to(run_device)
        labels_batch = labels_batch.squeeze().to(run_device)
        is_matmul = True
    else:  # slice
        run_device = args.device
        data_batch = data_batch.to(run_device)
        labels_batch = labels_batch.squeeze().to(run_device)
        is_matmul = False

    print("[*] Initializing model...")
    model = init_model(
        name=args.model,
        device=run_device,
        n_inputs=args.n_inputs,
        n_classes=n_classes,
        n_layers=args.n_layers,
        n_networks=args.n_networks,
        is_matmul=is_matmul,
        window_size=args.window_size,
    )
    model.to(run_device)

    loss_fn = torch.nn.CrossEntropyLoss()

    print("[*] Running benchmark...")
    stats = benchmark_model(
        model=model,
        data_batch=data_batch,
        labels_batch=labels_batch,
        loss_fn=loss_fn,
        device=run_device,
        verbose=args.verbose,
    )

    print(
        f"[TIME] forward={stats['forward_time']:.6f}s | "
        f"backward={stats['backward_time']:.6f}s | "
        f"total={stats['total_time']:.6f}s"
    )

    print(
        f"[CPU MEM] before={stats['cpu_mem_before_mb']:.2f} MB | "
        f"peak={stats['cpu_mem_peak_mb']:.2f} MB | "
        f"consumption={stats['cpu_mem_consumption_mb']:.2f} MB"
    )

    if run_device.type == "cuda":
        print(
            f"[GPU MEM] before={stats['gpu_mem_before_mb']:.2f} MB | "
            f"peak={stats['gpu_mem_peak_mb']:.2f} MB | "
            f"consumption={stats['gpu_mem_consumption_mb']:.2f} MB"
        )

    results = [{
        "mode": args.mode,
        "model": args.model,
        "dataset": args.dataset,
        "device": str(run_device),
        "seed": args.seed,
        "n_networks": args.n_networks,
        "n_inputs": args.n_inputs,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "window_size": getattr(args, "window_size", 1),

        "forward_time": stats["forward_time"],
        "backward_time": stats["backward_time"],
        "total_time": stats["total_time"],

        "cpu_mem_before_mb": stats["cpu_mem_before_mb"],
        "cpu_mem_after_mb": stats["cpu_mem_after_mb"],
        "cpu_mem_peak_mb": stats["cpu_mem_peak_mb"],
        "cpu_mem_consumption_mb": stats["cpu_mem_consumption_mb"],

        "gpu_mem_before_mb": stats["gpu_mem_before_mb"],
        "gpu_mem_after_mb": stats["gpu_mem_after_mb"],
        "gpu_mem_peak_mb": stats["gpu_mem_peak_mb"],
        "gpu_mem_consumption_mb": stats["gpu_mem_consumption_mb"],

        "loss_value": stats["loss_value"],
        "output_shape": str(stats["output_shape"]),
    }]

    filename = os.path.join(
        args.exp_dir,
        f"benchmark_{args.mode}_{args.model}.csv"
    )

    save_results(filename, results)

    print("[*] Done.")
