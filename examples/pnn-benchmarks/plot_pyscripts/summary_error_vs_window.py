#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = "../results/digits/numerical_error/collected_digits_numerical_win_error.csv"
OUTPUT_DIR = "tables"
def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize numerical error vs window size from merged LuxIA benchmark CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=INPUT_CSV,
        help="Path to merged numerical error CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory where aggregated CSV and table CSV will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model filter, e.g. lx_clements.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter, e.g. digits.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device filter, e.g. cpu.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size filter.",
    )
    parser.add_argument(
        "--n_networks",
        type=int,
        default=None,
        help="Optional n_networks filter.",
    )
    parser.add_argument(
        "--n_inputs",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of n_inputs values to keep, e.g. --n_inputs 64 128 256 512",
    )
    return parser.parse_args()


def ensure_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_df(df, args):
    print("[*] Initial rows in input CSV:", len(df))
    if args.model is not None:
        df = df[df["model"] == args.model]
        print(f"[*] After model == {args.model!r}: {len(df)} rows")
    if args.dataset is not None:
        df = df[df["dataset"] == args.dataset]
        print(f"[*] After dataset == {args.dataset!r}: {len(df)} rows")
    if args.device is not None:
        df = df[df["device"] == args.device]
        print(f"[*] After device == {args.device!r}: {len(df)} rows")
    if args.batch_size is not None:
        df = df[df["batch_size"] == args.batch_size]
        print(f"[*] After batch_size == {args.batch_size}: {len(df)} rows")
    if args.n_networks is not None:
        df = df[df["n_networks"] == args.n_networks]
        print(f"[*] After n_networks == {args.n_networks}: {len(df)} rows")
    if args.n_inputs is not None and len(args.n_inputs) > 0:
        df = df[df["n_inputs"].isin(args.n_inputs)]
        print(f"[*] After n_inputs in {args.n_inputs}: {len(df)} rows")
    return df


def aggregate_over_seeds(df):
    group_cols = [
        "model",
        "dataset",
        "device",
        "n_networks",
        "n_inputs",
        "n_layers",
        "batch_size",
        "window_size",
        "is_matmul",
    ]
    print("[*] Grouping by:", group_cols)

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "count"),

            # Timing (mean and std across seeds)
            ref_forward_time_mean=("ref_forward_time_s", "mean"),
            ref_forward_time_std=("ref_forward_time_s", "std"),
            ref_backward_time_mean=("ref_backward_time_s", "mean"),
            ref_backward_time_std=("ref_backward_time_s", "std"),
            ref_total_time_mean=("ref_total_time_s", "mean"),
            ref_total_time_std=("ref_total_time_s", "std"),

            forward_time_mean=("forward_time_s", "mean"),
            forward_time_std=("forward_time_s", "std"),
            backward_time_mean=("backward_time_s", "mean"),
            backward_time_std=("backward_time_s", "std"),
            total_time_mean=("total_time_s", "mean"),
            total_time_std=("total_time_s", "std"),

            # Errors (mean and std across seeds)
            forward_error_mean=("forward_error", "mean"),
            forward_error_std=("forward_error", "std"),
            avg_grad_error_mean=("avg_grad_error", "mean"),
            avg_grad_error_std=("avg_grad_error", "std"),
            max_grad_error_mean=("max_grad_error", "mean"),
            max_grad_error_std=("max_grad_error", "std"),
        )
    )

    for col in [
        "ref_forward_time_std",
        "ref_backward_time_std",
        "ref_total_time_std",
        "forward_time_std",
        "backward_time_std",
        "total_time_std",
        "forward_error_std",
        "avg_grad_error_std",
        "max_grad_error_std",
    ]:
        agg[col] = agg[col].fillna(0.0)

    print("[*] Aggregated rows (one per config x window x is_matmul):", len(agg))
    return agg


def add_tm_normalized_ratios(agg):
    """
    Attach TM-full error for each (model, dataset, device, n_networks, n_inputs, n_layers, batch_size),
    then compute ratios sliced/TM for forward and avg-grad errors.
    """
    tm = agg[agg["is_matmul"] == 1].copy()
    print("[*] TM-full rows:", len(tm))

    tm = tm.rename(
        columns={
            "window_size": "tm_window_size",
            "forward_error_mean": "tm_forward_error_mean",
            "avg_grad_error_mean": "tm_avg_grad_error_mean",
            "max_grad_error_mean": "tm_max_grad_error_mean",
        }
    )

    join_cols = [
        "model",
        "dataset",
        "device",
        "n_networks",
        "n_inputs",
        "n_layers",
        "batch_size",
    ]
    keep_cols = join_cols + [
        "tm_window_size",
        "tm_forward_error_mean",
        "tm_avg_grad_error_mean",
        "tm_max_grad_error_mean",
    ]

    merged = agg.merge(
        tm[keep_cols],
        on=join_cols,
        how="left",
        validate="many_to_one",
    )

    eps = 1e-30
    merged["forward_error_ratio_to_tm"] = merged["forward_error_mean"] / (
        merged["tm_forward_error_mean"] + eps
    )
    merged["avg_grad_error_ratio_to_tm"] = merged["avg_grad_error_mean"] / (
        merged["tm_avg_grad_error_mean"] + eps
    )

    print("[*] Added TM-normalized ratios")
    return merged


def make_flat_ready_csv(agg, output_dir: Path):
    """
    Build a 'flat' CSV with one row per configuration, suitable for reporting
    Columns:
      n_inputs, window_size, is_tm_full,
      time_total_mean_s, time_total_std_s,
      forward_error_mean, forward_error_std,
      avg_grad_error_mean, avg_grad_error_std,
      forward_error_ratio_to_tm, avg_grad_error_ratio_to_tm
    """
    cols = [
        "n_inputs",
        "window_size",
        "is_matmul",
        "n_seeds",
        "total_time_mean",
        "total_time_std",
        "forward_error_mean",
        "forward_error_std",
        "avg_grad_error_mean",
        "avg_grad_error_std",
        "forward_error_ratio_to_tm",
        "avg_grad_error_ratio_to_tm",
    ]
    table = agg[cols].copy()

    table = table.rename(
        columns={
            "is_matmul": "is_tm_full",
            "total_time_mean": "time_total_mean_s",
            "total_time_std": "time_total_std_s",
        }
    )

    table = table.sort_values(
        ["n_inputs", "is_tm_full", "window_size"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out_path = output_dir / "table_error_vs_window_flat.csv"
    table.to_csv(out_path, index=False)
    print("[*] Wrote LaTeX-ready flat table CSV:", out_path)
    print("[*] Example rows:")
    print(table.head())


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[*] Output directory:", output_dir)

    print("[*] Reading input CSV:", args.input_csv)
    df = pd.read_csv(args.input_csv)

    required_cols = [
        "model",
        "dataset",
        "device",
        "seed",
        "n_networks",
        "n_inputs",
        "n_layers",
        "batch_size",
        "window_size",
        "is_matmul",
        "ref_forward_time_s",
        "ref_backward_time_s",
        "ref_total_time_s",
        "forward_time_s",
        "backward_time_s",
        "total_time_s",
        "forward_error",
        "avg_grad_error",
        "max_grad_error",
    ]
    ensure_columns(df, required_cols)

    df = filter_df(df, args)

    if df.empty:
        raise ValueError("No rows left after filtering. Check your filters and input CSV.")

    print("[*] Unique n_inputs after filtering:", sorted(df["n_inputs"].unique()))
    print("[*] Unique window_size after filtering:", sorted(df["window_size"].unique()))
    print("[*] Unique is_matmul values:", sorted(df["is_matmul"].unique()))
    print("[*] Unique seeds:", sorted(df["seed"].unique()))

    agg = aggregate_over_seeds(df)
    agg = add_tm_normalized_ratios(agg)

    agg = agg.sort_values(["n_inputs", "window_size", "is_matmul"]).reset_index(drop=True)
    agg_path = output_dir / "aggregated_error_vs_window.csv"
    agg.to_csv(agg_path, index=False)
    print("[*] Wrote aggregated CSV:", agg_path)

    make_flat_ready_csv(agg, output_dir)

    print("[*] Done.")


if __name__ == "__main__":
    main()
