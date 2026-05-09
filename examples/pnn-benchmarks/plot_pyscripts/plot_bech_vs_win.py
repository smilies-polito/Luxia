#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
from pathlib import Path

from util import create_bar_plot

INPUT_CSV = "../results/digits/benchmark/collected_digits_win_bench.csv"
OUTPUT_DIR = "plots"

TARGET_NS = [100, 300, 500, 700, 900]


def prepare_data(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    numeric_cols = [
        "n_inputs", "n_layers", "batch_size", "window_size",
        "forward_time", "backward_time", "total_time",
        "cpu_mem_before_mb", "cpu_mem_after_mb",
        "cpu_mem_peak_mb", "cpu_mem_consumption_mb",
        "gpu_mem_before_mb", "gpu_mem_after_mb",
        "gpu_mem_peak_mb", "gpu_mem_consumption_mb",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use the new benchmark semantics:
    # memory consumption = peak - initial
    if "cpu_mem_consumption_mb" in df.columns:
        df["cpu_mem_mb"] = df["cpu_mem_consumption_mb"]
    else:
        df["cpu_mem_mb"] = df["cpu_mem_peak_mb"].fillna(df["cpu_mem_after_mb"])

    if "gpu_mem_consumption_mb" in df.columns:
        df["gpu_mem_mb"] = df["gpu_mem_consumption_mb"]
    else:
        df["gpu_mem_mb"] = df["gpu_mem_peak_mb"].fillna(df["gpu_mem_after_mb"]).fillna(0)

    agg = (
        df.groupby(
            ["mode", "device", "n_inputs", "n_layers", "batch_size", "window_size"],
            as_index=False
        )
        .agg(
            mean_forward_time=("forward_time", "mean"),
            mean_backward_time=("backward_time", "mean"),
            mean_total_time=("total_time", "mean"),
            mean_cpu_mem=("cpu_mem_mb", "mean"),
            mean_gpu_mem=("gpu_mem_mb", "mean"),
        )
        .sort_values(["batch_size", "n_inputs", "window_size", "mode", "device"])
        .reset_index(drop=True)
    )
    return agg


def build_window_axis(batch_df: pd.DataFrame):
    slice_windows = sorted(
        int(w) for w in batch_df.loc[
            batch_df["mode"] == "slice", "window_size"
        ].dropna().unique()
    )

    slice_windows = [w for w in slice_windows if w != 128]

    labels = ["nl"] + [f"nl - {w - 1}" for w in slice_windows[1:]] + ["no slicing"]

    return slice_windows, labels


def get_row(batch_df: pd.DataFrame, n: int, device: str, mode: str, window_size: int):
    sub = batch_df[
        (batch_df["n_inputs"] == n) &
        (batch_df["device"] == device) &
        (batch_df["mode"] == mode) &
        (batch_df["window_size"] == window_size)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_processed_data(batch_df: pd.DataFrame):
    slice_windows, x_labels = build_window_axis(batch_df)

    colors = sns.color_palette("colorblind", n_colors=len(TARGET_NS))
    processed_data = {}

    for i, n in enumerate(TARGET_NS):
        lib_name = f"N={n}"
        processed_data[lib_name] = {"color": colors[i]}
        processed_data[lib_name]["CPU"] = {"cpu_time": [], "cpu_mem": []}
        processed_data[lib_name]["GPU"] = {"gpu_time": [], "cpu_mem": [], "gpu_mem": []}

        # sliced windows
        for w in slice_windows:
            cpu_row = get_row(batch_df, n=n, device="cpu", mode="slice", window_size=w)
            gpu_row = get_row(batch_df, n=n, device="cuda", mode="slice", window_size=w)

            if cpu_row is not None:
                processed_data[lib_name]["CPU"]["cpu_time"].append(float(cpu_row["mean_total_time"]))
                processed_data[lib_name]["CPU"]["cpu_mem"].append(float(cpu_row["mean_cpu_mem"]) / 1024.0)
            else:
                processed_data[lib_name]["CPU"]["cpu_time"].append(0.0)
                processed_data[lib_name]["CPU"]["cpu_mem"].append(0.0)

            if gpu_row is not None:
                processed_data[lib_name]["GPU"]["gpu_time"].append(float(gpu_row["mean_total_time"]))
                processed_data[lib_name]["GPU"]["cpu_mem"].append(float(gpu_row["mean_cpu_mem"]) / 1024.0)
                processed_data[lib_name]["GPU"]["gpu_mem"].append(float(gpu_row["mean_gpu_mem"]) / 1024.0)
            else:
                processed_data[lib_name]["GPU"]["gpu_time"].append(0.0)
                processed_data[lib_name]["GPU"]["cpu_mem"].append(0.0)
                processed_data[lib_name]["GPU"]["gpu_mem"].append(0.0)

        # full baseline = "no slicing"
        cpu_full_row = get_row(batch_df, n=n, device="cpu", mode="full", window_size=1)
        gpu_full_row = get_row(batch_df, n=n, device="cuda", mode="full", window_size=1)

        if cpu_full_row is not None:
            processed_data[lib_name]["CPU"]["cpu_time"].append(float(cpu_full_row["mean_total_time"]))
            processed_data[lib_name]["CPU"]["cpu_mem"].append(float(cpu_full_row["mean_cpu_mem"]) / 1024.0)
        else:
            processed_data[lib_name]["CPU"]["cpu_time"].append(0.0)
            processed_data[lib_name]["CPU"]["cpu_mem"].append(0.0)

        if gpu_full_row is not None:
            processed_data[lib_name]["GPU"]["gpu_time"].append(float(gpu_full_row["mean_total_time"]))
            processed_data[lib_name]["GPU"]["cpu_mem"].append(float(gpu_full_row["mean_cpu_mem"]) / 1024.0)
            processed_data[lib_name]["GPU"]["gpu_mem"].append(float(gpu_full_row["mean_gpu_mem"]) / 1024.0)
        else:
            processed_data[lib_name]["GPU"]["gpu_time"].append(0.0)
            processed_data[lib_name]["GPU"]["cpu_mem"].append(0.0)
            processed_data[lib_name]["GPU"]["gpu_mem"].append(0.0)

    return processed_data, x_labels


def make_memory_plot(batch_df: pd.DataFrame, bs: int, out_dir: Path):
    processed_data, x_labels = build_processed_data(batch_df)

    img_base = out_dir / f"benchmark_window_memory_bs{int(bs)}"
    create_bar_plot(
        processed_data,
        x_labels,
        img_name=str(img_base),
        title="",
        y_label="Memory Consumption (GB)",
        x_label="Number of Windows (nw)",
        plot_type="memory",
        log_scale=False,
        max_y_value=17.5,
        figsize=(35, 8)
    )


def make_time_plot(batch_df: pd.DataFrame, bs: int, out_dir: Path):
    processed_data, x_labels = build_processed_data(batch_df)

    img_base = out_dir / f"benchmark_window_time_bs{int(bs)}"
    create_bar_plot(
        processed_data,
        x_labels,
        img_name=str(img_base),
        title="",
        y_label="Time (s)",
        x_label="Window Size",
        plot_type="time",
        log_scale=False,
        max_y_value=10,
        figsize=(35, 8)
    )


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = prepare_data(INPUT_CSV)
    agg.to_csv(out_dir / "benchmarks_window_agg.csv", index=False)

    batch_sizes = sorted(agg["batch_size"].dropna().unique())

    for bs in batch_sizes:
        batch_df = agg[agg["batch_size"] == bs].copy()
        if batch_df.empty:
            continue

        batch_df = batch_df[batch_df["n_inputs"].isin(TARGET_NS)].copy()

        make_memory_plot(batch_df, bs, out_dir)
        make_time_plot(batch_df, bs, out_dir)

    print(f"Done. Files saved in: {out_dir}")


if __name__ == "__main__":
    main()
