#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge benchmark and numerical_error CSV files into master CSVs. "
            "Automatically detects both types per dataset."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="../results",
        help="Base directory containing datasets (e.g. ../results).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digits",
        help=(
            "Dataset subfolder under root_dir (e.g. digits, cifar). "
            "Use 'all' to process all datasets under root_dir."
        ),
    )
    parser.add_argument(
        "--dedup_cols",
        type=str,
        nargs="*",
        default=[
            "model",
            "dataset",
            "device",
            "seed",
            "n_networks",
            "n_inputs",
            "n_layers",
            "batch_size",
            "window_size",
            "mode",
        ],
        help="Columns used to drop duplicates.",
    )
    parser.add_argument(
        "--sort_cols",
        type=str,
        nargs="*",
        default=[
            "run_type",
            "mode",
            "model",
            "dataset",
            "device",
            "n_inputs",
            "n_layers",
            "batch_size",
            "window_size",
            "seed",
        ],
        help="Columns used to sort the final CSV.",
    )
    return parser.parse_args()


def extract_metadata(csv_path: Path, scan_root: Path, dataset_name: str, run_type: str):
    """Extract metadata from the folder path structure, relative to the scan root."""
    rel = csv_path.relative_to(scan_root)
    parts = rel.parts

    metadata = {
        "source_file": str(csv_path),
        "model": None,
        "dataset": dataset_name,
        "device": None,
        "n_inputs": None,
        "n_layers": None,
        "batch_size": None,
        "mode": None,
        "timestamp": None,
        "window_size": None,
        "run_type": run_type,  # 'benchmark' or 'numerical_error'
    }

    # Expected relative path when scan_root is ../results/<dataset>/<run_type>:
    #   <experiment_folder>/<timestamp_folder>/<mode_folder>/file.csv
    if len(parts) >= 4:
        experiment_folder = parts[0]
        timestamp_folder = parts[1]
        mode_folder = parts[2]
        filename = csv_path.name

        metadata["timestamp"] = timestamp_folder

        # Parse experiment folder, e.g. lx_clements_64x64_bs64_cuda
        exp_parts = experiment_folder.split("_")
        if exp_parts:
            metadata["model"] = exp_parts[0]

        # Parse dimensions like 64x64
        dims_match = re.search(r"(\d+)x(\d+)", experiment_folder)
        if dims_match:
            metadata["n_inputs"] = int(dims_match.group(1))
            metadata["n_layers"] = int(dims_match.group(2))

        # Parse batch_size like bs64
        bs_match = re.search(r"bs(\d+)", experiment_folder)
        if bs_match:
            metadata["batch_size"] = int(bs_match.group(1))

        # Parse device (cpu/cuda at end)
        if experiment_folder.endswith("cpu"):
            metadata["device"] = "cpu"
        elif experiment_folder.endswith("cuda"):
            metadata["device"] = "cuda"

        # Parse mode and window_size (primarily for benchmark files)
        lower_filename = filename.lower()
        if "full" in lower_filename or mode_folder == "full":
            metadata["mode"] = "full"
        elif "slice" in lower_filename:
            w_match_folder = re.search(r"slice_w(\d+)", mode_folder)
            w_match_file = re.search(r"slice_.*?w(\d+)", lower_filename)
            if w_match_folder:
                metadata["window_size"] = int(w_match_folder.group(1))
            elif w_match_file:
                metadata["window_size"] = int(w_match_file.group(1))
            metadata["mode"] = "slice"

    return metadata


def find_files(scan_root: Path, run_type: str):
    """Return list of CSV files to process based on run_type."""
    if not scan_root.exists():
        print(f"[WARN] Directory does not exist: {scan_root}")
        return []

    if run_type == "benchmark":
        full_files = list(scan_root.rglob("benchmark_full_*.csv"))
        slice_files = list(scan_root.rglob("benchmark_slice_*.csv"))
        all_files = sorted(full_files + slice_files)

        print(f"[*] [{scan_root}] Found {len(full_files)} benchmark full files")
        print(f"[*] [{scan_root}] Found {len(slice_files)} benchmark slice files")

    elif run_type == "numerical_error":
        # Adjust this pattern if your numerical_error filenames differ
        numerical_error_files = list(scan_root.rglob("numerical_error_*.csv"))
        all_files = sorted(numerical_error_files)

        print(f"[*] [{scan_root}] Found {len(numerical_error_files)} numerical_error files")

    else:
        print(f"[ERROR] Unknown run_type: {run_type}")
        all_files = []

    return all_files


def merge_for_run_type(
    dataset_root: Path,
    dataset_name: str,
    run_type: str,
    args,
    output_path: Path | None = None,
):
    """Merge all CSVs for a given dataset and run_type, write to output_path."""
    scan_root = dataset_root / run_type
    print(f"\n[*] Processing dataset='{dataset_name}', run_type='{run_type}'")
    print(f"[*] Scanning directory: {scan_root}")

    files = find_files(scan_root, run_type)

    if not files:
        print(f"[WARN] No CSV files found for dataset='{dataset_name}', run_type='{run_type}'")
        return

    dfs = []
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {csv_file} ({e})")
            continue

        if df.empty:
            print(f"[WARN] Skipping empty file: {csv_file}")
            continue

        metadata = extract_metadata(csv_file, scan_root, dataset_name, run_type)

        for key, value in metadata.items():
            if value is not None:
                df[key] = value

        dfs.append(df)
        print(f"[*] Processed: {csv_file.name} -> {metadata}")

    if not dfs:
        print(f"[WARN] No valid CSV files were loaded for run_type='{run_type}'")
        return

    merged_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Drop duplicates
    valid_dedup_cols = [c for c in args.dedup_cols if c in merged_df.columns]
    if valid_dedup_cols:
        initial_rows = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=valid_dedup_cols, keep="last")
        print(f"[*] Dropped duplicates: {initial_rows} -> {len(merged_df)} rows")

    # Sort
    valid_sort_cols = [c for c in args.sort_cols if c in merged_df.columns]
    if valid_sort_cols:
        merged_df = merged_df.sort_values(valid_sort_cols).reset_index(drop=True)

    # Decide output path if not provided
    if output_path is None:
        if run_type == "benchmark":
            filename = f"collected_{dataset_name}_win_bench.csv"
        else:
            filename = f"collected_{dataset_name}_numerical_win_error.csv"
        output_path = scan_root / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"\n[*] SUCCESS: dataset='{dataset_name}', run_type='{run_type}' → {output_path}")
    print(f"[*] Final shape: {merged_df.shape}")
    print(f"[*] Columns: {list(merged_df.columns)}")
    if valid_dedup_cols:
        print(f"[*] Unique combinations:")
        print(merged_df[valid_dedup_cols].nunique().to_string())

    if "mode" in merged_df.columns:
        print(f"\n[*] Summary by mode:")
        print(merged_df.groupby("mode").size())


def main():
    args = parse_args()

    base_root = Path(args.root_dir).resolve()  # e.g. ../results
    print(f"[*] Base root_dir: {base_root}")

    # Determine which datasets to process
    if args.dataset == "all":
        datasets = [p.name for p in base_root.iterdir() if p.is_dir()]
    else:
        datasets = [args.dataset]

    print(f"[*] Datasets to process: {datasets}")

    for dataset_name in datasets:
        dataset_root = base_root / dataset_name
        if not dataset_root.exists():
            print(f"[WARN] Dataset directory does not exist: {dataset_root}")
            continue

        # Merge benchmark
        merge_for_run_type(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            run_type="benchmark",
            args=args,
            output_path=None,  # auto name: collected_<dataset>_win_bench.csv
        )

        # Merge numerical_error
        merge_for_run_type(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            run_type="numerical_error",
            args=args,
            output_path=None,  # auto name: collected_<dataset>_numerical_win_error.csv
        )

    print("\n[*] All done!")


if __name__ == "__main__":
    main()
