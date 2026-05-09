import seaborn as sns
from load_files.define_bech_files import define_ns_file_paths
from util import process_stats_data, create_bar_plot


def main():
    """Main function to process benchmark data and generate plots."""
    # Define paths and settings
    base_path = "../results/digits/train/"
    print(f"Using base path: {base_path}")

    # Get file paths and batch sizes
    library_files = define_ns_file_paths()
    n_sizes = [100, 300, 500, 700, 900]

    # Generate colors for libraries
    colors = sns.color_palette("colorblind", n_colors=len(library_files))
    # colors = sns.color_palette("gray", n_colors=len(library_files))
    # colors = sns.color_palette("bright", n_colors=len(library_files))
    # colors = sns.color_palette("dark:salmon", n_colors=len(library_files))

    # Process data for each library
    processed_data = {}
    for i, (name, device) in enumerate(library_files.items()):
        processed_data[name] = {"color": colors[i]}

        # Process CPU data
        if device["CPU"]:
            t, m = process_stats_data(device["CPU"], base_path, name)
            processed_data[name]["CPU"] = {"cpu_time": t, "cpu_mem": m}

        # Process GPU data
        if device["GPU"]:
            t, c_m, g_m = process_stats_data(
                device["GPU"], base_path, name, is_gpu=True)
            processed_data[name]["GPU"] = {
                "gpu_time": t, "cpu_mem": c_m, "gpu_mem": g_m}

    # Print processed data for debugging
    for lib_name, lib_data in processed_data.items():
        print(f"{lib_name}:")
        if "CPU" in lib_data:
            print(f"  CPU Time: {lib_data['CPU']['cpu_time'] }")
            print(f"  CPU Memory: {lib_data['CPU']['cpu_mem']}")
        if "GPU" in lib_data:
            print(f"  GPU Time: {lib_data['GPU']['gpu_time']}")
            print(f"  CPU Memory (on GPU): {lib_data['GPU']['cpu_mem']}")
            print(f"  GPU Memory: {lib_data['GPU']['gpu_mem']}")

    # Create memory plot
    create_bar_plot(
        processed_data,
        n_sizes,
        img_name="benchmark_ns_memory",
        title="",
        y_label="Memory (GB)",
        x_label="N Mesh Size",
        plot_type="memory",
        log_scale=False,
        max_y_value=20
    )

    # Create time plot
    create_bar_plot(
        processed_data,
        n_sizes,
        img_name="benchmark_ns_time",
        title="",
        y_label="Time (s)",
        x_label="N Mesh Size",
        plot_type="time",
        log_scale=False,
        max_y_value=60
    )


if __name__ == "__main__":
    main()
