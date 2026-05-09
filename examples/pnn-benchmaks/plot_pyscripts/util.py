import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_vacc_tloss(file_path, file_name):
    df = pd.read_csv(file_path + "/" + file_name)
    train_loss = df["loss"]
    val_acc = df["val_accuracy"]
    return val_acc, train_loss


def get_cpu_mem_time(file_path, file_name):
    df = pd.read_csv(file_path + "/" + file_name)
    cpu_mem = df["cpu_mem"]
    # Convet from MB to GB
    cpu_mem = cpu_mem / 1024
    time = df["time"]
    return time, cpu_mem


def get_gpu_mem_time(file_path, file_name):
    df = pd.read_csv(file_path + "/" + file_name)
    cpu_mem = df["cpu_mem"]
    # Convet from MB to GB
    cpu_mem = cpu_mem / 1024
    gpu_mem = df["gpu_mem"]
    gpu_mem = gpu_mem / 1024
    time = df["time"]
    return time, cpu_mem, gpu_mem


def process_library_data(file_list, base_path, library_name):
    """
    Process data for a specific library by loading and averaging results from multiple runs.
    Args:
        file_list: List of CSV files containing training statistics
        base_path: Base directory path where files are located
        library_name: Name of the library (for display purposes)
    Returns:
        Tuple containing mean, min, and max values for validation accuracy and training loss
    """
    avg_val_acc = []
    avg_train_loss = []
    for i, file_name in enumerate(file_list):
        print(f"Processing {library_name} {i+1}/{len(file_list)}: {file_name}")
        val_acc, t_loss = get_vacc_tloss(base_path, file_name)
        avg_val_acc.append(val_acc)
        avg_train_loss.append(t_loss)
    # Calculate mean, min, and max values across all runs
    mean_val_acc = np.mean(avg_val_acc, axis=0)
    min_val_acc = np.min(avg_val_acc, axis=0)
    max_val_acc = np.max(avg_val_acc, axis=0)
    mean_train_loss = np.mean(avg_train_loss, axis=0)
    min_train_loss = np.min(avg_train_loss, axis=0)
    max_train_loss = np.max(avg_train_loss, axis=0)

    return (mean_val_acc, min_val_acc, max_val_acc, mean_train_loss, min_train_loss, max_train_loss)


# def create_accuracy_plot(nx_data, pt_data, neu_data, lx_data, colors):
def create_plot(data, x_label="Epochs", y_label=None, figsize=(12, 8), marker_every=10,
                line_width=1, img_name=None
                ):
    """
    Create and save a plot comparing validation accuracy or training loss across libraries.
    Args:
        data: Dictionary where keys are library names and values are dictionaries with
              "mean", "min", "max", and "color" keys
        x_label: Label for x-axis (default: "Epochs")
        y_label: Label for y-axis
        figsize: Figure size as (width, height) tuple
        marker_every: Place markers every N points
        line_width: Width of plot lines
    """
    # Define visual elements with better visibility for grayscale
    # More distinct hatch patterns for better grayscale differentiation
    hatch_patterns = ['/////', 'xxxxx', '-----',
                      '|||||', '++++', '....', '\\\\\\\\\\']
    # Choose distinct markers that are easy to differentiate in grayscale
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    # More distinct line styles for grayscale differentiation
    linestyles = ['-', '--', '-.', ':',
                  (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Keep track of the longest data array to determine x-axis range
    max_length = 0

    # Plot each library's data
    for i, library_name in enumerate(data.keys()):
        x_length = len(data[library_name]["mean"])
        x = np.arange(x_length)
        max_length = max(max_length, x_length)

        current_color = data[library_name]["color"]
        hatch = hatch_patterns[i % len(hatch_patterns)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        # Plot confidence interval
        plt.fill_between(
            x,
            data[library_name]["min"],
            data[library_name]["max"],
            color=current_color,
            alpha=0.1,
            hatch=hatch
        )

        # Plot mean line
        plt.plot(
            x, data[library_name]["mean"],
            color=current_color,
            label=library_name,
            marker=marker,
            linestyle=linestyle,
            markevery=marker_every + 5*i,
            markersize=10,
            linewidth=line_width
        )

    # Add labels and styling
    plt.xlabel(x_label, fontsize=24)
    plt.xticks(fontsize=20)
    plt.ylabel(y_label, fontsize=24)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.grid(True, alpha=0.3)  # Make grid less prominent
    plt.tight_layout()

    # Set reasonable x-axis limits
    plt.xlim(0, max_length - 1)

    filename = f"{img_name}.png" if img_name else "plot.png"
    # Save the figure
    try:
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return f"Plot saved as {filename}"
    except Exception as e:
        return f"Error saving plot: {e}"


def create_vacc_tloss_plots(data, img_name="plot"):
    """
    Create both accuracy and loss plots from the same data.
    Args:
        data: Dictionary where keys are library names and values are tuples of 
              (mean_val_acc, min_val_acc, max_val_acc, mean_train_loss, min_train_loss, max_train_loss)
        colors: Dictionary mapping library names to color values
        x_label: Label for x-axis (default: "Epochs")
    """
    # Create validation accuracy plot
    # Get all the names, mean, max, min values, and colors
    accuracy_dict = {}
    loss_dict = {}
    for name in data.keys():
        accuracy_dict[name] = {}
        loss_dict[name] = {}
        accuracy_dict[name]["mean"] = data[name]["mean_val_acc"]
        accuracy_dict[name]["min"] = data[name]["min_val_acc"]
        accuracy_dict[name]["max"] = data[name]["max_val_acc"]
        accuracy_dict[name]["color"] = data[name]["color"]
        loss_dict[name]["mean"] = data[name]["mean_train_loss"]
        loss_dict[name]["min"] = data[name]["min_train_loss"]
        loss_dict[name]["max"] = data[name]["max_train_loss"]
        loss_dict[name]["color"] = data[name]["color"]
    # Create accuracy plot
    x_label = "Epochs"

    # image name for saving
    image_name = f"{img_name}_vacc"
    acc_result = create_plot(
        accuracy_dict, x_label=x_label, y_label="Validation Accuracy", img_name=image_name)
    image_name = f"{img_name}_tloss"
    loss_result = create_plot(
        loss_dict, x_label=x_label, y_label="Training Loss", img_name=image_name)
    return f"{acc_result}\n{loss_result}"


def create_bar_plot(data,
                    x_vals,
                    img_name,
                    title,
                    y_label="",
                    x_label="Batch Size",
                    plot_type="memory",
                    log_scale=False,
                    max_y_value=None,
                    figsize=(12, 8),
                    ):
    """
    Unified function to create both memory and time plots optimized for grayscale printing.
    
    Args:
        data: Dictionary with benchmark data
        x_vals: Values for x-axis (batch sizes)
        img_name: Output image name
        title: Plot title
        y_label: Y-axis label
        x_label: X-axis label
        plot_type: Either "memory" or "time"
        log_scale: Whether to use log scale for y-axis
        max_y_value: Optional maximum y value
    """

    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    x_positions = np.arange(len(x_vals))

    # Track legend items and current position
    legend_handles, legend_labels = [], []
    num_bars = sum(1 for lib in data.values()
                   for k in lib.keys() if k in ['CPU', 'GPU'])
    bar_width = 0.9 / max(1, num_bars)
    pos_index = 0

    # Define different hatches for CPU vs GPU
    cpu_hatches = ['|', 'o', '-', 'x', '.', '*', '/', 'O', '+']
    gpu_hatches = ['||||', 'oooo', '----', 'xxxx',
                   '....', '****', '////', 'OOO', '+++']

    # Process each library
    for i, (lib_name, lib_data) in enumerate(data.items()):
        # Instead of color, use grayscale
        color = data[lib_name]["color"]

        # Select hatches for this library
        cpu_hatch = cpu_hatches[i % len(cpu_hatches)]
        gpu_hatch = gpu_hatches[i % len(gpu_hatches)]

        # CPU time/memory
        if "CPU" in lib_data and lib_data["CPU"]:
            key_str = "cpu_time" if plot_type == "time" else "cpu_mem"
            y_vals = lib_data["CPU"].get(key_str, [])
            x_pos = x_positions - 0.4 + (pos_index + 0.5) * bar_width

            # Use grayscale and distinct hatching for CPU
            cpu_bar = ax.bar(x_pos, y_vals, bar_width,
                             color=color, edgecolor='black',
                             hatch=cpu_hatch, linewidth=1.5)

            legend_handles.append(cpu_bar[0])
            legend_labels.append(f"{lib_name} (CPU)")
            pos_index += 1

        # GPU time/memory
        if "GPU" in lib_data and lib_data["GPU"]:
            # Get key based on plot type
            key_str = "gpu_time" if plot_type == "time" else "gpu_mem"
            gpu_vals = lib_data["GPU"].get(key_str, [])
            x_pos = x_positions - 0.4 + (pos_index + 0.5) * bar_width

            gpu_bar = ax.bar(x_pos, gpu_vals, bar_width,
                             color=color, edgecolor='black',
                             hatch=gpu_hatch, linewidth=1.5)

            # For memory plots, stack CPU memory on top of GPU memory
            if plot_type == "memory":
                cpu_vals = lib_data["GPU"].get("cpu_mem", [])
                # Use lighter grayscale for stacked CPU part
                cpu_bar = ax.bar(x_pos, cpu_vals, bar_width,
                                 color=color, edgecolor='black',
                                 bottom=gpu_vals, hatch=cpu_hatch,
                                 linewidth=1.5)

            legend_handles.append(gpu_bar[0])
            legend_labels.append(f"{lib_name} (GPU)")
            pos_index += 1

    # Configure plot
    if log_scale:
        ax.set_yscale('log')
    if max_y_value:
        ax.set_ylim(0, max_y_value)
        trunk_bar(ax, max_y_value * 0.9, fontsize=14)

    # Set labels and titles
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_vals, fontsize=12)

    # Remove horizontal padding at the ends
    ax.set_xlim(-0.5, len(x_positions) - 0.5)
    ax.margins(x=0)

    # Enhance grid for better readability in grayscale
    ax.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=0.8)
    # Create a more readable legend with a white background
    legend = ax.legend(handles=legend_handles, labels=legend_labels,
                       loc='best', fontsize=10, framealpha=1,
                       edgecolor='black', facecolor='white')

    # Save figures with tight layout
    plt.tight_layout()
    filename = f"{img_name}.png" if img_name else "plot.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Plot saved as {filename}")

    return ax  # Return axis for further customization if needed


def process_stats_data(file_list, base_path, library_name, is_gpu=False):
    """Process benchmark data from files."""
    from util import get_cpu_mem_time, get_gpu_mem_time

    # Initialize empty lists for measurements
    time_measurements = []
    cpu_mem_measurements = []
    gpu_mem_measurements = [] if is_gpu else None

    # Handle empty file list
    if not file_list:
        print(
            f"No files to process for {library_name} {'GPU' if is_gpu else 'CPU'}")
        return ([], []) if not is_gpu else ([], [], [])

    # Process each file
    for i, file_name in enumerate(file_list):
        print(f"Processing {library_name} {i+1}/{len(file_list)}: {file_name}")
        try:
            if is_gpu:
                t, cpu_mem, gpu_mem = get_gpu_mem_time(base_path, file_name)
                time_measurements.append(t.values[0])
                cpu_mem_measurements.append(cpu_mem.values[0])
                gpu_mem_measurements.append(gpu_mem.values[0])
            else:
                t, cpu_mem = get_cpu_mem_time(base_path, file_name)
                time_measurements.append(t.values[0])
                cpu_mem_measurements.append(cpu_mem.values[0])
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            # Add zeros for missing values
            time_measurements.append(0)
            cpu_mem_measurements.append(0)
            if is_gpu:
                gpu_mem_measurements.append(0)
    return (time_measurements, cpu_mem_measurements, gpu_mem_measurements) if is_gpu else (time_measurements, cpu_mem_measurements)


def trunk_bar(ax, max_y, log_scale=False, fontsize=10):
    # Get the bars
    bars = ax.patches
    # Get the x values
    xb = [bar.get_x() for bar in bars]
    # Get the y values
    y = [bar.get_height() for bar in bars]
    # Get the bar width
    w = bars[0].get_width()
    arrow = chr(8593)

    # Add truncation indicator for bars exceeding max_y
    for i, (value, bar_elem) in enumerate(zip(y, xb)):
        if value > max_y:
            # Set the bar height to the maximum value
            ax.patches[i].set_height(max_y)
            # Add ellipsis to indicate truncation
            ax.text(xb[i] + w/2, max_y, arrow,
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='black', fontsize=fontsize)

            # Optional: Add full value annotation
            if log_scale:
                value = np.log10(value)
                str_value = f'10e{value:.1f}'
            else:
                str_value = f'{value:.1f}'

            ax.text(xb[i] + w/2, max_y, str_value,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='black', fontsize=fontsize)
