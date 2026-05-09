"""
Visualization Script for Comparing Multiple Neural Network Libraries

This script compares training performance metrics across four different neural network libraries:
- Neurophox
- Photontorch
- Neuroptica
- LuxIA

For each library, it loads validation accuracy and training loss data from multiple runs,
calculates mean, min, and max values, and creates visualizations with confidence bands.
The script generates both color and black-and-white friendly versions of plots using
distinct line styles, markers, and hatching patterns for accessibility.

Usage:
    python plot_comparison.py

Author: Tzamn Melendez
Date: May 10, 2025
"""

import seaborn as sns
from util import process_library_data, create_vacc_tloss_plots
from load_files.define_lx_files import define_lx_digits_file_paths
from load_files.define_lx_files import define_lx_iris_file_paths
from load_files.define_lx_files import define_lx_olivetti_file_paths
from load_files.define_lx_files import define_lx_mnist_file_paths


def plot_dataset(dic_of_files, base_path, dataset_name, ds_color="dark:green"):
    """Process a specific dataset and return the results."""
    # Define color palette for consistent visualization
    my_colors = sns.color_palette(ds_color, n_colors=len(dic_of_files))

    print(f"\nProcessing {dataset_name} dataset:")
    print(f"Using base path: {base_path}")

    data = {}
    for i, name in enumerate(dic_of_files.keys()):
        ret_val = process_library_data(dic_of_files[name], base_path, name)
        data[name] = {
            "mean_val_acc": ret_val[0],
            "min_val_acc": ret_val[1],
            "max_val_acc": ret_val[2],
            "mean_train_loss": ret_val[3],
            "min_train_loss": ret_val[4],
            "max_train_loss": ret_val[5],
            "color": my_colors[i]
        }

    # Create plots for this dataset
    create_vacc_tloss_plots(data, img_name=f"plot_lx_{dataset_name}")

    # Print results
    print(f"\nFinal Mean Accuracy and Loss for {dataset_name}:")
    for name in data.keys():
        print(f"{name}:")
        print(f"Mean Validation Accuracy: {data[name]['mean_val_acc'][-1]}")
        print(f"Mean Training Loss: {data[name]['mean_train_loss'][-1]}")


def main():
    """Main function to orchestrate data processing and visualization for all datasets."""
    # List of all datasets to process
    datasets = ["digits", "iris", "olivetti", "mnist"]
    col_palette = ["dark:green", "dark:blue", "dark:orange", "dark:magenta"]
    for i, dataset_name in enumerate(datasets):
        # Get file paths and set base path based on dataset name
        if dataset_name == "digits":
            files = define_lx_digits_file_paths()
        elif dataset_name == "iris":
            files = define_lx_iris_file_paths()
        elif dataset_name == "olivetti":
            files = define_lx_olivetti_file_paths()
        elif dataset_name == "mnist":
            files = define_lx_mnist_file_paths()
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

        # Set base path
        base_path = f"../results/{dataset_name}/train"
        plot_dataset(files, base_path, dataset_name, ds_color=col_palette[i])

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
