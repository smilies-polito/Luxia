import seaborn as sns
from load_files.define_sota_files import define_sota_file_paths
from util import process_library_data, create_vacc_tloss_plots

def plot_digits_dataset(dic_of_files, base_path, ds_color="dark:green"):
    """Process the digits dataset and return the results."""
    # Define color palette for consistent visualization
    my_colors = sns.color_palette(ds_color, n_colors=len(dic_of_files))

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
    # Plot Validation Accuracy and Training Loss
    create_vacc_tloss_plots(data, img_name=f"plot_sota")
    # Print results
    print(f"\nFinal Mean Accuracy and Loss:")
    for name in data.keys():
        print(f"{name}:")
        print(f"Mean Validation Accuracy: {data[name]['mean_val_acc'][-1]}")
        print(f"Mean Training Loss: {data[name]['mean_train_loss'][-1]}")

def main():
    """Main function to orchestrate data processing and visualization for all datasets."""
    files = define_sota_file_paths()
    # Set base path
    base_path = f"../results/digits/train"
    col_palette = "colorblind"  # Define color palette for digits dataset
    plot_digits_dataset(files, base_path,ds_color=col_palette)
    print("\nAll datasets processed successfully.")

if __name__ == "__main__":
    main()
