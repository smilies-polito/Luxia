from comparison_sota import main as main_sota
from luxia_ds_plots import main as main_luxia
from n_size import main as main_n_size
from batch_size import main as main_batch_size

def main():
    """Main function to orchestrate the execution of different modules."""
    print("Plotting all results...")
    
    # Call each module's main function
    main_sota()
    main_luxia()
    main_n_size()
    main_batch_size()
    print("All datasets processed successfully.")
    
if __name__ == "__main__":
    main()


