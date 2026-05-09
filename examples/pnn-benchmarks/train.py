import wandb
import torch
import numpy as np
import os
import random
import csv

from util.init_args import init_args
from util.init_model import init_model
from util.utils import load_dataset, expand_data_features
from util.utils import generate_dataloader
from util.utils import get_device
from util.mmi_imbalances import get_tensor_imbalances
from util.mmi_i_losses import get_tensor_i_losses
from util.train_torch_model import train_torch_model


def train_neuroptica_model(model, data, labels, datal_val, n_classes, args):
    data_lists = {
        "loss": [], "time": [], "cpu_mem": [], "val_accuracy": []
    }
    if args.verbose:
        print("[*] Define the neuroptica loss function and optimizer")
        print()

    import neuroptica as neu
    from neu_models.neu_forked_optimizer_gd import ForkedOptimizerAdam
    optimizer = ForkedOptimizerAdam(
        model,
        neu.CategoricalCrossEntropy,
        args.lr,
    )

    if args.verbose:
        print(f"[*] Start training \t EPOCHS:{args.epoch} \t LR:{args.lr}"
              f"\t BATCH_SIZE:{args.batch_size} \t VAL_FRAC:{args.val_frac}"
              f"\t SEED:{args.seed}")
        print()
    labels_1hot = np.zeros((labels.size, n_classes))
    labels_1hot[np.arange(labels.size), labels] = 1

    data_val_ds = datal_val.dataset
    data_val = data_val_ds.data_full[:, :data_val_ds.n_inputs].numpy()
    labels_val = data_val_ds.data_full[:,
                                       data_val_ds.n_inputs:].numpy().astype(int)

    labels_val_1hot = np.zeros((labels_val.size, n_classes))
    labels_val_1hot[np.arange(labels_val.size), labels_val.T] = 1

    losses, acc, times, mem = optimizer.fit(
        data=data.T,
        labels=labels_1hot.T,
        data_val=data_val.T,
        labels_val=labels_val_1hot.T,
        epochs=args.epoch,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    data_lists['loss'] = losses
    data_lists['val_accuracy'] = acc
    data_lists['time'] = times
    data_lists['cpu_mem'] = mem

    return data_lists


def save_data_train(filename, data_lists):
    print("Saving data...")
    with open(filename, mode='w', newline="") as file:
        writer = csv.writer(file)
        # Write the header
        headers = data_lists.keys()
        writer.writerow(headers)
        # Write the rows by zipping values of each key
        rows = zip(*data_lists.values())
        writer.writerows(rows)


if __name__ == '__main__':

    # Initialize the arguments
    args = init_args()

    # Check if arg.wandb is set
    if args.use_wandb:
        wandb.login(key=args.wandb_key)

    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_train, labels_train, data_test, labels_test = load_dataset(
        args.dataset, args.verbose
    )

    n_features = data_train.shape[-1]
    n_inputs = args.n_inputs

    data_train, data_test = expand_data_features(
        data_train, data_test, n_features, n_inputs, args.verbose
    )
    n_features = n_inputs

    datal_train, datal_val, datal_test = generate_dataloader(
        val_frac=args.val_frac,
        n_features=n_features,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        data_train=data_train,
        labels_train=labels_train,
        data_test=data_test,
        labels_test=labels_test,
        verbose=args.verbose
    )

    # Initialize the model
    # Check if the model include losses and imbalances
    mmi_imbalances = None
    if args.imbalance != 0.0:
        mmi_imbalances = get_tensor_imbalances(
            args.imbalance, args.rnd_imbalance,
            args.model, args.n_networks,
            args.n_layers, n_inputs, args.verbose
        )

    mmi_i_losses = None
    if args.i_loss != 0.0:
        mmi_i_losses = get_tensor_i_losses(
            args.i_loss, args.rnd_i_loss,
            args.model, args.n_networks,
            args.n_layers, n_inputs, args.verbose
        )

    if args.model == 'pt_clements':
        import photontorch
        print("[*] setting enviroment for photontorch model")
        photontorch.set_environment(frequency_domain=True, enable_grad=True)

    n_classes = len(torch.unique(labels_train))

    args.device = get_device(args.device, args.verbose)

    run_name = f"lr_{args.lr}_ep_{args.epoch}_bs_{args.batch_size}_dev_{args.device}_{args.is_matmul}"
    if args.use_wandb:
        wandb.init(project=args.wandb_proj, config=args, name=run_name)

    os.makedirs(args.exp_dir, exist_ok=True)
    model = init_model(name=args.model,
                       device=args.device,
                       n_inputs=n_inputs,
                       n_classes=n_classes,
                       n_layers=args.n_layers,
                       n_networks=args.n_networks,
                       mmi_i_losses=mmi_i_losses,
                       mmi_imbalances=mmi_imbalances,
                       is_matmul=args.is_matmul)

    if args.model == 'neu_clements':
        data_lists = train_neuroptica_model(model, data_train.numpy(),
                                            labels_train.numpy(),
                                            datal_val,
                                            n_classes,
                                            args)
    else:
        model.to(args.device)
        data_lists = train_torch_model(
            model, datal_train, datal_val, args)

    print(f"Average epoch time: {np.mean(data_lists['time']):.2f} seconds")

    if args.device.type == 'cuda':
        print(
            f"Average GPU memory used: {np.mean(data_lists['gpu_mem']):.2f} MB")

    print(f"Average CPU memory used: {np.mean(data_lists['cpu_mem']):.2f} MB")
    print(f"Average Loss: {np.mean(data_lists['loss']):.2f}")
    print(
        f"Average Validation Accuracy: {np.mean(data_lists['val_accuracy']):.2f}")

    filename = os.path.join(args.exp_dir, f"{run_name}.csv")
    save_data_train(filename, data_lists)

    if args.use_wandb:
        wandb.finish()
