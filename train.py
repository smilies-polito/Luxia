import torch
import numpy as np
import photontorch
import argparse
import os
import time

from util.init_model import init_model
from util.utils import load_dataset, expand_data_features, generate_dataloader, \
    train_epoch, val_epoch, plot_trend


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    # Parameters for the training
    parser.add_argument(
        '--dataset', type=str, default='iris', required=True,
        help='The dataset (mnist, digits, iris). Default is iris.'
    )
    parser.add_argument(
        '--epoch', type=int, default=50, required=False,
        help='The number of epochs. Default is 50.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=12, required=False,
        help='The number of samples per batch. Default is 64.'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.1, required=False,
        help='The fraction (0-1) of the original training data to be used for'
        'validation. Default is 0.1.'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, required=False,
        help='The learning rate. Default is 1e-3.'
    )
    parser.add_argument(
        '--exp_dir', type=str, default='experiment', required=False,
        help='The experiment directory. Default is experiment.'
    )
    parser.add_argument(
        '--seed', type=int, default=1, required=False,
        help='The seed. Default is 1e-4.'
    )
    parser.add_argument(
        '--verbose', type=bool, default=True, required=False,
        help='Verbosity.'
    )
    # Parameters for Photon_NN and Photontorch Models
    parser.add_argument(
        '--model', type=str, default='fldzhyan', required=True,
        help="The model to be trained (fldzhyan, clements, pt_clements)."
        "Deafult is fldzhyan"
    )
    parser.add_argument(
        '--n_inputs', type=int, default=2, required=True,
        help='The number of inputs in the photonic network. Default is 2.'
    )
    parser.add_argument(
        '--n_layers', type=int, default=2, required=False,
        help='The number of layers in the network. Default is 2.'
        'this parameter is not needed in the photontorch clements model,'
        'clements model within photontorch is NxN model'
    )
    parser.add_argument(
        '--n_networks', type=int, default=2, required=False,
        help='The number of photonic networks. Default is 2.'
    )
    parser.add_argument(
        '--capacity', type=int, default=2, required=False,
        help='The capacity of the photontorch network. Required only for the'
        'photontorch model.'
    )
    parser.add_argument('--mmi_imbalances',
                        action=argparse.BooleanOptionalAction,
                        help='imbalances of the mmi s within the models.'
                        'If True, make sure to create a mmi_imbalances.npy file'
                        )

    parser.add_argument('--mmi_i_losses',
                        action=argparse.BooleanOptionalAction,
                        help='Insertion losses of the mmi s within the models.'
                        'If True, make sure to create a mmi_i_losses.npy file.'
                        )
    args = parser.parse_args()

    # Prepare the Data Set
    torch.manual_seed(args.seed)

    print(f"[*] Loading {args.dataset} dataset")
    data_train, labels_train, data_test, labels_test = load_dataset(
        args.dataset
    )
    print()

    n_features = data_train.shape[-1]
    n_inputs = args.n_inputs
    print(f"[*] Expanding dataframe from {n_features} features to {n_inputs} inputs")
    data_train, data_test = expand_data_features(
        data_train, data_test, n_features, n_inputs
    )
    n_features = n_inputs
    print()

    print("[*] Splitting the data into training and validation"
          f"({int(args.val_frac*100):d}%) sets")

    print("And generating the dataloaders")
    datal_train, datal_val, datal_test = generate_dataloader(
        val_frac=args.val_frac,
        n_features=n_features,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        data_train=data_train,
        labels_train=labels_train,
        data_test=data_test,
        labels_test=labels_test
    )
    if args.verbose:
        print(f"\tTraining: {len(datal_train.dataset)} batches")
        print(f"\tValidation: {len(datal_val.dataset)} batches")
        print(f"\tTesting: {len(datal_test.dataset)} batches")

    print()

    # Initialize the model
    # Check if the model include losses and imbalances
    # FIX: Check why True is not working
 
    if args.mmi_imbalances:
        print("[*] Loading MMI imbalances")
        mmi_imbalances = torch.tensor(np.load('mmi_imbalances.npy'), dtype=torch.float32)
    else:
        print("[*] None Loading MMI imbalances")
        mmi_imbalances = None

    if args.mmi_i_losses:
        mmi_i_losses = torch.tensor(np.load('mmi_i_losses.npy'))
    else:
        mmi_i_losses = None

    if args.model == 'pt_clements':
        print("[*] setting enviroment for photontorch model")
        photontorch.set_environment(frequency_domain=True, enable_grad=True)

    n_classes = len(torch.unique(labels_train))

    print(f"[*] Initialize {args.model} model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = init_model(
        name=args.model,
        n_inputs=n_inputs,
        n_classes=n_classes,
        n_layers=args.n_layers,
        n_networks=args.n_networks,
        capacity=args.capacity,
        mmi_i_losses=mmi_i_losses,
        mmi_imbalances=mmi_imbalances
    ).to(device)
    os.makedirs(args.exp_dir, exist_ok=True)
    torch.save(model, os.path.join(args.exp_dir, f"model_epoch{0:03d}.pth"))

    if args.verbose:
        print(model)
        print("\nModel parameters:")
        for name, p in model.named_parameters():
            print(f"{name}, shape {p.shape}, requires grad {p.requires_grad}")

    print()

    print("[*] Define the loss function and optimizer")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr
    )

    print()

    print("[*] Start training")
    if args.verbose:
        print(f"\tEpochs: {args.epoch}")
        print(f"\tLearning rate: {args.lr}")
        print(f"\tSeed: {args.seed}")

    print()

    # Define the variables needed for epoch loop
    avg_loss_list = []
    avg_vloss_list = []
    avg_acc_list = []
    avg_vacc_list = []

    improv_counter = 0
    improv_th = 10
    avg_vloss_best = 100_000_000
    model_best = None

    plot_every = 50

    times = []

    for epoch in range(args.epoch):
        print(f'EPOCH {epoch+1}:')
        start = time.time()

        model.train()
        avg_loss, avg_acc = train_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dataloader=datal_train
        )

        # we don't need gradients to evaluate the model
        model.eval()
        avg_vloss, avg_vacc = val_epoch(
            model=model,
            loss_fn=loss_fn,
            device=device,
            dataloader=datal_val
        )
        # wandb.log({
        #     'train_loss': batch_loss,
        #     'val_loss': batch_vloss,
        # })

        avg_loss_list.append(avg_loss)
        avg_vloss_list.append(avg_vloss)
        avg_acc_list.append(avg_acc)
        avg_vacc_list.append(avg_vacc)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        print(f'ACC train {avg_acc} valid {avg_vacc}')

        end = time.time()
        deltat = end-start
        times.append(deltat)
        print()
        print(f"{deltat} seconds elapsed")
        print(flush=True)

        if avg_vloss < avg_vloss_best:
            improv_counter += 1
            avg_vloss_best = avg_vloss
            if improv_counter >= improv_th:
                print("Saving model...")
                torch.save(
                    model,
                    os.path.join(args.exp_dir, f"model_epoch{epoch+1:03d}.pth")
                )
                improv_counter = 0

        if (epoch+1) % plot_every == 0:
            plot_trend(
                avg_loss_list,
                avg_vloss_list,
                'Loss',
                args.exp_dir,
                f'loss_epoch{epoch+1:03d}.png'
            )
            plot_trend(
                avg_acc_list,
                avg_vacc_list,
                'Accuracy',
                args.exp_dir,
                f'acc_epoch{epoch+1:03d}.png'
            )

    print(f"Average epoch time: {np.mean(times)}")
    print()

