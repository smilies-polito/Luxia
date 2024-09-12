# TODO: delete wandb before publishing
# import wandb
import torch
import numpy as np
import os
import time

from util.init_args import init_args
from util.init_model import init_model
from util.mem_usage import get_mem_usage
from util.utils import load_dataset, expand_data_features, generate_dataloader, train_epoch, val_epoch, plot_trend

# TODO: Uncomment the following line to login to wandb and delete before publishing
# wandb.login(key="a2d765b3686313fdb440d0d239660d944e97da5f")

if __name__ == '__main__':

    # Initialize the arguments
    args = init_args()
    # Prepare the Data Set
    torch.manual_seed(args.seed)

    # Load the dataset
    print(f"[*] Loading {args.dataset} dataset")
    data_train, labels_train, data_test, labels_test = load_dataset(
        args.dataset
    )
    print()

    # Expand the data features
    n_features = data_train.shape[-1]
    n_inputs = args.n_inputs
    print(
        f"[*] Expanding dataframe from {n_features} features to {n_inputs} inputs")
    data_train, data_test = expand_data_features(
        data_train, data_test, n_features, n_inputs
    )
    n_features = n_inputs
    print()

    # Split the data into training and validation sets and
    # generate the dataloaders
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
    mmi_imbalances = None
    if args.imbalances != 0.0:
        print("[*] Loading MMI imbalances")
        if args.model == 'clements' or args.model == 'clements_bell':
            from mmi_imbalances import get_mzi_fix_imbalances
            mmi_imbalances = get_mzi_fix_imbalances(
                args.n_networks, args.n_layers, n_inputs, args.imbalances)
        elif args.model == 'fldzhyan' or args.model == 'fldzhyan_bell':
            from mmi_imbalances import get_mmi_fix_imbalances
            mmi_imbalances = get_mmi_fix_imbalances(
                args.n_networks, args.n_layers, n_inputs, args.imbalances)

        print(" Network 0 Imbalances:")
        print(mmi_imbalances[0])
        print(" Network 1 Imbalances:")
        print(mmi_imbalances[1])

    if args.rnd_imbalances != 0.0:
        print("[*] Loading random MMI imbalances")
        if args.model == 'clements' or args.model == 'clements_bell':
            from mmi_imbalances import get_mzi_rand_imbalances
            mmi_imbalances = get_mzi_rand_imbalances(
                args.n_networks, args.n_layers, n_inputs, args.rnd_imbalances)
        elif args.model == 'fldzhyan' or args.model == 'fldzhyan_bell':
            from mmi_imbalances import get_mmi_rand_imbalances
            mmi_imbalances = get_mmi_rand_imbalances(
                args.n_networks, args.n_layers, n_inputs, args.rnd_imbalances)

        print(" Network 0 Imbalances:")
        print(mmi_imbalances[0])
        print(" Network 1 Imbalances:")
        print(mmi_imbalances[1])

    mmi_i_losses = None
    if args.i_losses != 0.0:
        print("[*] Loading MMI insertion losses")
        if args.model == 'clements' or args.model == 'clements_bell':
            from mmi_i_losses import get_mzi_fix_losses
            mmi_i_losses = get_mzi_fix_losses(
                args.n_networks, args.n_layers, n_inputs, args.i_losses)
        elif args.model == 'fldzhyan' or args.model == 'fldzhyan_bell':
            from mmi_i_losses import get_mmi_fix_losses
            mmi_i_losses = get_mmi_fix_losses(
                args.n_networks, args.n_layers, n_inputs, args.i_losses)
        print(" Network 0 MMI Insertion Losses:")
        print(mmi_i_losses[0])
        print(" Network 1 MMI Insertion Losses:")
        print(mmi_i_losses[1])

    if args.rnd_i_losses != 0.0:
        print("[*] Loading random MMI insertion losses")
        if args.model == 'clements' or args.model == 'clements_bell':
            from mmi_i_losses import get_mzi_rand_losses
            mmi_i_losses = get_mzi_rand_losses(
                args.n_networks, args.n_layers, n_inputs, args.rnd_i_losses)
        elif args.model == 'fldzhyan' or args.model == 'fldzhyan_bell':
            from mmi_i_losses import get_mmi_rand_losses
            mmi_i_losses = get_mmi_rand_losses(
                args.n_networks, args.n_layers, n_inputs, args.rnd_i_losses)

        print(" Network 0 MMI Insertion Losses:")
        print(mmi_i_losses[0])
        print(" Network 1 MMI Insertion Losses:")
        print(mmi_i_losses[1])

    n_classes = len(torch.unique(labels_train))

    print(f"[*] Initialize {args.model} model")
    print()

    # Check if the device is available
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available")
            device = torch.device('cpu')
            args.device = 'cpu'
        else:
            device = torch.device('cuda')
            args.device = 'cuda'
    else:
        device = torch.device('cpu')

    print(f"[*] Using {device} device")

    #TODO: Uncomment the following line to initialize wandb and delete before publishing
    # wandb.init(project=args.model, config=args)

    model = init_model(
        name=args.model,
        device=device,
        n_inputs=n_inputs,
        n_classes=n_classes,
        n_layers=args.n_layers,
        n_networks=args.n_networks,
        mmi_i_losses=mmi_i_losses,
        mmi_imbalances=mmi_imbalances,
        is_matmul=args.is_matmul
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
    memory = []
    if device.type == 'cuda':
        mem_gpu = []

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
        #     'train_loss': avg_loss,
        #     'val_loss': avg_vloss,
        #     'train_acc': avg_acc,
        #     'val_acc': avg_vacc,
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
        get_mem_usage(device)
        print()

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
    print("Training finished")
    # wandb.finish()
