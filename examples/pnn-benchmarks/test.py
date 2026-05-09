import wandb
import torch
import numpy as np
import os
import random
# import csv

from util.init_args import init_args
from util.utils import load_dataset, expand_data_features, \
    generate_dataloader, test_model
from util.load_model import load_model
from util.utils import get_device


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

    n_classes = len(torch.unique(labels_train))

    args.device = get_device(args.device, args.verbose)

    run_name = f"bs_{args.batch_size}_{args.device}"

    if args.dac_res is not None:
        run_name += f"_dac_{args.dac_res}"

    if args.adc_res is not None:
        run_name += f"_adc_{args.adc_res}"

    if args.heaters_res is not None:
        run_name += f"_het_{args.heaters_res}"

    if args.use_wandb:
        wandb.init(project=args.model, config=args, name=run_name)

    os.makedirs(args.exp_dir, exist_ok=True)
    model = load_model(
        name=args.model,
        n_inputs=args.n_inputs,
        n_classes=n_classes,
        dac_resolution=args.dac_res,
        adc_resolution=args.adc_res,
        het_resolution=args.heaters_res,
        pth_model=args.path_model
    )
    # Move the corresponding model to the device
    model.to(args.device)
    test_loss, test_accuracy = test_model(model=model,
                                          loss_fn=torch.nn.CrossEntropyLoss(),
                                          device=args.device,
                                          dataloader=datal_test)

    print(f"Average Test Loss: {test_loss:.5f}")
    print(f"Average Test Accuracy: {test_accuracy:.5f}")
    #
    # filename = os.path.join(args.exp_dir, f"{run_name}.csv")
    # save_data_train(filename, data_lists)

    if args.use_wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
        })
        wandb.finish()
