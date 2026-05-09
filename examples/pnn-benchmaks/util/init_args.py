import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Train the model.')
    # Parameters for the training
    parser.add_argument(
        '--dataset', type=str, default='iris', required=True,
        help='The dataset (mnist, digits, iris, olivetti). Default is iris.'
    )
    parser.add_argument(
        '--epoch', type=int, default=50, required=False,
        help='The number of epochs. Default is 50.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, required=False,
        help='The number of samples per batch. Default is 64.'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.1, required=False,
        help='The fraction (0-1) of the original training data to be used for'
        'validation. Default is 0.1.'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, required=False,
        help='The learning rate. Default is 1e-4.'
    )
    parser.add_argument(
        '--exp_dir', type=str, default='results', required=False,
        help='The experiment directory. Default is results.'
    )
    parser.add_argument(
        '--seed', type=int, default=1, required=False,
        help='The seed. Default is 1e-4.'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', required=False,
        help='The device to be used (cpu, cuda). Default is cpu.'
    )
    parser.add_argument(
        '--verbose', type=bool, default=True, required=False,
        help='Verbosity.'
    )

    # Parameters for luxia models
    parser.add_argument(
        '--model', type=str, default='lx_clements', required=True,
        help="available models: LuxIA's models (clements, "
        "clements_bell, fldzhyan, fldzhyan_bell,"
        "ful_clements"
    )
    parser.add_argument(
        '--n_inputs', type=int, default=4, required=True,
        help='The number of inputs in the photonic mesh. Default is 4.'
    )
    parser.add_argument(
        '--n_layers', type=int, default=4, required=False,
        help='The number of layers in the photonic mesh. Default is 4.'
        'this parameter is not needed in the Photontorch clements model,'
        ' since by default is NxN model'
    )
    parser.add_argument(
        '--window_size', type=int, default=1, required=False,
        help='The number of photonic layers grouped into each window. '
             'Default is 1.'
    )

    parser.add_argument(
    '--mode', type=str, default=None, required=False,
    help='Benchmark mode: ref, full, or slice.'
    )

    parser.add_argument(
        '--n_networks', type=int, default=2, required=False,
        help='The number of photonic mesh. Default is 2.'
    )
    parser.add_argument(
        '--imbalance', type=float, default=0.0, required=False,
        help='If rnd_imbalance is False, the imbalance is fixed to the setted'
        'value meaning that all the mmis will have the same imbalance.'
        'Default is 0.0. otherwise, the imbalance is random from 0 to'
        'the setted value. and all the mmis will have different imbalances.'
    )
    parser.add_argument(
        '--i_loss', type=float, default=0.0, required=False,
        help='If rnd_i_losses is False, the insertion losses are fixed to the'
        'setted value meaning that all the mmis will have the same insertion'
        'losses. Default is 0.0. otherwise, the insertion losses are random'
        'from 0 to the setted value. and all the mmis will have different'
        'insertion losses.'
    )

    # Parameters for IO components (when needed)
    parser.add_argument(
        '--dac_res', type=int, default=None, required=False,
        help='Bit resolution of the DAC.'
    )
    parser.add_argument(
        '--adc_res', type=int, default=None, required=False,
        help='Bit resolution of the ADC.'
    )
    parser.add_argument(
        '--heaters_res', type=int, default=None, required=False,
        help='Bit resolution of the heaters.'
    )
    parser.add_argument(
        '--round_type', type=str, default='ceil', required=False,
        help='The type of rounding to be used (ceil, floor, round).'
    )

    parser.add_argument(
        '--rnd_imbalance',
        action=argparse.BooleanOptionalAction,
        help='If active, the imbalance is random from 0 to the setted value'
        'in the imbalance parameter.'
    )
    parser.add_argument(
        '--rnd_i_loss',
        action=argparse.BooleanOptionalAction,
        help='If active, the insertion losses are random from 0 to the setted'
        'value in the i_losses parameter.'
    )
    parser.add_argument(
        '--is_matmul',
        action=argparse.BooleanOptionalAction,
        help='The fastforward parameter that is used in the photonic model,'
        'it indicates if the model is matrix multiplication or fastforward.'
        'multiplication.'
    )
    parser.add_argument(
        '--save_model',
        action=argparse.BooleanOptionalAction,
        help='if active, the model is saved, otherwise it is not saved.'
    )

    # Testing parameters
    parser.add_argument(
        '--path_model', type=str, default=None, required=False,
        help='The path of the model to be loaded.'
    )
    parser.add_argument(
        '--use_wandb',
        action=argparse.BooleanOptionalAction,
        help='if active, the model will be logged to wandb, to save the'
        'results.'
    )
    parser.add_argument(
        '--wandb_key', type=str, default=None, required=False,
        help='The wandb key to log the results.'
    )
    parser.add_argument(
        '--wandb_proj', type=str, default=None, required=False,
        help='The wandb project to log the results.'
    )

    args = parser.parse_args()
    return args
