import argparse


def init_args():
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
        '--lr', type=float, default=1e-4, required=False,
        help='The learning rate. Default is 1e-4.'
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
        '--device', type=str, default='cpu', required=False,
        help='The device to be used (cpu, cuda). Default is cpu.'
    )
    parser.add_argument(
        '--verbose', type=bool, default=True, required=False,
        help='Verbosity.'
    )
    # Parameters for Photon_NN and Photontorch Models
    parser.add_argument(
        '--model', type=str, default='fldzhyan', required=True,
        help="The model to be trained (fldzhyan, clements, pt_clements,"
        " nx_rec_mesh). Deafult is fldzhyan"
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
    parser.add_argument(
        '--imbalances', type=float, default=0.0, required=False,
        help='imbalances of the mmis or mzis within the models.'
    )
    parser.add_argument(
        '--i_losses', type=float, default=0.0, required=False,
        help='Insertion losses of the mmi s within the models.'
    )
    parser.add_argument(
        '--rnd_imbalances', type=float, default=0.0, required=False,
        help='imbalances of the mmi s within the models. from 0 to set value.'
    )
    parser.add_argument(
        '--rnd_i_losses', type=float, default=0.0, required=False,
        help='Insertion losses of the mmi s or mzis within the models. from 0'
        'to the set value.'
    )
    parser.add_argument(
        '--is_matmul',
        action=argparse.BooleanOptionalAction,
        help='The fastforward parameter that is used in the photonic model,'
        'it indicates if the model is matrix multiplication or fastforward.'
        'multiplication.'
    )
    args = parser.parse_args()
    return args
