import torch
from sklearn.model_selection import train_test_split
from datasets.pytorch_dataset import LabeledDataset


def print_torch_model_summary(model):
    print("\nModel parameters:")
    print(model)
    # print the number of parameters
    print("[*] Number of parameters:"
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # for name, p in model.named_parameters():
    #     print(f"{name}, shape {p.shape}, requires grad {p.requires_grad}")
    print()


def get_device(device, verbose):
    if device == 'cuda' or device == 'gpu':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available Changing to CPU")
            dev = torch.device('cpu')
        else:
            dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    if verbose:
        print(f"[*] Using {dev} device")
    return dev


def load_dataset(name: str, verbose: bool = True):
    """
    Load the dataset based on the name provided

    Args:
        name (str): The name of the dataset to be loaded

    Returns:
        data_train (torch.Tensor): The training data
        labels_train (torch.Tensor): The training labels
        data_test (torch.Tensor): The testing data
        labels_test (torch.Tensor): The testing labels
    """
    if verbose:
        print(f"[*] Loading {name} dataset")
        print()

    if name == 'mnist':
        from datasets.mnist_dataset import fetch_mnist
        return fetch_mnist()
    elif name == 'iris':
        from datasets.sklearn_datasets import fetch_iris
        return fetch_iris()
    elif name == 'digits':
        from datasets.sklearn_datasets import fetch_digits
        return fetch_digits()
    elif name == 'olivetti':
        from datasets.sklearn_datasets import fetch_olivetti
        return fetch_olivetti()
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def expand_data_features(data_train: torch.Tensor,
                         data_test: torch.Tensor,
                         n_features: int,
                         n_inputs: int,
                         verbose: bool = True):
    """
    Resize the input data based on the number of features and the number of
    inputs of the photonic network.

    Args:
        data_train (torch.Tensor): The training data
        data_test (torch.Tensor): The test data
        n_features (int): The number of features of the dataset
        n_inputs (int): The number of inputs of the photonic network
        verbose (bool): Whether to print function output

    Return:
        datal_train(torch.utils.data.DataLoader):
            The resized training dataloader
        datal_test(torch.utils.data.DataLoader):
            The resized testing dataloader
    """
    if n_features > n_inputs:
        raise ValueError(
            f'The number of features exceeds the capacity of the photonic'
            f'network. The maximum allowed number is: {n_inputs}'
        )

    # Resize the input data if the number of features is less than the
    if verbose:
        print(f"[*] Expanding dataframe from {n_features} features to "
              f"{n_inputs} inputs")

    # number of inputs of the photnic network
    if n_features < n_inputs:
        d_resized_list = []
        for d in [data_train, data_test]:
            d_resized = torch.zeros(
                (d.shape[0], n_inputs),
                dtype=d.dtype
            )

            start_limit = (n_inputs - n_features) // 2
            end_limit = start_limit + n_features
            d_resized[..., start_limit:end_limit] = d
            d_resized_list.append(d_resized)

        return tuple(d_resized_list)

    return (data_train, data_test)


def generate_dataloader(
    val_frac: float, n_features: int, batch_size: int, shuffle: bool,
    seed: int, data_train: torch.Tensor, labels_train: torch.Tensor,
    data_test: torch.Tensor, labels_test: torch.Tensor, verbose: bool = True
):
    """
    Generate the dataloaders for training, validation, and testing

    Args:
        val_frac (float): The fraction (0-1) of the original training data to
            be used for validation
        n_features (int): The number of features of the dataset
        batch_size (int): The batch size
        shuffle (bool): Whether to shuffle the data
        seed (int): The seed
        data_train (torch.Tensor): The training data
        labels_train (torch.Tensor): The training labels
        data_test (torch.Tensor): The test data
        labels_test (torch.Tensor): The test labels
        verbose (bool): Whether to print function information

    Returns:
        datal_train (torch.utils.data.DataLoader): The training dataloader
        datal_val (torch.utils.data.DataLoader): The validation dataloader
        datal_test (torch.utils.data.DataLoader): The testing dataloader
    """
    if verbose:
        print("[*] Splitting the data into training and validation"
              f"({int(val_frac*100):d}%) sets")

    # Split training data into training and validation
    data_full_list = train_test_split(
        torch.cat([data_train, labels_train[:, None]], dim=-1),
        test_size=val_frac,
        random_state=seed
    )
    data_full_train = data_full_list[0]
    data_full_val = data_full_list[1]

    if verbose:
        print("Generating dataloaders")
    datal_train = torch.utils.data.DataLoader(
        LabeledDataset(data_full_train, n_features),
        batch_size=batch_size, shuffle=shuffle
    )

    datal_val = torch.utils.data.DataLoader(
        LabeledDataset(data_full_val, n_features),
        batch_size=batch_size, shuffle=shuffle
    )

    datal_test = torch.utils.data.DataLoader(
        LabeledDataset(
            torch.cat([data_test, labels_test[:, None]], dim=-1), n_features),
        batch_size=batch_size,
        shuffle=shuffle
    )

    if verbose:
        print(f"[*] Training set size: {len(datal_train)}")
        print(f"[*] Validation set size: {len(datal_val)}")
        print(f"[*] Testing set size: {len(datal_test)}")
        print()

    return datal_train, datal_val, datal_test


def accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor
) -> float:
    accuracy = (torch.argmax(pred, dim=1) == labels)
    return accuracy.float().mean().detach().item()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    is_qerror_het: bool = False,
    is_qerror_dac: bool = False,
    is_qerror_adc: bool = False,
) -> float:
    """
    Train the model for one epoch

    Args:
        model (torch.nn.Module): The model
        optimizer (torch.optim.Optimizer): The optimizer
        loss_fn (torch.nn.Module): The loss function
        device (torch.device): The device to use
        dataloader (torch.utils.data.DataLoader): The dataloader

    Returns:
        avg_loss (float): The average loss
        avg_acc (float): The average accuracy
    """
    ret_values = {"loss": [], "acc": []}

    running_loss = []
    running_acc = []
    if is_qerror_het:
        ret_values["heaters_qerror"] = []
        running_htr_qerror = []

    if is_qerror_dac:
        ret_values["dac_qerror"] = []
        running_dac_qerror = []
    if is_qerror_adc:
        ret_values["adc_qerror"] = []
        running_adc_qerror = []

    for _, batch in enumerate(dataloader):
        # zero your gradients for every batch!
        optimizer.zero_grad()

        data, labels = batch
        data = data.to(device)
        labels = labels.squeeze().to(device)

        # make predictions for this batch
        pred = model(data)

        if is_qerror_het:
            q_error = model.get_heaters_average_qerror()
            running_htr_qerror.append(q_error)
        if is_qerror_dac:
            q_error = model.get_dac_average_qerror()
            running_dac_qerror.append(q_error)
        if is_qerror_adc:
            q_error = model.get_adc_average_qerror()
            running_adc_qerror.append(q_error)

        # compute the loss and its gradients
        loss = loss_fn(pred, labels)
        running_loss.append(loss.detach().item())

        loss.backward()

        # adjust learnable parameters
        optimizer.step()

        running_acc.append(accuracy(pred, labels))

    avg_loss = sum(running_loss) / len(running_loss)
    ret_values["loss"].append(avg_loss)
    avg_acc = sum(running_acc) / len(running_acc)
    ret_values["acc"].append(avg_acc)
    if is_qerror_het:
        avg_qerror_het = sum(running_htr_qerror) / len(running_htr_qerror)
        ret_values["heaters_qerror"].append(avg_qerror_het)
    if is_qerror_dac:
        avg_qerror_dac = sum(running_dac_qerror) / len(running_dac_qerror)
        ret_values["dac_qerror"].append(avg_qerror_dac)
    if is_qerror_adc:
        avg_qerror_adc = sum(running_adc_qerror) / len(running_adc_qerror)
        ret_values["adc_qerror"].append(avg_qerror_adc)
    return ret_values


def val_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader
) -> float:
    """
    Validate the model for one epoch

    Args:
        model (torch.nn.Module): The model
        loss_fn (torch.nn.Module): The loss function
        device (torch.device): The device to use
        dataloader (torch.utils.data.DataLoader): The dataloader

    Returns:
        avg_vloss (float): The average validation loss
        avg_vacc (float): The average validation accuracy
    """

    with torch.no_grad():
        running_vloss = []
        running_vacc = []

        for _, batch in enumerate(dataloader):

            data, labels = batch
            data = data.to(device)
            labels = labels.squeeze().to(device)

            pred = model(data)

            loss = loss_fn(pred, labels)
            running_vloss.append(loss.detach().item())

            running_vacc.append(accuracy(pred, labels))

        avg_vloss = sum(running_vloss) / len(running_vloss)
        avg_vacc = sum(running_vacc) / len(running_vacc)

    return avg_vloss, avg_vacc


def get_trained_model(model_path):
    """Load the trained PyTorch model."""
    try:
        print(f"Trying to load model from '{model_path}'")
        model = torch.load(f=model_path, weights_only=False)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found at '{model_path}'.")
    except Exception as e:
        print(f"Error loading model: {e}")
    return None


def test_model(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader
) -> float:
    """
    Test the model

    Args:
        model (torch.nn.Module): The model
        loss_fn (torch.nn.Module): The loss function
        device (torch.device): The device to use
        dataloader (torch.utils.data.DataLoader): The dataloader

    Returns:
        avg_tloss (float): The average test loss
        avg_tacc (float): The average test accuracy
    """
    # set the model to evaluation mode
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        running_tloss = []
        running_tacc = []

        for _, batch in enumerate(dataloader):

            data, labels = batch
            data = data.to(device)
            labels = labels.squeeze().to(device)

            pred = model(data)

            loss = loss_fn(pred, labels)
            running_tloss.append(loss.detach().item())

            running_tacc.append(accuracy(pred, labels))

        avg_tloss = sum(running_tloss) / len(running_tloss)
        avg_tacc = sum(running_tacc) / len(running_tacc)

    return avg_tloss, avg_tacc
