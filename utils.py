import torch
from sklearn.model_selection import train_test_split
from datasets.pytorch_dataset import LabeledDataset
import matplotlib.pyplot as plt
import os


def load_dataset(name:str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    if name == 'mnist':
        from datasets.mnist_dataset import fetch_mnist
        return fetch_mnist()
    elif name == 'iris':
        from datasets.sklearn_datasets import fetch_iris
        return fetch_iris()
    elif name == 'digits':
        from datasets.sklearn_datasets import fetch_digits
        return fetch_digits()
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def generate_dataloader(
    val_frac:float, n_inputs:int, batch_size:int, shuffle:bool, seed:int,
    data_train:torch.Tensor, labels_train:torch.Tensor,
    data_test:torch.Tensor, labels_test:torch.Tensor
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Generate the dataloaders for training, validation, and testing

    Args:
        val_frac (float): The fraction (0-1) of the original training data to be used for validation
        n_inputs (int): The number of inputs (features) of the dataset
        batch_size (int): The batch size
        shuffle (bool): Whether to shuffle the data
        seed (int): The seed
        data_train (torch.Tensor): The training data
        labels_train (torch.Tensor): The training labels
        data_test (torch.Tensor): The test data
        labels_test (torch.Tensor): The test labels

    Returns:
        datal_train (torch.utils.data.DataLoader): The training dataloader
        datal_val (torch.utils.data.DataLoader): The validation dataloader
        datal_test (torch.utils.data.DataLoader): The testing dataloader
    """
    # Split training data into training and validation
    data_full_list = train_test_split(
        torch.cat([data_train, labels_train[:, None]], dim=-1),
        test_size=val_frac,
        random_state=seed
    )
    data_full_train = data_full_list[0]
    data_full_val = data_full_list[1]

    datal_train = torch.utils.data.DataLoader(LabeledDataset(data_full_train, n_inputs), batch_size=batch_size, shuffle=shuffle)
    datal_val = torch.utils.data.DataLoader(LabeledDataset(data_full_val, n_inputs), batch_size=batch_size, shuffle=shuffle)
    datal_test = torch.utils.data.DataLoader(
        LabeledDataset(torch.cat([data_test, labels_test[:, None]], dim=-1), n_inputs),
        batch_size=batch_size,
        shuffle=shuffle
    )

    return datal_train, datal_val, datal_test

    
def init_model(
    name:str,
    n_inputs:int,
    n_classes:int,
    n_layers:int,
    n_networks:int,
    capacity:int=None
) -> torch.nn.Module:
    """
    Initialize the model based on the name provided
    
    Args:
        name (str): The name of the model to be initialized
        n_inputs (int): The number of inputs (features) of the dataset
        n_classes (int): The number of classes of the dataset
        n_layers (int): The number of layers
        n_networks (int): The number of photonic networks
        capacity (int): The capacity needed only for the photontorch model

    Returns:
        model (torch.nn.Module): The model
    """

    if name == 'fldzhyan':
        from photonic_models.fldzhyan import FldzhyanModel
        return FldzhyanModel(n_inputs, n_layers, n_classes, n_networks)
    elif name == 'clements':
        from photonic_models.clements import clementsModel
        return clementsModel(n_inputs, n_layers, n_classes, n_networks)
    elif name == 'pt_clements':
        from pt_models.pt_clements_model import PtClementsModel
        if capacity is None:
            raise ValueError('Capacity is required for the PtClementsModel')
        return PtClementsModel(n_inputs, n_layers, n_classes, capacity, n_networks)
    else:
        raise ValueError('Unknown model: {}'.format(name))
    

def accuracy(
    pred:torch.Tensor,
    labels:torch.Tensor
) -> float:
    accuracy = (torch.argmax(pred, dim=1) == labels)
    return accuracy.float().mean().detach().item()

def train_epoch(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    loss_fn:torch.nn.Module,
    device:torch.device,
    dataloader:torch.utils.data.DataLoader
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
    
    running_loss = []
    running_acc = []

    for _, batch in enumerate(dataloader):
        # zero your gradients for every batch!
        optimizer.zero_grad()

        data, labels = batch
        data = data.to(device)
        labels = labels.squeeze().to(device)

        # make predictions for this batch
        pred = model(data)

        # compute the loss and its gradients
        loss = loss_fn(pred, labels)
        running_loss.append(loss.detach().item())

        loss.backward()
    
        # adjust learnable parameters
        optimizer.step()

        running_acc.append(accuracy(pred, labels))

    avg_loss = sum(running_loss) / len(running_loss)
    avg_acc = sum(running_acc) / len(running_acc)
    
    return avg_loss, avg_acc

def val_epoch(
    model:torch.nn.Module,
    loss_fn:torch.nn.Module,
    device:torch.device,
    dataloader:torch.utils.data.DataLoader
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


def plot_trend(
    train_list:list[float],
    val_list:list[float],
    title:str,
    exp_dir:str,
    filename:str
):
    """
    Plot train and validation measures

    Args:
        train_list (list[float]): The list of training measures
        val_list (list[float]): The list of validation measures
        title (str): The title of the plot
        exp_dir (str): The experiment directory
        filename (str): The filename to save the plot

    Returns:
        None
    """
    plt.figure()
    plt.plot(train_list, c='blue')
    plt.plot(val_list, c='orange')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(exp_dir, filename))
    plt.close()