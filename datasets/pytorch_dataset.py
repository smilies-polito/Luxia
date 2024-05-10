import torch
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    """
    Dataset wrapping labeled data tensors.

    Each sample will be retrieved by indexing tensors along the first
    dimension.

    Arguments:
        data_full (torch.Tensor): The data samples
        n_inputs (int): The number of inputs (features) of the dataset
    """

    def __init__(self, data_full: torch.Tensor, n_inputs: int):
        self.data_full = data_full
        self.n_inputs = n_inputs

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data_full[index, :self.n_inputs], self.data_full[index, self.n_inputs:].long()

    def __len__(self) -> int:
        return len(self.data_full)
