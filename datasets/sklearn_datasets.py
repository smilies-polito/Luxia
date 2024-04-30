import torch

from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def fetch_iris():
    print("Fetching Iris Dataset")
    iris = load_iris()
    data = iris['data']
    # if is_optical:
    scaler = MinMaxScaler()             # scaling all the data between 0 and 1
    # else:
        # scaler = StandardScaler()           # scaling all the data with average 0 and dev std 1
    data_scaled = torch.tensor(scaler.fit_transform(data), dtype=torch.float)
    labels = torch.tensor(iris['target'], dtype=torch.long)

    # Total n dat 150 => 120 for train and 30 for test
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        data_scaled, labels, test_size=0.2, random_state=2)
    return data_train_val, labels_train_val, data_test, labels_test


def fetch_digits():
    print("Start DIGITS_np_dataset")
    ds = load_digits()
    data = ds['data']
    # WARNING: when the model is not optical Federico is using StandardScaler()
    scaler = MinMaxScaler()             # scaling all the data between 0 and 1
    data_scaled = torch.tensor(scaler.fit_transform(data), dtype=torch.float)
    labels = torch.tensor(ds['target'], dtype=torch.long)

    # Total n dat 150 => 120 for train and 30 for test
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        data_scaled, labels, test_size=0.2, random_state=2)
    print(data_train_val.shape)
    return data_train_val, labels_train_val, data_test, labels_test
