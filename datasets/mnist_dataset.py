import torchvision.transforms as transforms
import torchvision.datasets as datasets


def fetch_mnist():

    print("Fetching MNIST Data set")
    data_train = datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor())

    labels_train = data_train.targets
    data_train = data_train.data
    # Reshape the tensor to a single vector
    # 28x28 = 784
    reshaped_tensor = data_train.view(-1, 28 * 28)
    # Normalize the values to be between 0 and 1
    data_train = reshaped_tensor.float() / 255.0
    print('data_train', data_train.shape)
    print('labels_train', labels_train.shape)

    data_test = datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor())
    labels_test = data_test.targets
    data_test = data_test.data
    reshaped_tensor = data_test.view(-1, 28 * 28)
    # Normalize the values to be between 0 and 1
    data_test = reshaped_tensor.float() / 255.0
    print('data_test', data_test.shape)
    print('labels_test', labels_test.shape)
    
    return data_train, labels_train, data_test, labels_test
