from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .fashion_mnist import FashionMNIST_Dataset


def load_dataset(dataset_name, data_path, normal_class, noise_std: float = 0.0):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'fashion_mnist')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class, noise_std=noise_std)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class, noise_std=noise_std)

    if dataset_name == 'fashion_mnist':
        dataset = FashionMNIST_Dataset(root=data_path, normal_class=normal_class, noise_std=noise_std)

    return dataset
