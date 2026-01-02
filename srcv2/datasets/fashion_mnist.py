from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx

import torchvision.transforms as transforms
import torch


class FashionMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0, noise_std: float = 0.0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        self.noise_std = noise_std

        # For now, we'll use a simple ToTensor transform.
        # We will come back to normalization later.
        transform = transforms.ToTensor()

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyFashionMNIST(root=self.root, train=True, download=True,
                                   transform=transform, target_transform=target_transform, noise_std=self.noise_std)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFashionMNIST(root=self.root, train=False, download=True,
                                       transform=transform, target_transform=target_transform, noise_std=self.noise_std)


class MyFashionMNIST(FashionMNIST):
    """Torchvision FashionMNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, noise_std: float = 0.0, **kwargs):
        super(MyFashionMNIST, self).__init__(*args, **kwargs)
        self.noise_std = noise_std

    def __getitem__(self, index):
        """Override the original method of the FashionMNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.noise_std > 0:
            img = img + torch.randn(img.size()) * self.noise_std

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
