from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np


def create_loaders(batch_size, transform=None):
    valid_size = 0.2

    train_dataset = CustomCIFAR10(root="data", train=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return train_loader, valid_loader


class CustomCIFAR10(CIFAR10):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(CustomCIFAR10, self).__init__(
            root, train, transform, target_transform, download
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
