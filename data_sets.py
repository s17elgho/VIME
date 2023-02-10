import pandas as pd
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

class TabUnlabDataset(Dataset):
    def __init__(self, x_unlab, x_tilde, m_label, transform=transforms.ToTensor()):
        self.original = x_unlab
        self.corrupted = x_tilde
        self.mask = m_label
        self.transform = transform

    def __len__(self):
        return self.original.shape[0]

    def __getitem__(self, index):

        x = self.transform(self.original[index].reshape((1,784)))
        x_tilde = self.transform(self.corrupted[index].reshape((1,784)))
        mask = self.transform(self.mask[index].reshape((1,784)))

        return (x, x_tilde, mask)

class TabSemiUnlabDataset(Dataset):
    def __init__(self, x_unlab, transform=transforms.ToTensor()):
        self.original = x_unlab
        self.transform = transform

    def __len__(self):
        return self.original.shape[0]

    def __getitem__(self, index):

        x = self.transform(self.original[index].reshape((1,784)))

        return (x)

class TabLabDataset(Dataset):
    def __init__(self, x_train, y_train,transform=transforms.ToTensor()):
        self.x = x_train
        self.y = y_train
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):

        x = self.transform(self.x[index].reshape((1,784)))
        y = self.transform(self.y[index].reshape((1,10)))

        return (x, y)

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)