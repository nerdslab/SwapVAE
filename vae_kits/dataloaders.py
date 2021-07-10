import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset


class spatial_only_neural(Dataset):
    '''
    original unchanged neural data for vae to learn from
    '''
    def __init__(self, dataset, transform=None, target_transform=None, train='train'):

        if train == 'train':
            dataset.train()
        elif train == 'test':
            dataset.test()
        else:
            raise NotImplementedError

        self.firing_rates = torch.Tensor(dataset.firing_rates)
        self.transform, self.target_transform = transform, target_transform
        self.labels = torch.Tensor(dataset.labels)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        x, target = self.firing_rates[index, :], int(self.labels[index])

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self):
        return self.firing_rates.shape[0]


class spatial_only_neural_angle(Dataset):
    '''
    original unchanged neural data for vae to learn from.
    The label used in this class is the angle (2-dim) instead of label (8-dim)
    '''
    def __init__(self, dataset, transform=None, target_transform=None, train='train'):

        if train == 'train':
            dataset.train()
        elif train == 'test':
            dataset.test()
        else:
            raise NotImplementedError

        self.transform, self.target_transform = transform, target_transform

        velocity_threshold = 5 # -1 for better delta-acc, 5 for better acc
        velocity_mask = np.linalg.norm(dataset.velocity, 2, axis=1) > velocity_threshold
        self.firing_rates = torch.Tensor(dataset.firing_rates[velocity_mask])
        labels = dataset.labels[velocity_mask]
        angles = (2 * np.pi / 8 * labels)[:, np.newaxis]
        self.labels = torch.Tensor(np.concatenate([np.cos(angles), np.sin(angles)], axis=1))

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        x, target = self.firing_rates[index, :], self.labels[index, :]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self):
        return self.firing_rates.shape[0]

