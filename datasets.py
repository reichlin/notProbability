import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll, make_moons


class Dataset_Toy(Dataset):

    def __init__(self, type=0, n_samples=100000, noise=0.0):

        if type == '2d_swiss_roll':
            X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
            X = np.concatenate((X[:, 0:1], X[:, 2:3]), -1)
        elif type == '3d_swiss_roll':
            X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        elif type == 'moons':
            X, _ = make_moons(n_samples=n_samples, noise=noise)

        X = (X - np.mean(X)) / np.std(X)
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]
