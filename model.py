import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
from munkres import Munkres
from tqdm import tqdm


class regSet(nn.Module):


    def __init__(self, input_dim, x_mu, x_std):
        super().__init__()

        self.x_mu = x_mu
        self.x_std = x_std

        self.f = nn.Sequential(nn.Linear(input_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, input_dim)
                               )
        self.mse = nn.MSELoss()

    def generate_noise(self, X):

        n = X.shape[0]
        self.Z = torch.randn_like(X)
        dist = torch.cdist(X, self.Z, p=2.0)

        m = Munkres()
        indexes = m.compute(dist.detach().cpu().numpy())

        self.lookup_table = {}
        for i, j in indexes:
            key = tuple(X[i].cpu().tolist())
            self.lookup_table[hash(key)] = self.Z[j]

        # n = X.shape[0]
        # Z = torch.randn_like(X)
        # self.lookup_table = {}
        # for b in tqdm(range(0, n, 100)):
        #     dist = torch.cdist(X[b:b + 100], Z[b:b + 100], p=2.0)
        #     m = Munkres()
        #     indexes = m.compute(dist.detach().cpu().numpy())
        #     for i, j in indexes:
        #         key = tuple(X[b+i].cpu().tolist())
        #         self.lookup_table[hash(key)] = Z[b+j]

        # assigned = torch.ones(n, dtype=torch.bool, device=X.device)
        # # hash_ = torch.empty(n, dtype=torch.long, device=X.device)
        # self.lookup_table = {}
        #
        # for i in range(n):
        #     valid_indices = assigned.nonzero(as_tuple=True)[0]
        #     j_local = torch.argmin(dist[i, valid_indices])
        #     j = valid_indices[j_local]
        #     assigned[j] = False
        #     # hash_[i] = j
        #     key = tuple(X[i].cpu().tolist())
        #     self.lookup_table[hash(key)] = Z[j]


    def forward(self, x):
        return self.f(x)

    def get_loss(self, x):

        z = []
        for x_i in x:
            key = tuple(x_i.cpu().tolist())
            z.append(self.lookup_table[hash(key)])
        z = torch.stack(z, 0).to(x.device)

        x_hat = self(z)

        return self.mse(x, x_hat)





























