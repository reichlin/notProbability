import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class CircularMetric(nn.Module):


    def __init__(self, X, z_dim):
        super().__init__()

        N = X.shape[0]
        x_dim = X.shape[1]

        self.diff_x = torch.max(X, 0)[0] - torch.min(X, 0)[0]
        self.mean_x = torch.mean(X, 0)

        # self.Z = torch.tanh(torch.randn([N, z_dim]))
        self.Z = nn.Parameter(torch.rand(N, z_dim).float() * 2 - 1, requires_grad=True)

        self.lookup_table = {}
        for i in range(N):
            key = tuple(X[i].cpu().tolist())
            self.lookup_table[hash(key)] = self.Z[i]

        self.f = nn.Sequential(nn.Linear(z_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, x_dim)
                               )

        self.mse = nn.MSELoss()

    def norm_z(self, z):
        return (torch.tanh(z) + self.mean_x) * self.diff_x

    def sample_z(self, n):
        return self.norm_z(torch.rand(n, self.Z.shape[-1]).float() * 2 - 1)

    def forward(self, x):
        return self.f(x)

    def get_loss(self, x):

        N = self.Z.shape[0]
        min_d = torch.max(self.diff_x) / np.sqrt(N)

        z = []
        for x_i in x:
            key = tuple(x_i.cpu().tolist())
            z.append(self.lookup_table[hash(key)])

        # z = torch.tanh(torch.stack(z, 0).to(x.device))
        z = self.norm_z(torch.stack(z, 0).to(x.device))

        # L_cost = torch.mean(torch.sum((z - x)**2, -1))
        d_xz = torch.linalg.norm(z - x, dim=-1, ord=2)
        L_cost = torch.clamp(d_xz, min=min_d).mean()
        # L_spread = - torch.log(torch.cdist(z, z).mean() + 1e-4)
        all_z_dist = torch.cdist(z, z)
        mask = all_z_dist < min_d
        L_spread = - (all_z_dist[mask]).mean()
        #L_spread = - torch.clamp(torch.cdist(z, z), max=min_d).sum()

        x_hat = self(z.detach())

        L_dec = self.mse(x, x_hat)

        # plt.scatter(x[:, 0].detach().cpu().numpy(), x[:, 1].detach().cpu().numpy(), alpha=0.2)
        # plt.scatter(z[:, 0].detach().cpu().numpy(), z[:, 1].detach().cpu().numpy(), alpha=0.2)
        # for i in range(N):
        #     xx = [x[i, 0].detach().cpu().numpy(), z[i, 0].detach().cpu().numpy()]
        #     yy = [x[i, 1].detach().cpu().numpy(), z[i, 1].detach().cpu().numpy()]
        #     plt.plot(xx, yy)
        # plt.show()

        return L_cost, L_spread, L_dec





























