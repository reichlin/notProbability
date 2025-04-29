import numpy as np
import torch
from torch import nn
from torch import vmap
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from scipy.spatial import Delaunay
from utils import DelaunayGraph, GabrielViaDelaunayVoronoi, EpsilonGraph, clean_graph
import networkx as nx


class GraphMetric(nn.Module):

    def __init__(self, X, z_dim, N, temp, loss_type=0):
        super().__init__()

        self.temp = temp
        self.loss_type = loss_type

        self.mean_x = X.mean(0)
        self.max_x = np.abs(X - X.mean(0)).max(0)[0] * 1.3
        self.x_dim = X.shape[1]
        self.z_dim = z_dim
        self.X = X
        self.N = N

        self.z_mult = N // X.shape[0]


        self.Z = nn.Parameter(torch.randn(N, z_dim).float(), requires_grad=True)
        # self.Z = nn.Parameter(self.norm_z(self.X).float(), requires_grad=True)
        # self.Z = nn.Parameter(self.norm_z(torch.ones_like(self.X)+torch.rand(N, z_dim)*0.01), requires_grad=True)

        # center_dist = torch.cdist(torch.tensor([[-0.5, -0.2]]), X)
        # center_i = torch.argmin(center_dist).item()
        # center = X[center_i]
        # registered_nodes = np.zeros(X.shape[0])
        # registered_nodes[center_i] = 1
        #
        # self.graph = nx.Graph()
        # tmp_node = center
        # tmp_i = center_i
        # while registered_nodes.prod() == 0:
        #
        #     all_d = torch.cdist(torch.unsqueeze(tmp_node, 0), X)
        #     sorted_idx = np.argsort(all_d[0].detach().cpu().numpy())
        #     for closest_available in sorted_idx:
        #         if (closest_available == np.nonzero(1 - registered_nodes)).any():
        #             break
        #     self.graph.add_edge(tmp_i, closest_available)
        #     tmp_node = X[closest_available]
        #     tmp_i = closest_available
        #     registered_nodes[closest_available] = 1

        # self.graph = GabrielViaDelaunayVoronoi(X)
        # eps = 0.1
        # self.graph = EpsilonGraph(X, eps)
        eps = 0.01
        one_connected_component = False
        while not one_connected_component:
            self.graph = EpsilonGraph(X, eps)
            # self.graph = clean_graph(tmp_graph)
            eps += 0.01
            if nx.number_connected_components(self.graph) == 1:
                one_connected_component = True
        print("Epsilon graph: ", str(eps-0.01))
        print()

        self.n_connected_components = nx.number_connected_components(self.graph)

        #
        # self.graph = clean_graph(tmp_graph)

        plt.scatter(X[:, 0], X[:, 1], color='tab:blue', alpha=0.05)
        for edge in self.graph.edges():
            plt.plot([X[edge[0], 0], X[edge[1], 0]], [X[edge[0], 1], X[edge[1], 1]], color='tab:orange', alpha=0.5)
        plt.show()

        self.max_degree = max([degree for n, degree in self.graph.degree()])

        self.lookup_table = {}
        for i in range(N):
            i_mul = i % self.z_mult
            i_x = int(i/self.z_mult)
            key = tuple(X[i_x].cpu().tolist() + [i_mul])
            self.lookup_table[hash(key)] = {
                "idx": i,
                "nei_idx": list(self.graph.adj[i_x].keys()),
            }

        self.decoder_epochs = 0

    def norm_x(self, x):
        return (x - self.mean_x) / self.max_x

    def norm_z(self, z):
        return F.normalize(z, p=2, dim=-1)

    def sample_z(self, n):
        return self.norm_z(torch.randn(n, self.Z.shape[-1]).float())

    def forward(self, x):
        return self.f(x)

    def get_prior_loss(self, x):

        B = x.shape[0]

        z = []
        z_nei = []
        for x_i in x:
            i_mu = np.random.randint(0, self.z_mult)
            key = tuple(x_i.cpu().tolist() + [i_mu])
            vals = self.lookup_table[hash(key)]
            i = vals['idx']
            neighbors = vals['nei_idx']

            z.append(self.Z[i])
            z_nei.append(self.Z[np.random.choice(neighbors)])

        z = torch.stack(z, 0).float().to(x.device)
        z_norm = self.norm_z(z)
        z_nei = torch.stack(z_nei, 0).float().to(x.device)
        z_norm_nei = self.norm_z(z_nei)

        # pos_d_z = torch.mean(torch.sum(z_norm * z_norm_nei, -1) / self.temp)
        # all_z = torch.cat([z_norm, z_norm_nei], 0)
        # neg_sim = torch.mm(all_z, all_z.t())
        # neg_dist = 2.0 - 2.0 * neg_sim.clamp(-1.0, 1.0)
        # B2 = all_z.size(0)
        # mask = ~torch.eye(B2, dtype=torch.bool, device=z.device)
        # neg_d_z = torch.log(torch.exp(-2.0 * neg_dist[mask]).mean())
        #
        # loss_contr = - pos_d_z + neg_d_z

        if self.loss_type == "cord":  # distance on the cord

            alpha = 2.0
            pos_d_z = ((z_norm - z_norm_nei).pow(alpha).sum(-1))
            aggr_pos_d_z = torch.mean(pos_d_z)

            neg_d_z = - self.temp * torch.pdist(z_norm, p=2).pow(2)
            aggr_neg_d_z = torch.log(torch.mean(torch.exp(neg_d_z)))

            # dot = z_norm @ z_norm.T
            # dot_squared = 2 - 2 * dot.clamp(-1, 1)
            # mask = ~torch.eye(dot_squared.size(0), dtype=torch.bool, device=z.device)
            # neg_d_z = - self.temp * dot_squared[mask]
            # aggr_neg_d_z = torch.log(torch.mean(torch.exp(neg_d_z)))

            lam = 1.0
            loss_contr = aggr_pos_d_z + lam * aggr_neg_d_z

            pos_term = aggr_pos_d_z
            neg_term = aggr_neg_d_z

        else:  # distance on the arc

            pos_d_z = torch.sum(z_norm * z_norm_nei, -1) / self.temp

            logits = torch.unsqueeze(pos_d_z, -1)
            neg_sim = torch.mm(z_norm, z_norm.t()) / self.temp
            logits = torch.cat([logits, neg_sim], 1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=z.device)

            loss_contr = F.cross_entropy(logits, labels)

            pos_term = pos_d_z.mean()
            neg_term = neg_sim.mean()

        return loss_contr, pos_term, neg_term

    def train_decoder(self, EPOCHS, writer):

        self.f = nn.Sequential(nn.Linear(self.z_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, self.x_dim)
                               )

        opt = torch.optim.Adam(self.f.parameters(), lr=3e-4)
        mse = nn.MSELoss()

        for _ in range(EPOCHS):
            batch_size = 64
            z_sampled = self.norm_z(torch.randn(batch_size, self.Z.shape[-1]).float())
            dist_dataset = torch.cdist(z_sampled, self.norm_z(self.Z))
            idx = torch.argmin(dist_dataset, -1)
            i_x = (idx / self.z_mult).int()
            x_closest = self.norm_x(self.X[i_x])
            x_hat = self.f(z_sampled.detach())
            loss = mse(x_closest, x_hat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("L_D", loss.detach().cpu().item(), self.decoder_epochs)
            self.decoder_epochs += 1

    def train_decoder_clean(self, EPOCHS, dataloader):

        self.f2 = nn.Sequential(nn.Linear(self.z_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, self.x_dim)
                                )

        opt = torch.optim.Adam(self.f2.parameters(), lr=3e-4)
        mse = nn.MSELoss()

        for _ in range(EPOCHS):
            for batch in dataloader:
                x = batch
                z = []
                for x_i in x:
                    i_mu = np.random.randint(0, self.z_mult)
                    key = tuple(x_i.cpu().tolist() + [i_mu])
                    vals = self.lookup_table[hash(key)]
                    i = vals['idx']
                    z.append(self.Z[i])

                z = torch.stack(z, 0).float().to(x.device)
                z_norm = self.norm_z(z)
                x_norm = self.norm_x(x)

                x_hat = self.f2(z_norm.detach())
                loss = mse(x_norm, x_hat)

                opt.zero_grad()
                loss.backward()
                opt.step()






















