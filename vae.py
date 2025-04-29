import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import Dataset_Toy


class VAE(nn.Module):

    def __init__(self, x_dim, z_dim, device):
        super().__init__()

        self.z_dim = z_dim
        self.device = device

        self.f = nn.Sequential(nn.Linear(x_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64)
                               )

        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

        self.g = nn.Sequential(nn.Linear(z_dim, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, 64),
                               nn.ReLU(),
                               nn.Linear(64, x_dim)
                               )

    def encode(self, x):
        h = self.f(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.g(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample_x(self, n):

        z = torch.randn(n, self.z_dim).float().to(self.device)
        return self.decode(z)

    def get_loss(self, x):

        x_hat, mu, logvar = self(x)

        MSE = torch.mean(torch.sum((x_hat - x)**2, -1))
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1))

        return MSE, KLD


if __name__ == "__main__":

    EPOCHS = 100000
    frq_test = 100
    batch_size = 64  # 1000

    device = torch.device('cpu')  # 'cuda:0' if torch.cuda.is_available() else 'cpu')

    exp = '2d_swiss_roll'
    n_samples = 1000
    noise = 0.0

    writer = SummaryWriter("./logs3/exp=" + exp + "_vae9")

    dataset = Dataset_Toy(exp, n_samples, noise)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_dim = dataset.X.shape[1]
    z_dim = 3

    model = VAE(x_dim, z_dim, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    logs_idx = 0
    for epoch in tqdm(range(2 * EPOCHS)):

        tot_mse = 0
        tot_kl = 0

        for batch in dataloader:
            x = batch.to(device)

            MSE, KLD = model.get_loss(x)

            beta = 0.01
            loss = MSE + beta * KLD

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_mse += MSE.detach().cpu().item()
            tot_kl += KLD.detach().cpu().item()

        writer.add_scalar("mse", tot_mse / len(dataloader), logs_idx)
        writer.add_scalar("kl", tot_kl / len(dataloader), logs_idx)
        logs_idx += 1

        if epoch % frq_test == frq_test - 1:

            # z_sampled = model.sample_z(10000).float().to(device)
            x_sampled = model.sample_x(10000).detach().cpu().numpy()
            x_train = dataset.X
            # z_sampled = z_sampled.detach().cpu().numpy()

            fig = plt.figure()
            plt.scatter(x_sampled[:, 0], x_sampled[:, 1], alpha=0.01)
            plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.01)
            writer.add_figure("dist_sampled", fig, epoch)
            plt.close(fig)








