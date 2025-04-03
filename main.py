import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from PIL import Image

from torchvision import datasets as torch_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from datasets import Dataset_Toy
from model import regSet
from metricModel import CircularMetric

EPOCHS = 100000
frq_test = 1000
batch_size = 1000

device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available() else 'cpu')

exp = '2d_swiss_roll'
n_samples = 1000
noise = 0.0

writer = SummaryWriter("./logs/exp="+exp+"_metric_detach_mask_warmup_annealing")

dataset = Dataset_Toy(exp, n_samples, noise)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

x_dim = dataset.X.shape[1]

x_mu = dataset.X.mean(0).to(device)
x_std = dataset.X.std(0).to(device)

# model = regSet(x_dim, x_mu, x_std).to(device)
# opt = torch.optim.Adam(model.parameters(), lr=3e-4)

model_metric = CircularMetric(dataset.X, x_dim).to(device)
opt = torch.optim.Adam(model_metric.parameters(), lr=3e-4)

# model.generate_noise(dataset.X.to(device))


logs_idx = 0
for epoch in tqdm(range(2*EPOCHS)):

    tot_L_c = 0
    tot_L_s = 0
    tot_L_d = 0
    for batch in dataloader:

        x = batch.to(device)

        L_cost, L_spread, L_dec = model_metric.get_loss(x)

        if epoch < EPOCHS:
            # w_c = 3 * (0.1/3)**((epoch/EPOCHS)**0.5)
            # w_c = 10 + (0.1 - 10)*(epoch/EPOCHS)
            w_c = 10.0 if epoch < 5000 else 0.01
            w_s = 1.0
            w_d = 0
        else:
            w_c = 0
            w_s = 0
            w_d = 1

        loss = w_c*L_cost + w_s*L_spread + w_d*L_dec

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot_L_c += L_cost.detach().cpu().item()
        tot_L_s += L_spread.detach().cpu().item()
        tot_L_d += L_dec.detach().cpu().item()

    writer.add_scalar("L_cost", tot_L_c / len(dataloader), logs_idx)
    writer.add_scalar("L_spread", tot_L_s / len(dataloader), logs_idx)
    writer.add_scalar("L_dec", tot_L_d / len(dataloader), logs_idx)
    logs_idx += 1

    if epoch % frq_test == frq_test-1:

        z = torch.randn(1000, x_dim).float().to(device)

        # z = torch.rand(1000, x_dim).float().to(device) * 2 - 1
        z = model_metric.sample_z(1000).float().to(device)
        x_hat = model_metric(z).detach().cpu().numpy()

        fig = plt.figure()
        plt.scatter(dataset.X[:,0].detach().cpu().numpy(), dataset.X[:,1].detach().cpu().numpy(), alpha=1.0)
        plt.scatter(x_hat[:,0], x_hat[:,1], alpha=0.2)
        writer.add_figure("dist_sampled", fig, epoch)
        plt.close(fig)

        x_hat = model_metric(model_metric.norm_z(model_metric.Z)).detach().cpu().numpy()

        fig = plt.figure()
        plt.scatter(dataset.X[:,0].detach().cpu().numpy(), dataset.X[:,1].detach().cpu().numpy(), alpha=0.1)
        plt.scatter(x_hat[:,0], x_hat[:,1], alpha=1.0)
        writer.add_figure("dist_train", fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        all_z = torch.tanh(model_metric.Z).detach().cpu().numpy()
        plt.scatter(all_z[:, 0], all_z[:, 1])
        writer.add_figure("Z dist", fig, epoch)
        plt.close(fig)


writer.close()

















