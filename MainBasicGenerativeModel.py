import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from PIL import Image

import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


from datasets import Dataset_Toy
from BasicGenerativeModel import GraphMetric



EPOCHS = 10000000000000
frq_test = 1000
batch_size = 64

device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available() else 'cpu')

exp = 'moons' #'2d_swiss_roll'
n_samples = 1000
noise = 0.0
temp = 0.1

z_dim = 3
N = n_samples * 1
loss_type = "arc"  # "cord" "arc"

name_log = "exp="+exp+"_N="+str(N)+"_temp="+str(temp)+"_loss_type="+str(loss_type)+"_B="+str(batch_size) + "_long_z_dim=" + str(z_dim) + ""

writer = SummaryWriter("./logs5/"+name_log)

dataset = Dataset_Toy(exp, n_samples, noise)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

x_dim = dataset.X.shape[1]

x_mu = dataset.X.mean(0).to(device)
x_std = dataset.X.std(0).to(device)

model = GraphMetric(dataset.X, z_dim, N, temp).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)


logs_idx = 0
for epoch in tqdm(range(EPOCHS)):

    tot_L = 0
    tot_L_pos = 0
    tot_L_neg = 0
    for batch in dataloader:

        x = batch.to(device)

        loss, pos, neg = model.get_prior_loss(x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot_L += loss.detach().cpu().item()
        tot_L_pos += pos.detach().cpu().item()
        tot_L_neg += neg.detach().cpu().item()

    writer.add_scalar("L_contr", tot_L / len(dataloader), logs_idx)
    writer.add_scalar("L_pos", tot_L_pos / len(dataloader), logs_idx)
    writer.add_scalar("L_neg", tot_L_neg / len(dataloader), logs_idx)
    logs_idx += 1

    if epoch % frq_test == frq_test-1:

        writer.add_scalar("avg magnitude Z", torch.mean(torch.abs(model.Z)), logs_idx)

        z_train_norm = model.norm_z(model.Z)
        N = z_train_norm.shape[0]
        angles = torch.arccos(torch.mm(z_train_norm, z_train_norm.t()))
        i, j = torch.triu_indices(N, N, 1, device=z_train_norm.device)
        angles = angles[i, j]
        fig = plt.figure()
        plt.hist(angles.detach().cpu().numpy(), 1000)
        plt.ylim(0, N)
        writer.add_figure("pair-wise distances", fig, epoch)
        plt.close(fig)

        if model.n_connected_components == 1:

            center_dist = torch.cdist(torch.tensor([[-0.25, -0.25]]), model.norm_x(model.X))
            center_i = torch.argmin(center_dist).item()
            center = model.norm_x(model.X[center_i]).detach().cpu().numpy()
            all_d = []
            for i in range(N):
                i_x = int(i / model.z_mult)
                d = nx.shortest_path_length(model.graph, center_i, i_x)
                all_d.append(d)
            all_d = np.array(all_d)
            inv_d = np.max(all_d) - all_d

            max_d = np.max(all_d)
            unif_x = []
            unif_i = []
            for d in np.linspace(0, max_d, 10):
                i = np.argmin(np.abs(all_d - d))
                unif_i.append(i)
                i_x = int(i / model.z_mult)
                unif_x.append(model.norm_x(model.X[i_x]))
            unif_x = np.stack(unif_x, 0)

        x_train_norm = model.norm_x(model.X).detach().cpu().numpy()
        z_train_norm = model.norm_z(model.Z).detach().cpu().numpy()
        model.train_decoder(10000, writer)
        model.train_decoder_clean(1000, dataloader)
        x_train_hat = model(model.norm_z(model.Z)).detach().cpu().numpy()
        z_sampled = model.sample_z(1000).float().to(device)
        x_sampled = model(z_sampled).detach().cpu().numpy()
        z_sampled2 = model.norm_z(torch.randn(1000, model.Z.shape[-1]).float()).float().to(device)
        x_sampled2 = model.f2(z_sampled2).detach().cpu().numpy()

        if z_dim == 2:

            if model.n_connected_components == 1:
                fig = plt.figure()
                plt.scatter(z_train_norm[:, 0], z_train_norm[:, 1], c=inv_d, alpha=0.5)
                plt.scatter(x_train_norm[:, 0], x_train_norm[:, 1], alpha=0.2)
                for i, x_u_i in zip(unif_i, unif_x):
                    plt.scatter(x_u_i[0], x_u_i[1], color='tab:orange')
                    plt.scatter(z_train_norm[i, 0], z_train_norm[i, 1], s=80, facecolors='none', edgecolors='tab:red')
                    xx = [x_u_i[0], z_train_norm[i, 0]]
                    yy = [x_u_i[1], z_train_norm[i, 1]]
                    plt.plot(xx, yy, color='black', alpha=0.2)
                writer.add_figure("field", fig, epoch)
                plt.close(fig)

            fig = plt.figure()
            plt.scatter(z_train_norm[:, 0], z_train_norm[:, 1])
            writer.add_figure("train prior", fig, epoch)
            plt.close(fig)

        elif z_dim == 3:

            if model.n_connected_components == 1:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(z_train_norm[:, 0], z_train_norm[:, 1], z_train_norm[:, 2], c=inv_d, alpha=0.5)
                ax.scatter(x_train_norm[:, 0], x_train_norm[:, 1], x_train_norm[:, 1] * 0, alpha=0.2)
                for i, x_u_i in zip(unif_i, unif_x):
                    ax.scatter(x_u_i[0], x_u_i[1], 0, color='tab:orange')
                    ax.scatter(z_train_norm[i, 0], z_train_norm[i, 1], z_train_norm[i, 2], s=80, facecolors='none', edgecolors='tab:red')
                    ax.plot([x_u_i[0], z_train_norm[i, 0]],
                            [x_u_i[1], z_train_norm[i, 1]],
                            [0, z_train_norm[i, 2]],
                            color="black", alpha=0.2)
                writer.add_figure("field", fig, epoch)
                plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(z_train_norm[:, 0], z_train_norm[:, 1], z_train_norm[:, 2])
            writer.add_figure("train prior", fig, epoch)
            plt.close(fig)

        fig = plt.figure()
        plt.scatter(x_train_hat[:, 0], x_train_hat[:, 1], alpha=0.1, zorder=2)
        plt.scatter(x_train_norm[:, 0], x_train_norm[:, 1], alpha=0.1, zorder=1)
        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        writer.add_figure("train posterior", fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(x_sampled[:, 0], x_sampled[:, 1], alpha=0.1, zorder=2)
        plt.scatter(x_train_norm[:, 0], x_train_norm[:, 1], alpha=0.1, zorder=1)
        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        writer.add_figure("test posterior", fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(x_sampled2[:, 0], x_sampled2[:, 1], alpha=0.1, zorder=2)
        plt.scatter(x_train_norm[:, 0], x_train_norm[:, 1], alpha=0.1, zorder=1)
        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        writer.add_figure("test posterior clean", fig, epoch)
        plt.close(fig)

        if model.n_connected_components == 1:
            fig, axes = plt.subplots(
                nrows=1, ncols=3, figsize=(12, 4),  # width = 3 × height looks good
                constrained_layout=True  # auto-fit the panels
            )
            for ax, i in zip(axes, [center_i, 0, 1]):
                center_dist_Z = torch.cdist(model.norm_x(model.X)[i:i + 1], model.norm_x(model.X))
                inv_d_z = (-center_dist_Z).squeeze(0).cpu().numpy()
                ax.scatter(x_train_norm[:, 0], x_train_norm[:, 1], c=inv_d_z)
                ax.scatter(x_train_norm[i, 0], x_train_norm[i, 1], s=80, facecolors='none', edgecolors='tab:red')
            writer.add_figure("dist in Z from i", fig, epoch)
            plt.close(fig)

    if epoch % 10000 == 9999:

        x_train_norm = model.norm_x(model.X).detach().cpu().numpy()
        model.train_decoder(100000, writer)
        z_sampled = model.sample_z(10000).float().to(device)
        x_sampled = model(z_sampled).detach().cpu().numpy()

        fig = plt.figure()
        plt.scatter(x_sampled[:, 0], x_sampled[:, 1], alpha=0.1, zorder=2)
        plt.scatter(x_train_norm[:, 0], x_train_norm[:, 1], alpha=0.1, zorder=1)
        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        writer.add_figure("test posterior long", fig, epoch)
        plt.close(fig)


writer.close()


















