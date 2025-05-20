# %%
import json
import torch
import pylab as pl
import numpy as np
from torch_geometric.data import Batch
from ulot.utils_data import SBM
from ulot.utils import (
    get_filename,
    get_model,
    get_loss,
)

from tqdm import tqdm

import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ExponentialLR
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_file = "parameter_files/params_SBM_50000.json"

with open(param_file, "r") as file:
    parameters = json.load(file)

model = get_model(parameters, 3)
model = model.to(device)
path_trained_model = get_filename(parameters, "trained_model")
checkpoint = torch.load(
    path_trained_model, weights_only=False, map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# %% Optimize the parameters rho and alpha on the label propagation task for clusters (1,2,3) -> (1,2)
###################################################################################################

torch.manual_seed(42)
np.random.seed(42)
random.seed(41)

loss = get_loss(parameters)

alphas = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32, device=device))
rhos = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32, device=device))
alphas.requires_grad_()
rhos.requires_grad_()

n_graphs = 10
dataset_2 = SBM(
    n_graphs=n_graphs,
    nc1_i=3,
    nc2_i=2,
    clusters1=[1, 2, 3],
    clusters2=[1, 2],
    random_graph=False,
    random_n_nodes=True,
    noise=0.3,
    p_intra=0.5,
    p_inter=0.05,
)

n_steps = 2000
dt_rho = 0.0001
dt_alpha = 0.001

eps = 10**-6

accuracy_2 = torch.zeros(n_graphs, n_steps)
loss_2 = torch.zeros(n_graphs, n_steps)
alphas_2 = torch.zeros(n_graphs, n_steps)
rhos_2 = torch.zeros(n_graphs, n_steps)

rhos_init_all = torch.exp(-torch.rand(n_graphs) * 3)
# rhos_init_all=rhos_init_all[torch.randperm(rhos_init_all.size(0))]
all_validation_nodes = []
lens = []
for k in tqdm(range(n_graphs)):
    alpha_init = torch.tensor([0.5])
    rho_init = torch.tensor([0.1])
    alphas = torch.nn.Parameter(torch.tensor([[alpha_init]]))
    rhos = torch.nn.Parameter(torch.tensor([[rho_init]]))
    alphas.requires_grad_()
    rhos.requires_grad_()
    optimizer_rho = torch.optim.Adam([rhos], lr=dt_rho)
    optimizer_alpha = torch.optim.Adam([alphas], lr=dt_alpha)
    scheduler_rho = ExponentialLR(optimizer_rho, gamma=1)
    scheduler_alpha = ExponentialLR(optimizer_alpha, gamma=1)

    graph_pair = dataset_2.__getitem__(k)
    graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
    n_nodes_target = graph_pair_batch.x_t.shape[0]
    validation_nodes = np.random.randint(0, n_nodes_target, int(0.5 * n_nodes_target))
    all_validation_nodes.append(validation_nodes)
    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch

    for n in range(n_steps):
        rhos_2[k, n] = rhos[0, 0].detach()
        alphas_2[k, n] = alphas[0, 0].detach()
        if (
            n > 1
            and torch.abs(rhos_2[k, n] - rhos_2[k, n - 1]) < eps
            and torch.abs(alphas_2[k, n] - alphas_2[k, n - 1]) < eps
        ):
            break

        batched_rhos_s = rhos[batch_indices_s].view(-1, 1)
        batched_rhos_t = rhos[batch_indices_t].view(-1, 1)
        batched_alphas_s = alphas[batch_indices_s].view(-1, 1)
        batched_alphas_t = alphas[batch_indices_t].view(-1, 1)
        P, mask1, mask2 = model(
            graph_pair_batch.x_s,
            graph_pair_batch.x_t,
            graph_pair_batch.edge_index_s,
            graph_pair_batch.edge_index_t,
            graph_pair_batch.x_s_batch,
            graph_pair_batch.x_t_batch,
            batched_rhos_s,
            batched_rhos_t,
            batched_alphas_s,
            batched_alphas_t,
        )

        col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
        safe_col_sums = torch.clamp(col_sums, min=1e-15)

        if torch.sum(P[0] < 10**-15):
            print("null")

        labels_s = graph_pair.y_s
        labels_t = graph_pair.y_t

        one_hot_matrix_s = F.one_hot(labels_s, num_classes=3).float()
        one_hot_matrix_t = F.one_hot(labels_t, num_classes=3).float()

        weight = torch.diag(1 / safe_col_sums)
        weighted_P = torch.mm(P[0], weight).T
        labels_probability = torch.mm(weighted_P, one_hot_matrix_s)

        log_probs = torch.log(
            torch.softmax(torch.clamp(labels_probability, min=1e-10), dim=1)
        )
        predicted_labels = torch.max(log_probs, axis=1).indices
        bool_accuracy = predicted_labels[validation_nodes] == labels_t[validation_nodes]
        # loss = criterion(logits, labels_t)
        loss = F.kl_div(
            log_probs[validation_nodes],
            one_hot_matrix_t[validation_nodes],
            reduction="batchmean",
        )

        accuracy = torch.sum(bool_accuracy) / len(validation_nodes)
        accuracy = accuracy.detach()
        accuracy_2[k, n] = accuracy.detach()
        loss_2[k, n] = loss.detach()
        optimizer_alpha.zero_grad()
        optimizer_rho.zero_grad()
        loss.backward()
        optimizer_alpha.step()
        optimizer_rho.step()
        scheduler_rho.step()
        scheduler_alpha.step()
        with torch.no_grad():
            rhos.clamp_(min=0.003, max=1)
            alphas.clamp_(min=0, max=1)
    lens.append(n)

# %% Optimize the parameters rho and alpha on the label propagation task for clusters (1,2,3) -> (1,2,3)
###################################################################################################

torch.manual_seed(42)
np.random.seed(42)
random.seed(41)

loss = get_loss(parameters)

alphas = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32, device=device))
rhos = torch.nn.Parameter(torch.tensor([[0.1]], dtype=torch.float32, device=device))
alphas.requires_grad_()
rhos.requires_grad_()

n_graphs = 10
dataset_3 = SBM(
    n_graphs=n_graphs,
    nc1_i=3,
    nc2_i=3,
    clusters1=[1, 2, 3],
    clusters2=[1, 2, 3],
    random_graph=False,
    random_n_nodes=True,
    noise=0.5,
    p_intra=0.5,
    p_inter=0.05,
)

n_steps = 2000
dt_rho = 0.001
dt_alpha = 0.0005
optimizer_rho = torch.optim.Adam([rhos], lr=dt_rho)
optimizer_alpha = torch.optim.Adam([alphas], lr=dt_alpha)
scheduler_rho = ExponentialLR(optimizer_rho, gamma=1)
scheduler_alpha = ExponentialLR(optimizer_alpha, gamma=1)

accuracy_3 = torch.zeros(n_graphs, n_steps)
loss_3 = torch.zeros(n_graphs, n_steps)
alphas_3 = torch.zeros(n_graphs, n_steps)
rhos_3 = torch.zeros(n_graphs, n_steps)

rhos_init_all = torch.exp(-torch.rand(n_graphs) * 3)
# rhos_init_all=rhos_init_all[torch.randperm(rhos_init_all.size(0))]
all_validation_nodes_3 = []

eps = 10**-6
lens_3 = []

for k in tqdm(range(n_graphs)):
    alpha_init = torch.tensor([0.5])
    rho_init = torch.tensor([0.1])
    alphas = torch.nn.Parameter(torch.tensor([[alpha_init]]))
    rhos = torch.nn.Parameter(torch.tensor([[rho_init]]))
    alphas.requires_grad_()
    rhos.requires_grad_()
    optimizer_rho = torch.optim.Adam([rhos], lr=dt_rho)
    optimizer_alpha = torch.optim.Adam([alphas], lr=dt_alpha)
    scheduler_rho = ExponentialLR(optimizer_rho, gamma=1)
    scheduler_alpha = ExponentialLR(optimizer_alpha, gamma=1)

    graph_pair = dataset_3.__getitem__(k)
    graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
    n_nodes_target = graph_pair_batch.x_t.shape[0]
    validation_nodes = np.random.randint(0, n_nodes_target, int(0.5 * n_nodes_target))
    all_validation_nodes_3.append(validation_nodes)

    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch

    for n in range(n_steps):
        if (
            n > 1
            and torch.abs(rhos_3[k, n] - rhos_3[k, n - 1]) < eps
            and torch.abs(alphas_3[k, n] - alphas_3[k, n - 1]) < eps
        ):
            break

        batched_rhos_s = rhos[batch_indices_s].view(-1, 1)
        batched_rhos_t = rhos[batch_indices_t].view(-1, 1)
        batched_alphas_s = alphas[batch_indices_s].view(-1, 1)
        batched_alphas_t = alphas[batch_indices_t].view(-1, 1)
        P, mask1, mask2 = model(
            graph_pair_batch.x_s,
            graph_pair_batch.x_t,
            graph_pair_batch.edge_index_s,
            graph_pair_batch.edge_index_t,
            graph_pair_batch.x_s_batch,
            graph_pair_batch.x_t_batch,
            batched_rhos_s,
            batched_rhos_t,
            batched_alphas_s,
            batched_alphas_t,
        )

        col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
        safe_col_sums = torch.clamp(col_sums, min=1e-15)

        if torch.sum(P[0] < 10**-15):
            print("null")

        labels_s = graph_pair.y_s
        labels_t = graph_pair.y_t[validation_nodes]

        one_hot_matrix_s = F.one_hot(labels_s, num_classes=3).float()
        one_hot_matrix_t = F.one_hot(labels_t, num_classes=3).float()

        weight = torch.diag(1 / safe_col_sums)
        weighted_P = torch.mm(P[0], weight).T
        labels_probability = torch.mm(weighted_P, one_hot_matrix_s)[validation_nodes]

        probs = F.softmax(labels_probability, dim=1)
        log_probs = torch.log(torch.clamp(probs, min=1e-10))
        predicted_labels = torch.max(log_probs, axis=1).indices
        bool_accuracy = predicted_labels == labels_t
        # loss = criterion(logits, labels_t)
        loss = F.kl_div(log_probs, one_hot_matrix_t, reduction="batchmean")
        accuracy = torch.sum(bool_accuracy) / len(validation_nodes)
        accuracy_3[k, n] = accuracy.detach()
        loss_3[k, n] = loss.detach()
        optimizer_alpha.zero_grad()
        optimizer_rho.zero_grad()
        loss.backward()
        optimizer_alpha.step()
        optimizer_rho.step()
        scheduler_rho.step()
        scheduler_alpha.step()
        with torch.no_grad():
            rhos.clamp_(min=0.003, max=1)
            alphas.clamp_(min=0, max=1)
        rhos_3[k, n] = rhos[0, 0].detach()
        alphas_3[k, n] = alphas[0, 0].detach()
    lens_3.append(n)


# %% Visualize the alpha and rho trajectories

def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


fig, ax = pl.subplots(figsize=(2.5, 2.5))


for i in range(n_graphs):
    len_i = lens[i]
    t = np.arange(len_i)
    x = rhos_2[i, :len_i]
    y = alphas_2[i, :len_i]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=t.min(), vmax=t.max())
    cmap = truncate_colormap(cm.Reds, 0.2, 1.0)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t)  # Set the time values for colormap mapping
    lc.set_linewidth(2)
    lc.set_alpha(0.6)

    ax.add_collection(lc)
    ax.autoscale()
    if i == 0:
        ax.plot(
            x[0],
            y[0],
            marker="o",
            markersize=6,
            color="green",
            label="Initialization",
            linestyle="None",
            zorder=10,
        )
        ax.plot(
            x[-1],
            y[-1],
            marker="o",
            markersize=2,
            color="black",
            label="Convergence",
            linestyle="None",
            zorder=10,
        )
    else:
        ax.plot(x[-1], y[-1], marker="o", markersize=2, color="black", zorder=10)

ax.set_title(
    "Optim. $(\\alpha,\\rho)$ with respect \n to KL for $(1,2,3) \\rightarrow (1,2)$ "
)
ax.set_ylim(0, 1)
ax.set_xlim(0.003, 1)
ax.set_xscale("log")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\alpha$")
pl.legend()
pl.grid()

fig, ax = pl.subplots(figsize=(2.5, 2.5))


for i in range(n_graphs):
    len_i = lens_3[i]
    t = np.arange(len_i)
    x = rhos_3[i, :len_i]
    y = alphas_3[i, :len_i]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=t.min(), vmax=t.max())
    cmap = truncate_colormap(cm.Blues, 0.2, 1.0)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t)  # Set the time values for colormap mapping
    lc.set_linewidth(2)
    lc.set_alpha(0.6)

    ax.add_collection(lc)
    ax.autoscale()
    if i == 0:
        ax.plot(
            x[0],
            y[0],
            marker="o",
            markersize=6,
            color="green",
            label="Initialization",
            linestyle="None",
            zorder=10,
        )
        ax.plot(
            x[-1],
            y[-1],
            marker="o",
            markersize=2,
            color="black",
            label="Convergence",
            linestyle="None",
            zorder=10,
        )
    else:
        ax.plot(x[-1], y[-1], marker="o", markersize=2, color="black", zorder=10)

ax.set_title(
    "Optim. $(\\alpha,\\rho)$ with respect \n to KL for $(1,2,3) \\rightarrow (1,2,3)$ "
)
ax.set_ylim(0, 1)
ax.set_xlim(0.003, 1)
ax.set_xscale("log")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$\alpha$")
pl.legend()
pl.grid()


# %% test for clusters (1,2,3) -> (1,2,3)
############################################

torch.manual_seed(40)
np.random.seed(6)
random.seed(42)


accuracy_test_3 = torch.zeros(n_graphs)

for k in tqdm(range(n_graphs)):
    graph_pair = dataset_3.__getitem__(k)
    graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch
    n_nodes_target = graph_pair_batch.x_t.shape[0]
    rhos = rhos_3[k, -1].reshape(-1, 1)
    alphas = alphas_3[k, -1].reshape(-1, 1)
    batched_rhos_s = rhos[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
    batched_rhos_t = rhos[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
    batched_alphas_s = alphas[batch_indices_s].reshape(len(alphas[batch_indices_s]), 1)
    batched_alphas_t = alphas[batch_indices_t].reshape(len(alphas[batch_indices_t]), 1)
    validation_nodes = all_validation_nodes_3[k]
    testing_nodes = [i for i in range(n_nodes_target) if i not in validation_nodes]
    P, mask1, mask2 = model(
        graph_pair_batch.x_s,
        graph_pair_batch.x_t,
        graph_pair_batch.edge_index_s,
        graph_pair_batch.edge_index_t,
        graph_pair_batch.x_s_batch,
        graph_pair_batch.x_t_batch,
        batched_rhos_s,
        batched_rhos_t,
        batched_alphas_s,
        batched_alphas_t,
    )
    col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
    safe_col_sums = torch.clamp(col_sums, min=1e-15)

    if torch.sum(P[0] < 10**-15):
        print("null")

    labels_s = graph_pair.y_s
    labels_t = graph_pair.y_t[testing_nodes]

    one_hot_matrix_s = F.one_hot(labels_s, num_classes=3).float()
    one_hot_matrix_t = F.one_hot(labels_t, num_classes=3).float()

    weight = torch.diag(1 / safe_col_sums)
    weighted_P = torch.mm(P[0], weight).T
    labels_probability = torch.mm(weighted_P, one_hot_matrix_s)[testing_nodes]
    probs = F.softmax(labels_probability, dim=1)
    log_probs = torch.log(torch.clamp(probs, min=1e-10))
    predicted_labels = torch.max(log_probs, axis=1).indices
    bool_accuracy = predicted_labels == labels_t
    accuracy = torch.sum(bool_accuracy) / len(testing_nodes)
    accuracy_test_3[k] = accuracy.detach()
print(torch.mean(accuracy_test_3))
print(torch.std(accuracy_test_3))

# %% test for clusters (1,2,3) -> (1,2)
############################################

torch.manual_seed(40)
np.random.seed(6)
random.seed(42)


accuracy_test_2 = torch.zeros(n_graphs)

for k in tqdm(range(n_graphs)):
    graph_pair = dataset_2.__getitem__(k)
    graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch
    n_nodes_target = graph_pair_batch.x_t.shape[0]
    rhos = rhos_3[k, -1].reshape(-1, 1)
    alphas = alphas_3[k, -1].reshape(-1, 1)
    batched_rhos_s = rhos[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
    batched_rhos_t = rhos[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
    batched_alphas_s = alphas[batch_indices_s].reshape(len(alphas[batch_indices_s]), 1)
    batched_alphas_t = alphas[batch_indices_t].reshape(len(alphas[batch_indices_t]), 1)
    validation_nodes = all_validation_nodes_3[k]
    testing_nodes = [i for i in range(n_nodes_target) if i not in validation_nodes]
    P, mask1, mask2 = model(
        graph_pair_batch.x_s,
        graph_pair_batch.x_t,
        graph_pair_batch.edge_index_s,
        graph_pair_batch.edge_index_t,
        graph_pair_batch.x_s_batch,
        graph_pair_batch.x_t_batch,
        batched_rhos_s,
        batched_rhos_t,
        batched_alphas_s,
        batched_alphas_t,
    )
    col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
    safe_col_sums = torch.clamp(col_sums, min=1e-15)

    if torch.sum(P[0] < 10**-15):
        print("null")

    labels_s = graph_pair.y_s
    labels_t = graph_pair.y_t[testing_nodes]

    one_hot_matrix_s = F.one_hot(labels_s, num_classes=3).float()
    one_hot_matrix_t = F.one_hot(labels_t, num_classes=3).float()

    weight = torch.diag(1 / safe_col_sums)
    weighted_P = torch.mm(P[0], weight).T
    labels_probability = torch.mm(weighted_P, one_hot_matrix_s)[testing_nodes]
    probs = F.softmax(labels_probability, dim=1)
    log_probs = torch.log(torch.clamp(probs, min=1e-10))
    predicted_labels = torch.max(log_probs, axis=1).indices
    bool_accuracy = predicted_labels == labels_t
    accuracy = torch.sum(bool_accuracy) / len(testing_nodes)
    accuracy_test_2[k] = accuracy.detach()
print(torch.mean(accuracy_test_2))
print(torch.std(accuracy_test_2))
