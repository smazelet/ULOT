# %%
import json
import torch
from ulot.fugw import FUGWSolver
import pylab as pl
import numpy as np
from torch_geometric.data import Batch
from ulot.utils import get_filename
from ulot.utils_data import SBM
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ulot.utils import (
    get_model,
    get_loss,
)
import random


# %% function for generating SBMs with two clusters

def create_sbm_2_clusters(
    n_graphs=1,
    nc1_i=2,
    nc2_i=2,
    clusters1=[1, 2],
    clusters2=[2, 3],
    random_graph=False,
    random_n_nodes=False,
    noise=0.6,
    p_intra=0.6,
    p_inter=0.03,
    n_nodes1=30,
    n_nodes2=30,
):
    dataset = SBM(
        n_graphs=n_graphs,
        nc1_i=nc1_i,
        nc2_i=nc2_i,
        clusters1=clusters1,
        clusters2=clusters2,
        random_graph=random_graph,
        random_n_nodes=random_n_nodes,
        noise=noise,
        p_intra=p_intra,
        p_inter=p_inter,
        n_nodes1=n_nodes1,
        n_nodes2=n_nodes2,
    )
    graph_pair = dataset.__getitem__(0)
    M = torch.cdist(graph_pair.x_s, graph_pair.x_t) ** 2
    M = M / (10 * torch.mean(M))

    n1 = graph_pair.x_s.shape[0]
    n2 = graph_pair.x_t.shape[0]

    G1 = Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s, y=graph_pair.y_s)
    G2 = Data(x=graph_pair.x_t, edge_index=graph_pair.edge_index_t, y=graph_pair.y_t)

    G1_nx = to_networkx(G1, to_undirected=True)
    G2_nx = to_networkx(G2, to_undirected=True)

    shortest_path_matrix1 = dict(nx.all_pairs_shortest_path_length(G1_nx))
    shortest_path_matrix2 = dict(nx.all_pairs_shortest_path_length(G2_nx))

    D1 = np.full((n1, n1), np.inf)  # Initialize with infinity
    for i, paths in shortest_path_matrix1.items():
        for j, length in paths.items():
            D1[i, j] = length
    D1 = torch.tensor(D1)

    D2 = np.full((n2, n2), np.inf)  # Initialize with infinity
    for i, paths in shortest_path_matrix2.items():
        for j, length in paths.items():
            D2[i, j] = length
    D2 = torch.tensor(D2)

    D1 = D1 / (10 * torch.mean(D1))
    D2 = D2 / (10 * torch.mean(D2))
    max = np.max([torch.max(graph_pair.x_s), torch.max(graph_pair.x_t)])
    min = np.min([torch.min(graph_pair.x_s), torch.min(graph_pair.x_t)])

    rgb1 = (G1.x - min) / (max - min)
    rgb2 = (G2.x - min) / (max - min)

    nc1_1 = torch.sum(G1.y == 0)
    nc2_1 = torch.sum(G1.y == 1)
    nc2_2 = torch.sum(G2.y == 1)
    nc3_2 = torch.sum(G2.y == 2)

    pos1_center = [0.0, 0.0]
    pos1 = np.random.multivariate_normal(
        pos1_center, [[0.1, 0], [0, 0.1]], size=(nc1_1.item())
    )

    pos2_center = [2.0, 0.0]
    pos2 = np.random.multivariate_normal(
        pos2_center, [[0.1, 0], [0, 0.1]], size=(nc2_1.item())
    )

    pos_g1 = np.stack([pos1, pos2]).reshape(nc1_1 + nc2_1, 2)
    pos_g1 = {node: (x, y) for node, (x, y) in enumerate(pos_g1)}

    pos2_center = [2.0, 0.0]
    pos2 = np.random.multivariate_normal(
        pos2_center, [[0.1, 0], [0, 0.1]], size=(nc2_2.item())
    )

    pos3_center = [4.0, 0.0]
    pos3 = np.random.multivariate_normal(
        pos3_center, [[0.1, 0], [0, 0.1]], size=(nc3_2.item())
    )

    pos_g2 = np.stack([pos2, pos3]).reshape(nc2_2 + nc3_2, 2)

    offset = 1.8

    pos_g2 = {node: (x, y - offset) for node, (x, y) in enumerate(pos_g2)}

    return dataset, pos_g1, pos_g2, G1_nx, G2_nx, D1, D2, M, rgb1, rgb2


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
# %% Plans for increasing rho

torch.manual_seed(42)
np.random.seed(42)
random.seed(41)


dataset_2, pos_g1, pos_g2, G1_nx, G2_nx, D1, D2, M, rgb1, rgb2 = create_sbm_2_clusters(
    n_graphs=1,
    nc1_i=2,
    nc2_i=2,
    clusters1=[1, 2],
    clusters2=[2, 3],
    random_graph=False,
    noise=0.3,
    p_intra=0.5,
    p_inter=0.05,
    n_nodes1=20,
    n_nodes2=20,
)
graph_pair = dataset_2.__getitem__(0)
graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])

torch.save(pos_g1, "results/figure_files/pos_g1_rho")
torch.save(pos_g2, "results/figure_files/pos_g2_rho")
torch.save(G1_nx, "results/figure_files/G1_nx_rho")
torch.save(rgb1, "results/figure_files/rgb1_rho")
torch.save(rgb2, "results/figure_files/rgb2_rho")
torch.save(G2_nx, "results/figure_files/G2_nx_rho")
torch.save(graph_pair, "results/figure_files/graph_pair_rho")

all_P_solver = []
all_P_ulot = []
rhos = np.exp(-np.linspace(5, 7, 4))
alpha = torch.tensor([[0.8]])
n = 4
num = 0
loss = get_loss(parameters)
max = np.max([torch.max(graph_pair.x_s), torch.max(graph_pair.x_t)])
min = np.min([torch.min(graph_pair.x_s), torch.min(graph_pair.x_t)])
for i, rho in enumerate(rhos[::-1]):
    # if i==0 or i==len(rhos)//2 or i==len(rhos)-1:
    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch
    rho = torch.tensor(rho, dtype=torch.float32).reshape(1, 1)
    alpha = torch.tensor(alpha, dtype=torch.float32).reshape(1, 1)

    batched_rhos_s = rho[batch_indices_s].reshape(len(rho[batch_indices_s]), 1)
    batched_rhos_t = rho[batch_indices_t].reshape(len(rho[batch_indices_t]), 1)
    batched_alphas_s = alpha[batch_indices_s].reshape(len(alpha[batch_indices_s]), 1)
    batched_alphas_t = alpha[batch_indices_t].reshape(len(alpha[batch_indices_t]), 1)

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
    L = loss(
        P=P,
        x1=graph_pair_batch.x_s,
        x2=graph_pair_batch.x_t,
        batch1=graph_pair_batch.x_s_batch,
        batch2=graph_pair_batch.x_t_batch,
        mask1=mask1,
        mask2=mask2,
        rhos=rho[0],
        alphas=alpha[0],
        connectivity_s=graph_pair_batch.connectivity_s,
        connectivity_t=graph_pair_batch.connectivity_t,
    )

    all_P_ulot.append(P)

    rho = rho[0]
    alpha = alpha[0]
    solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)
    solution = solver.solve(
        alpha,
        rho,
        rho,
        0.0,
        "joint",
        "kl",
        M,
        D1,
        D2,
    )

    P = solution["pi"]
    loss_solver = solution["loss"]["total"][-1]
    all_P_solver.append(P)

torch.save(all_P_ulot, "results/figure_files/all_P_ulot_rho")
torch.save(all_P_solver, "results/figure_files/all_P_solver_rho")


# %% Plans for increasing alpha
####################################################################
#function for generating SBMs with three clusters

torch.manual_seed(42)
np.random.seed(1)

param_file = "parameter_files/params_SBM_50000.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(param_file, "r") as file:
    parameters = json.load(file)


def create_sbm_3_clusters(
    n_graphs=1,
    nc1_i=2,
    nc2_i=2,
    clusters1=[1, 2],
    clusters2=[2, 3],
    random_graph=False,
    noise=0.6,
    p_intra=0.6,
    p_inter=0.03,
    n_nodes1=30,
    n_nodes2=30,
):
    dataset = SBM(
        n_graphs=n_graphs,
        nc1_i=nc1_i,
        nc2_i=nc2_i,
        clusters1=clusters1,
        clusters2=clusters2,
        random_graph=random_graph,
        random_n_nodes=False,
        noise=noise,
        p_intra=p_intra,
        p_inter=p_inter,
        n_nodes1=n_nodes1,
        n_nodes2=n_nodes2,
    )
    graph_pair = dataset.__getitem__(0)
    M = torch.cdist(graph_pair.x_s, graph_pair.x_t) ** 2

    M = M / (10 * torch.mean(M))

    n1 = graph_pair.x_s.shape[0]
    n2 = graph_pair.x_t.shape[0]

    G1 = Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s, y=graph_pair.y_s)
    G2 = Data(x=graph_pair.x_t, edge_index=graph_pair.edge_index_t, y=graph_pair.y_t)

    G1_nx = to_networkx(G1, to_undirected=True)
    G2_nx = to_networkx(G2, to_undirected=True)

    shortest_path_matrix1 = dict(nx.all_pairs_shortest_path_length(G1_nx))
    shortest_path_matrix2 = dict(nx.all_pairs_shortest_path_length(G2_nx))

    D1 = np.full((n1, n1), np.inf)  # Initialize with infinity
    for i, paths in shortest_path_matrix1.items():
        for j, length in paths.items():
            D1[i, j] = length
    D1 = torch.tensor(D1)

    D2 = np.full((n2, n2), np.inf)  # Initialize with infinity
    for i, paths in shortest_path_matrix2.items():
        for j, length in paths.items():
            D2[i, j] = length
    D2 = torch.tensor(D2)

    D1 = D1 / (10 * torch.mean(D1))
    D2 = D2 / (10 * torch.mean(D2))
    max = np.max([torch.max(graph_pair.x_s), torch.max(graph_pair.x_t)])
    min = np.min([torch.min(graph_pair.x_s), torch.min(graph_pair.x_t)])

    rgb1 = (G1.x - min) / (max - min)
    rgb2 = (G2.x - min) / (max - min)

    nc1_1 = torch.sum(G1.y == 0)
    nc2_1 = torch.sum(G1.y == 1)
    nc3_1 = torch.sum(G1.y == 2)
    nc1_2 = torch.sum(G2.y == 0)
    nc2_2 = torch.sum(G2.y == 1)
    nc3_2 = torch.sum(G2.y == 2)

    pos1_center = [0.0, 0.0]
    pos1 = np.random.multivariate_normal(
        pos1_center, [[0.1, 0], [0, 0.1]], size=(nc1_1.item())
    )

    pos2_center = [2.0, 0.0]
    pos2 = np.random.multivariate_normal(
        pos2_center, [[0.1, 0], [0, 0.1]], size=(nc2_1.item())
    )

    pos3_center = [4.0, 0.0]
    pos3 = np.random.multivariate_normal(
        pos3_center, [[0.1, 0], [0, 0.1]], size=(nc3_1.item())
    )

    pos_g1 = np.stack([pos1, pos2, pos3]).reshape(nc1_1 + nc2_1 + nc3_1, 2)

    pos_g1 = {node: (x, y) for node, (x, y) in enumerate(pos_g1)}

    pos1_center = [0.0, 0.0]
    pos1 = np.random.multivariate_normal(
        pos1_center, [[0.1, 0], [0, 0.1]], size=(nc1_2.item())
    )

    pos2_center = [2.0, 0.0]
    pos2 = np.random.multivariate_normal(
        pos2_center, [[0.1, 0], [0, 0.1]], size=(nc2_2.item())
    )

    pos3_center = [4.0, 0.0]
    pos3 = np.random.multivariate_normal(
        pos3_center, [[0.1, 0], [0, 0.1]], size=(nc3_2.item())
    )

    pos_g2 = np.stack([pos1, pos2, pos3]).reshape(nc1_2 + nc2_2 + nc3_2, 2)

    offset = 2

    pos_g2 = {node: (x, y - offset) for node, (x, y) in enumerate(pos_g2)}

    return dataset, pos_g1, pos_g2, G1_nx, G2_nx, D1, D2, M, rgb1, rgb2


# %%
torch.manual_seed(42)
np.random.seed(42)
random.seed(41)


dataset_3, pos_g1, pos_g2, G1_nx, G2_nx, D1, D2, M, rgb1, rgb2 = create_sbm_3_clusters(
    n_graphs=10,
    nc1_i=3,
    nc2_i=3,
    clusters1=[1, 2, 3],
    clusters2=[1, 2, 3],
    random_graph=False,
    noise=0.4,
    p_intra=0.5,
    p_inter=0.05,
    n_nodes1=30,
    n_nodes2=30,
)
graph_pair = dataset_3.__getitem__(1)
graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])

torch.save(pos_g1, "results/figure_files/pos_g1_alpha")
torch.save(pos_g2, "results/figure_files/pos_g2_alpha")
torch.save(G1_nx, "results/figure_files/G1_nx_alpha")
torch.save(rgb1, "results/figure_files/rgb1_alpha")
torch.save(rgb2, "results/figure_files/rgb2_alpha")
torch.save(G2_nx, "results/figure_files/G2_nx_alpha")
torch.save(graph_pair, "results/figure_files/graph_pair_alpha")

rhos = torch.tensor([0.01])
alphas = torch.linspace(0, 1, 3)
n = 4
num = 0
cmap = pl.get_cmap("Reds")

max = np.max([torch.max(graph_pair.x_s), torch.max(graph_pair.x_t)])
min = np.min([torch.min(graph_pair.x_s), torch.min(graph_pair.x_t)])

all_P_ulot = []
all_P_solver = []
for i, alpha in enumerate(alphas):
    batch_indices_s = graph_pair_batch.x_s_batch
    batch_indices_t = graph_pair_batch.x_t_batch
    rho = torch.tensor(rho, dtype=torch.float32).reshape(1, 1)
    alpha = torch.tensor(alpha, dtype=torch.float32).reshape(1, 1)

    batched_rhos_s = rho[batch_indices_s].reshape(len(rho[batch_indices_s]), 1)
    batched_rhos_t = rho[batch_indices_t].reshape(len(rho[batch_indices_t]), 1)
    batched_alphas_s = alpha[batch_indices_s].reshape(len(alpha[batch_indices_s]), 1)
    batched_alphas_t = alpha[batch_indices_t].reshape(len(alpha[batch_indices_t]), 1)

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

    all_P_ulot.append(P)

    rho = rhos[0]
    alpha = alpha[0]
    solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)
    solution = solver.solve(
        alpha,
        rho,
        rho,
        0.0,
        "joint",
        "kl",
        M,
        D1,
        D2,
    )

    P = solution["pi"]

    all_P_solver.append(P)

torch.save(all_P_ulot, "results/figure_files/all_P_ulot_alpha")
torch.save(all_P_solver, "results/figure_files/all_P_solver_alpha")

# %%
