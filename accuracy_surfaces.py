# %%
import json
import torch
import numpy as np
from torch_geometric.data import Batch
from ulot.utils_data import SBM
from ulot.utils import (
    get_filename,
    get_model,
)
from tqdm import tqdm
import torch.nn.functional as F
import random


# %%
torch.manual_seed(40)
np.random.seed(6)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %% Load model
###################################################################################################


torch.manual_seed(42)
np.random.seed(42)
random.seed(41)

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


# %% Surface for clusters (1,2,3) -> (1,2)


torch.manual_seed(42)
np.random.seed(42)
random.seed(41)

n_graphs = 1

dataset_2 = SBM(
    n_graphs=n_graphs,
    nc1_i=3,
    nc2_i=2,
    clusters1=[1, 2, 3],
    clusters2=[1, 2],
    random_graph=False,
    random_n_nodes=True,
    noise=0.5,
    p_intra=0.5,
    p_inter=0.05,
)
n_alpha = 50
n_rho = 50
alphas = torch.linspace(0, 1, n_alpha).reshape(n_alpha, 1)
rhos = torch.exp(-torch.linspace(0, 6, n_rho)).reshape(n_rho, 1)


all_rhos_2 = torch.zeros(n_graphs, n_alpha, n_rho)
all_alphas_2 = torch.zeros(n_graphs, n_alpha, n_rho)
all_losses_2 = torch.zeros(n_graphs, n_alpha, n_rho)
all_accuracy_2 = torch.zeros(n_graphs, n_alpha, n_rho)
all_sums = torch.zeros(n_graphs, n_alpha, n_rho)

for k in tqdm(range(n_graphs)):
    for i, alpha in enumerate(alphas):
        for j, rho in enumerate(rhos):
            # graph_pair = torch.load("graph_pair")
            graph_pair = dataset_2.__getitem__(k)
            graph_pair_batch = Batch.from_data_list(
                [graph_pair], follow_batch=["x_s", "x_t"]
            )

            batch_indices_s = graph_pair_batch.x_s_batch
            batch_indices_t = graph_pair_batch.x_t_batch
            n_nodes_target = graph_pair_batch.x_t.shape[0]
            validation_nodes = np.random.randint(
                0, n_nodes_target, int(0.5 * n_nodes_target)
            )
            n_validation_nodes = len(validation_nodes)
            batched_rhos_s = rho[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
            batched_rhos_t = rho[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
            batched_alphas_s = alpha[batch_indices_s].reshape(
                len(alphas[batch_indices_s]), 1
            )
            batched_alphas_t = alpha[batch_indices_t].reshape(
                len(alphas[batch_indices_t]), 1
            )

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
            # print(P)

            col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
            safe_col_sums = torch.clamp(col_sums, min=1e-15)

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
            bool_accuracy = predicted_labels == labels_t
            # loss = criterion(logits, labels_t)
            loss = F.kl_div(log_probs, one_hot_matrix_t, reduction="batchmean")

            accuracy = torch.sum(bool_accuracy) / n_nodes_target

            all_losses_2[k, i, j] = loss.detach()
            accuracy = accuracy.detach()
            all_accuracy_2[k, i, j] = accuracy


# %% Surface for clusters (1,2,3) -> (1,2,3)


torch.manual_seed(42)
np.random.seed(42)
random.seed(41)

n_graphs = 1

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
n_alpha = 50
n_rho = 50
alphas = torch.linspace(0, 1, n_alpha).reshape(n_alpha, 1)
rhos = torch.exp(-torch.linspace(0, 6, n_rho)).reshape(n_rho, 1)


all_rhos_3 = torch.zeros(n_graphs, n_alpha, n_rho)
all_alphas_3 = torch.zeros(n_graphs, n_alpha, n_rho)
all_losses_3 = torch.zeros(n_graphs, n_alpha, n_rho)
all_accuracy_3 = torch.zeros(n_graphs, n_alpha, n_rho)
all_sums = torch.zeros(n_graphs, n_alpha, n_rho)

for k in tqdm(range(n_graphs)):
    for i, alpha in enumerate(alphas):
        for j, rho in enumerate(rhos):
            # graph_pair = torch.load("graph_pair")
            graph_pair = dataset_3.__getitem__(k)
            graph_pair_batch = Batch.from_data_list(
                [graph_pair], follow_batch=["x_s", "x_t"]
            )

            batch_indices_s = graph_pair_batch.x_s_batch
            batch_indices_t = graph_pair_batch.x_t_batch
            n_nodes_target = graph_pair_batch.x_t.shape[0]
            validation_nodes = np.random.randint(
                0, n_nodes_target, int(0.5 * n_nodes_target)
            )
            n_validation_nodes = len(validation_nodes)
            batched_rhos_s = rho[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
            batched_rhos_t = rho[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
            batched_alphas_s = alpha[batch_indices_s].reshape(
                len(alphas[batch_indices_s]), 1
            )
            batched_alphas_t = alpha[batch_indices_t].reshape(
                len(alphas[batch_indices_t]), 1
            )

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
            # print(P)

            col_sums = torch.sum(P[0], dim=0)  # Shape: [n_t]
            safe_col_sums = torch.clamp(col_sums, min=1e-15)

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
            bool_accuracy = predicted_labels == labels_t
            # loss = criterion(logits, labels_t)
            loss = F.kl_div(log_probs, one_hot_matrix_t, reduction="batchmean")

            accuracy = torch.sum(bool_accuracy) / n_nodes_target
            accuracy = accuracy.detach()
            all_accuracy_3[k, i, j] = accuracy


# %%

torch.save(all_accuracy_2, "results/figure_files/all_accuracy_2")
torch.save(all_accuracy_3, "results/figure_files/all_accuracy_3")
torch.save(rhos, "results/figure_files/rhos")
torch.save(alphas, "results/figure_files/alphas")
