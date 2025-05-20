# %%
import json
import torch
import numpy as np
from torch_geometric.data import Batch
from ulot.utils_data import SBM
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ulot.utils import (
    get_filename,
    get_model,
    get_loss,
)
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops

# %%

torch.manual_seed(40)
np.random.seed(6)


param_file = "parameter_files/params_SBM_50000.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open(param_file, "r") as file:
    parameters = json.load(file)

dataset = SBM(
    n_graphs=1,
    nc1_i=3,
    nc2_i=2,
    clusters1=[1, 2, 3],
    clusters2=[1, 2],
    random_graph=False,
    noise=0.3,
    p_intra=0.6,
    p_inter=0.05,
)

graph_pair = dataset.__getitem__(0)

G1 = Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s, y=graph_pair.y_s)
G2 = Data(x=graph_pair.x_t, edge_index=graph_pair.edge_index_t, y=graph_pair.y_t)
G1_nx = to_networkx(G1, to_undirected=True)
G2_nx = to_networkx(G2, to_undirected=True)

max = np.max([torch.max(graph_pair.x_s), torch.max(graph_pair.x_t)])
min = np.min([torch.min(graph_pair.x_s), torch.min(graph_pair.x_t)])


# %% Minimization for alpha=1.0
###################################################################################################

torch.manual_seed(40)
np.random.seed(6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_file = "parameter_files/params_SBM_50000.json"

plot = False


rhos = torch.exp(-torch.tensor([3])).reshape(1, 1)
rhos = torch.tensor([0.05]).reshape(1, 1)
alphas = torch.tensor([1.0]).reshape(1, 1)

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

loss = get_loss(parameters)

graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
batch_indices_s = graph_pair_batch.x_s_batch
batch_indices_t = graph_pair_batch.x_t_batch

batched_rhos_s = rhos[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
batched_rhos_t = rhos[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
batched_alphas_s = alphas[batch_indices_s].reshape(len(alphas[batch_indices_s]), 1)
batched_alphas_t = alphas[batch_indices_t].reshape(len(alphas[batch_indices_t]), 1)


n_steps = 4000
losses = np.zeros(n_steps)
F = graph_pair_batch.x_s
F.requires_grad_()
G = graph_pair_batch.connectivity_s
G.requires_grad_()
dt = 5
dt_geo = 300


F_along_step_01 = np.zeros((n_steps, G1.x.shape[0], G1.x.shape[1]))
G_along_step_01 = np.zeros(
    (
        n_steps,
        graph_pair_batch.connectivity_s.shape[0],
        graph_pair_batch.connectivity_s.shape[1],
    )
)
edge_index_flow = G1.edge_index

F_grad = torch.ones(1)
G_grad = torch.ones(1)

for n in tqdm(range(n_steps)):
    F_along_step_01[n] = F.detach().numpy()
    G_along_step_01[n] = G.detach().numpy()
    P, mask1, mask2 = model(
        F,
        graph_pair_batch.x_t,
        edge_index_flow,
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
        x1=F,
        x2=graph_pair_batch.x_t,
        batch1=graph_pair_batch.x_s_batch,
        batch2=graph_pair_batch.x_t_batch,
        mask1=mask1,
        mask2=mask2,
        rhos=rhos[0],
        alphas=alphas[0],
        connectivity_s=G,
        connectivity_t=graph_pair_batch.connectivity_t,
    )
    losses[n] = L["total"]

    F.retain_grad()
    G.retain_grad()
    L["total"].backward()
    F_grad = F.grad
    G_grad = G.grad
    F = F - dt * F.grad
    G = G - dt_geo * G.grad

    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
n_1 = n

# %% Minimization for alpha=0.5
###################################################################################################

torch.manual_seed(40)
np.random.seed(6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_file = "parameter_files/params_SBM_50000.json"

plot = False


rhos = torch.exp(-torch.tensor([3])).reshape(1, 1)

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

loss = get_loss(parameters)

graph_pair_batch = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
batch_indices_s = graph_pair_batch.x_s_batch
batch_indices_t = graph_pair_batch.x_t_batch

rhos = torch.tensor([0.01]).reshape(1, 1)
alphas = torch.tensor([0.5]).reshape(1, 1)

batched_rhos_s = rhos[batch_indices_s].reshape(len(rhos[batch_indices_s]), 1)
batched_rhos_t = rhos[batch_indices_t].reshape(len(rhos[batch_indices_t]), 1)
batched_alphas_s = alphas[batch_indices_s].reshape(len(alphas[batch_indices_s]), 1)
batched_alphas_t = alphas[batch_indices_t].reshape(len(alphas[batch_indices_t]), 1)


n_steps = 4000
losses = np.zeros(n_steps)
F = graph_pair_batch.x_s
F.requires_grad_()
G = graph_pair_batch.connectivity_s
G.requires_grad_()
dt = 5
dt_geo = 300


F_along_step_05 = np.zeros((n_steps, G1.x.shape[0], G1.x.shape[1]))
G_along_step_05 = np.zeros(
    (
        n_steps,
        graph_pair_batch.connectivity_s.shape[0],
        graph_pair_batch.connectivity_s.shape[1],
    )
)
edge_index_flow = G1.edge_index

F_grad = torch.ones(1)
G_grad = torch.ones(1)


for n in tqdm(range(n_steps)):
    if torch.norm(F_grad) < 10**-5 and torch.norm(G_grad) < 10**-5:
        break
    F_along_step_05[n] = F.detach().numpy()
    G_along_step_05[n] = G.detach().numpy()
    P, mask1, mask2 = model(
        F,
        graph_pair_batch.x_t,
        edge_index_flow,
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
        x1=F,
        x2=graph_pair_batch.x_t,
        batch1=graph_pair_batch.x_s_batch,
        batch2=graph_pair_batch.x_t_batch,
        mask1=mask1,
        mask2=mask2,
        rhos=rhos[0],
        alphas=alphas[0],
        connectivity_s=G,
        connectivity_t=graph_pair_batch.connectivity_t,
    )
    losses[n] = L["total"]

    F.retain_grad()
    G.retain_grad()
    L["total"].backward()
    F_grad = F.grad
    G_grad = G.grad
    F = F - dt * F.grad
    G = G - dt_geo * G.grad

    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
n_05 = n

# %%

n_steps = n_05
steps_01 = [0, 400, 800, 1200, 1778]

G_flows_01 = []
for step in steps_01:
    F = F_along_step_05[step]
    G = torch.tensor(G_along_step_05[step])
    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
    output_graph = Data(x=F, edge_index=edge_index_flow, y=G1.y)
    output_graph_nx = to_networkx(output_graph, to_undirected=True)
    rgb_output = (output_graph.x - min) / (max - min)

    max = np.max(
        [
            torch.max(graph_pair.x_s),
            torch.max(graph_pair.x_t),
            torch.max(torch.tensor(F)),
        ]
    )
    min = np.min(
        [
            torch.min(graph_pair.x_s),
            torch.min(graph_pair.x_t),
            torch.min(torch.tensor(F)),
        ]
    )

    rgb1 = (G1.x - min) / (max - min)
    rgb2 = (G2.x - min) / (max - min)
    rgb_output = (F - min) / (max - min)

    G_flow = Data(x=torch.tensor(F), edge_index=edge_index_flow)
    G_flows_01.append(G_flow)
# %%

steps_05 = [0, 400, 800, 1200, 1778]

G_flows_05 = []
for step in steps_05:
    F = F_along_step_05[step]
    G = torch.tensor(G_along_step_05[step])
    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
    output_graph = Data(x=F, edge_index=edge_index_flow, y=G1.y)
    output_graph_nx = to_networkx(output_graph, to_undirected=True)
    rgb_output = (output_graph.x - min) / (max - min)

    max = np.max(
        [
            torch.max(graph_pair.x_s),
            torch.max(graph_pair.x_t),
            torch.max(torch.tensor(F)),
        ]
    )
    min = np.min(
        [
            torch.min(graph_pair.x_s),
            torch.min(graph_pair.x_t),
            torch.min(torch.tensor(F)),
        ]
    )

    rgb1 = (G1.x - min) / (max - min)
    rgb2 = (G2.x - min) / (max - min)
    rgb_output = (F - min) / (max - min)

    G_flow = Data(x=torch.tensor(F), edge_index=edge_index_flow)
    G_flows_05.append(G_flow)


# %% Save results

steps_01 = [0, 100, 300, 1000, 3000]
steps_05 = [0, 100, 300, 1000, 3000]

F_along_step_01_sub = [F_along_step_01[n] for n in steps_01]
F_along_step_05_sub = [F_along_step_05[n] for n in steps_05]
G_along_step_01_sub = [G_along_step_01[n] for n in steps_01]
G_along_step_05_sub = [G_along_step_05[n] for n in steps_05]


torch.save(F_along_step_01_sub, "results/figure_files/F_along_step_01_sub")
torch.save(G_along_step_01_sub, "results/figure_files/G_along_step_01_sub")

torch.save(F_along_step_05_sub, "results/figure_files/F_along_step_05_sub")
torch.save(G_along_step_05_sub, "results/figure_files/G_along_step_05_sub")
torch.save(G2, "results/figure_files/G2_gradient_flow")
