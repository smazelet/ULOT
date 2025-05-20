import json
from ulot.utils import (
    get_filename,
    get_dataset,
    get_model,
    get_loss,
)
from tqdm import tqdm
import torch
import time
import warnings
import numpy as np
from torch_geometric.data import Batch
import argparse
import random
import os


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-param_file",
    type=str,
    help="name of the parameter file",
    default="params_SBM_10000.json",
)
parser.add_argument(
    "--nosave", dest="save", action="store_false", help="Do not save results"
)
parser.set_defaults(save=True)

args = vars(parser.parse_args())
save = args["save"]


param_file = os.path.join("parameter_files", args["param_file"])
with open(param_file, "r") as file:
    parameters = json.load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_graphs_to_test = 200

# load data

dataset = get_dataset(
    parameters,
)

test_amount, val_amount = (
    int(dataset.__len__() * parameters["test_size"]),
    int(dataset.__len__() * parameters["val_size"]),
)

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [(dataset.__len__() - (test_amount + val_amount)), test_amount, val_amount],
)
inds_graph_test = torch.randint(low=0, high=test_amount, size=(n_graphs_to_test,))
inds_graph_train = torch.randint(
    low=0,
    high=dataset.__len__() - (test_amount + val_amount),
    size=(n_graphs_to_test,),
)

example_graph = dataset.__getitem__(0)
in_channels = example_graph.x_s.shape[1]

# load trained model
model = get_model(parameters, in_channels)
model = model.to(device)
path_trained_model = get_filename(parameters, "trained_model")
path_solver_losses = get_filename(parameters, "solver_losses")

print(path_trained_model)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    solver_losses = np.load(path_solver_losses, allow_pickle=True).item()
checkpoint = torch.load(
    path_trained_model, weights_only=False, map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# load loss
loss = get_loss(parameters)

times = []

# initialize result dictionary
dic_results = {
    "plans_test": [],
    "losses_test": [],
    "plans_train": [],
    "losses_train": [],
    "times_train": [],
    "times_test": [],
}

rhos = solver_losses["rhos"].to(device)
alphas = solver_losses["alphas"].to(device)
dic_results["rhos"] = rhos
dic_results["alphas"] = alphas

# test the trained model on the test set
print("testing")
for type in ["train", "test"]:
    if type == "train":
        inds_graph = inds_graph_train
    else:
        inds_graph = inds_graph_test

    for i, ind in tqdm(enumerate(inds_graph)):
        if type == "train":
            graph_pair = train_set.__getitem__(ind)
        elif type == "test":
            graph_pair = test_set.__getitem__(ind)

        graph_pair = Batch.from_data_list([graph_pair], follow_batch=["x_s", "x_t"])
        graph_pair = graph_pair.to(device)
        batch_indices_s = graph_pair.x_s_batch
        batch_indices_t = graph_pair.x_t_batch
        batched_rhos_s = rhos[i][batch_indices_s].reshape(
            len(rhos[i][batch_indices_s]), 1
        )
        batched_rhos_t = rhos[i][batch_indices_t].reshape(
            len(rhos[i][batch_indices_t]), 1
        )
        batched_alphas_s = alphas[i][batch_indices_s].reshape(
            len(alphas[i][batch_indices_s]), 1
        )
        batched_alphas_t = alphas[i][batch_indices_t].reshape(
            len(alphas[i][batch_indices_t]), 1
        )
        start = time.time()
        P, mask1, mask2 = model(
            graph_pair.x_s,
            graph_pair.x_t,
            graph_pair.edge_index_s,
            graph_pair.edge_index_t,
            graph_pair.x_s_batch,
            graph_pair.x_t_batch,
            batched_rhos_s,
            batched_rhos_t,
            batched_alphas_s,
            batched_alphas_t,
        )
        L = loss(
            P=P,
            x1=graph_pair.x_s,
            x2=graph_pair.x_t,
            batch1=graph_pair.x_s_batch,
            batch2=graph_pair.x_t_batch,
            mask1=mask1,
            mask2=mask2,
            rhos=rhos[i][0],
            alphas=alphas[i][0],
            connectivity_s=graph_pair.connectivity_s,
            connectivity_t=graph_pair.connectivity_t,
        )
        end = time.time()

        if type == "test":
            dic_results["plans_test"].append(P)
            dic_results["losses_test"].append(L)
            dic_results["times_test"].append(end - start)
        else:
            dic_results["plans_train"].append(P)
            dic_results["losses_train"].append(L)
            dic_results["times_train"].append(end - start)


path_save_perfs = get_filename(parameters, "test_performances")

if save:
    torch.save(dic_results, path_save_perfs)
