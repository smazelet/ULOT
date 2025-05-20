import json
from ulot.utils import (
    get_filename,
    get_dataset,
    get_loss,
)
from tqdm import tqdm
import torch
import time
import ot
from ulot.fugw import FUGWSolver
import numpy as np
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


param_file = "parameter_files/params_SBM_10000.json"
save = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

n_graphs_to_test = 200

with open(param_file, "r") as file:
    parameters = json.load(file)

path_save = get_filename(parameters, "solver_losses")

# load data

if parameters["dataset"] == "SBM":
    dataset = get_dataset(
        parameters,
    )

test_amount, val_amount = (
    int(dataset.__len__() * parameters["test_size"]),
    int(dataset.__len__() * parameters["val_size"]),
)

train_set, _, test_set = torch.utils.data.random_split(
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

# load loss
loss = get_loss(parameters)

# initialize result dictionary
dic_results = {}

dic_results["plans_test_solver"] = []
dic_results["filenames_test_solver"] = []
dic_results["filenames_train_solver"] = []
dic_results["losses_test_solver"] = []
dic_results["losses_test_solver"] = []
dic_results["losses_test_solver_all"] = []
dic_results["times_test_solver"] = []
dic_results["plans_train_solver"] = []
dic_results["losses_train_solver"] = []
dic_results["losses_train_solver_all"] = []
dic_results["times_train_solver"] = []

rhos = torch.exp(
    torch.rand((n_graphs_to_test, 1), device=device) * -parameters["rho_range"]
)
alphas = torch.rand((n_graphs_to_test, 1), device=device)
dic_results["rhos"] = rhos
dic_results["alphas"] = alphas

# test the trained model on the test set
for type in ["train", "test"]:
    if type == "test":
        inds_graph = inds_graph_test
    else:
        inds_graph = inds_graph_train

    # test the trained model
    print("testing on test data for random rhos")
    for i, ind in enumerate(tqdm(inds_graph)):
        if type == "train":
            if parameters["dataset"] == "SBM":
                graph_pair = train_set.__getitem__(ind)
            else:
                graph_pair, filename_train = train_set.__getitem__(ind)
                dic_results["filenames_train_solver"].append(filename_train)
        elif type == "test":
            if parameters["dataset"] == "SBM":
                graph_pair = test_set.__getitem__(ind)
            else:
                graph_pair, filename_test = test_set.__getitem__(ind)
                dic_results["filenames_test_solver"].append(filename_test)
        graph_pair = graph_pair.to(device)
        rho = rhos[i][0]
        alpha = alphas[i][0]
        solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)
        M = ot.dist(graph_pair.x_s, graph_pair.x_t)
        C1, C2 = graph_pair.connectivity_s, graph_pair.connectivity_t
        C1 = C1[:, : M.shape[0]]
        C2 = C2[:, : M.shape[1]]

        C1 = C1 / (10 * torch.mean(C1))
        C2 = C2 / (10 * torch.mean(C2))
        M = M / (10 * torch.mean(M))

        start = time.time()

        solution = solver.solve(
            alpha,
            rho,
            rho,
            0.0,
            "joint",
            "kl",
            M,
            C1,
            C2,
        )
        end = time.time()

        if type == "test":
            try:
                dic_results["plans_test_solver"].append(solution["pi"])
                dic_results["losses_test_solver"].append(solution["loss"]["total"][-1])
                dic_results["losses_test_solver_all"].append(solution["loss"]["total"])
                dic_results["times_test_solver"].append(solution["times"])
            except TypeError:
                dic_results["plans_test_solver"].append(torch.nan)
                dic_results["losses_test_solver"].append(torch.nan)
                dic_results["losses_test_solver_all"].append(torch.nan)
                dic_results["times_test_solver"].append(torch.nan)

        if type == "train":
            try:
                dic_results["plans_train_solver"].append(solution["pi"])
                dic_results["losses_train_solver"].append(solution["loss"]["total"][-1])
                dic_results["losses_train_solver_all"].append(solution["loss"]["total"])
                dic_results["times_train_solver"].append(solution["times"])
            except TypeError:
                dic_results["plans_train_solver"].append(torch.nan)
                dic_results["losses_train_solver"].append(torch.nan)
                dic_results["losses_train_solver_all"].append(torch.nan)
                dic_results["times_train_solver"].append(torch.nan)

if save:
    np.save(
        path_save,
        dic_results,
    )
