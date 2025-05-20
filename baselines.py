import json
from tqdm import tqdm
import torch
import time
import ot
from ulot.fugw import FUGWSolver
import numpy as np
import random
from ot.gromov import fused_unbalanced_gromov_wasserstein
import os
from torch_geometric.data import Batch
from ulot.utils import (
    get_filename,
    get_model,
    get_loss,
    get_dataset,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Load data
######################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

save = True

param_file = "params_IBC_individual_1000_local.json"
n_graphs_to_test = 100
param_file = os.path.join("parameter_files", param_file)
with open(param_file, "r") as file:
    parameters = json.load(file)
path_save = "baselines_gpu"

dataset = get_dataset(
    parameters,
    return_subject_info=True,
    geodesic=parameters["geodesic"],
    random_sizes=parameters["random_sizes"],
    write_dir_contrasts=parameters["write_dir_contrasts"],
    no_preloading=True,
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
if parameters["dataset"] == "SBM":
    example_graph = dataset.__getitem__(0)
else:
    example_graph, _ = dataset.__getitem__(0)
in_channels = example_graph.x_s.shape[1]

model = get_model(parameters, in_channels)
model = model.to(device)
path_trained_model = get_filename(parameters, "trained_model")

checkpoint = torch.load(
    path_trained_model, weights_only=False, map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# load loss
loss = get_loss(parameters)

# initialize result dictionary

dic_results = {"mm": {}, "lbfgsb": {}, "ibpp": {}, "filenames_test_solver": []}
dic_results["mm"] = {
    "plans_test_solver": [],
    "losses_test_solver": [],
    "times_test_solver": [],
}
dic_results["lbfgsb"] = {
    "plans_test_solver": [],
    "losses_test_solver": [],
    "times_test_solver": [],
}
dic_results["ibpp"] = {
    "plans_test_solver": [],
    "losses_test_solver": [],
    "times_test_solver": [],
}
dic_results["gnn"] = {
    "plans_test_solver": [],
    "losses_test_solver": [],
    "times_test_solver": [],
}
dic_results["entropic"] = {
    "plans_test_solver": [],
    "losses_test_solver": [],
    "times_test_solver": [],
}


rhos = torch.exp(
    torch.rand((n_graphs_to_test, 1), device=device) * -parameters["rho_range"]
)
alphas = torch.rand((n_graphs_to_test, 1), device=device)
dic_results["rhos"] = rhos
dic_results["alphas"] = alphas


# run solvers
######################################################################################################

# test the trained model on the test set
for type in ["test"]:
    if type == "test":
        inds_graph = inds_graph_test
    for i, ind in enumerate(tqdm(inds_graph)):
        graph_pair, filename_test = test_set.__getitem__(ind)
        dic_results["filenames_test_solver"].append(filename_test)
        graph_pair = graph_pair.to(device)
        rho = rhos[i][0]
        alpha = alphas[i][0]
        solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)
        M = ot.dist(graph_pair.x_s, graph_pair.x_t)

        C1 = graph_pair.geodesic_s
        C2 = graph_pair.geodesic_t
        C1 = C1[:, : M.shape[0]]
        C2 = C2[:, : M.shape[1]]
        C1 = C1 / (10 * torch.mean(C1))
        C2 = C2 / (10 * torch.mean(C2))
        M = M / (10 * torch.mean(M))

        alpha_POT = (1 - alpha) / alpha

        print("mm")
        try:
            start_mm = time.time()
            solution_mm = fused_unbalanced_gromov_wasserstein(
                C1,
                C2,
                M=M,
                epsilon=0,
                alpha=alpha_POT,
                reg_marginals=(rho.item() / alpha.item()),
                log=True,
                unbalanced_solver="mm",
            )
            end_mm = time.time()
            dic_results["mm"]["plans_test_solver"].append(solution_mm[0])
            dic_results["mm"]["losses_test_solver"].append(
                solution_mm[2]["fugw_cost"] * alpha
            )
            dic_results["mm"]["times_test_solver"].append(end_mm - start_mm)
        except ValueError as e:
            if "NaN" in str(e):  # optionally check that it's a NaN-related error
                print("Caught NaN error in mm, skipping...")
                dic_results["mm"]["plans_test_solver"].append(np.nan)
                dic_results["mm"]["losses_test_solver"].append(np.nan)
                dic_results["mm"]["times_test_solver"].append(np.nan)

        print("lbfgsb")
        try:
            start_lbfgsb = time.time()
            solution_lbfgsb = fused_unbalanced_gromov_wasserstein(
                C1,
                C2,
                M=M,
                epsilon=0,
                alpha=alpha_POT,
                reg_marginals=(rho.item() / alpha.item()),
                log=True,
                unbalanced_solver="lbfgsb",
            )
            end_lbfgsb = time.time()
            dic_results["lbfgsb"]["plans_test_solver"].append(solution_lbfgsb[0])
            dic_results["lbfgsb"]["losses_test_solver"].append(
                solution_lbfgsb[2]["fugw_cost"] * alpha
            )
            dic_results["lbfgsb"]["times_test_solver"].append(end_lbfgsb - start_lbfgsb)
        except ValueError as e:
            if "NaN" in str(e):  # optionally check that it's a NaN-related error
                print("Caught NaN error in lbfgsb, skipping...")
                dic_results["lbfgsb"]["plans_test_solver"].append(np.nan)
                dic_results["lbfgsb"]["losses_test_solver"].append(np.nan)
                dic_results["lbfgsb"]["times_test_solver"].append(np.nan)

        print("entropic")
        try:
            start_entropic = time.time()
            solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)

            solution = solver.solve(
                alpha,
                rho,
                rho,
                1.0,
                "joint",
                "kl",
                M,
                C1,
                C2,
            )
            end_entropic = time.time()
            n0 = graph_pair.x_s.shape[0]
            n1 = graph_pair.x_t.shape[0]
            p = torch.ones(n0, device=device) / n0
            q = torch.ones(n1, device=device) / n1
            loss_entropic = solver.fugw_loss(
                solution["pi"],
                solution["pi"],
                data_const=[C1**2, C2**2, C1, C2, M],
                tuple_weights=[p, q, torch.outer(p, q)],
                hyperparams=[
                    rhos[i][0],
                    rhos[i][0],
                    parameters["eps"],
                    alphas[i][0],
                    "joint",
                    "kl",
                ],
            )["total"]
            dic_results["entropic"]["plans_test_solver"].append(solution["pi"])
            dic_results["entropic"]["losses_test_solver"].append(loss_entropic)
            dic_results["entropic"]["times_test_solver"].append(
                end_entropic - start_entropic
            )
        except ValueError as e:
            if "NaN" in str(e):
                print("Caught NaN error in entropic, skipping...")
                dic_results["entropic"]["plans_test_solver"].append(np.nan)
                dic_results["entropic"]["losses_test_solver"].append(np.nan)
                dic_results["entropic"]["times_test_solver"].append(np.nan)

        print("ibpp")
        start = time.time()
        solver = FUGWSolver(tol_bcd=1e-6, tol_uot=1e-6, tol_loss=1e-6)

        solution = solver.solve(
            alpha,
            rho,
            rho,
            parameters["eps"],
            "joint",
            "kl",
            M,
            C1,
            C2,
        )

        end = time.time()

        try:
            dic_results["ibpp"]["plans_test_solver"].append(solution["pi"])
            dic_results["ibpp"]["losses_test_solver"].append(
                solution["loss"]["total"][-1]
            )
            dic_results["ibpp"]["times_test_solver"].append(end - start)
        except TypeError:
            dic_results["ibpp"]["plans_test_solver"].append(torch.nan)
            dic_results["ibpp"]["losses_test_solver"].append(torch.nan)
            dic_results["ibpp"]["times_test_solver"].append(torch.nan)

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
        print("gnn")
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
        end = time.time()
        L = loss(
            P,
            graph_pair.x_s,
            graph_pair.x_t,
            graph_pair.edge_index_s,
            graph_pair.edge_index_t,
            batch1=graph_pair.x_s_batch,
            batch2=graph_pair.x_t_batch,
            mask1=mask1,
            mask2=mask2,
            rhos=rhos[i][0],
            alphas=alphas[i][0],
            geodesic=parameters["geodesic"],
            geodesic_s=graph_pair.geodesic_s,
            geodesic_t=graph_pair.geodesic_t,
        )
        dic_results["gnn"]["plans_test_solver"].append(P)
        dic_results["gnn"]["losses_test_solver"].append(L["total"])
        dic_results["gnn"]["times_test_solver"].append(end - start)


if save:
    torch.save(dic_results, path_save)
