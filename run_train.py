import torch
import json
from ulot.utils import (
    get_filename,
    get_model,
    get_loss,
    get_optimizer,
    get_dataset,
)
from torch_geometric.loader import DataLoader
from ulot.utils import (
    train,
    validation,
)
import numpy as np
import random
import argparse
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-param_file",
    type=str,
    help="name of the parameter file",
    default="params_SBM.json",
)
parser.add_argument(
    "--nosave", dest="save", action="store_false", help="Do not save results"
)
parser.set_defaults(save=True)


args = vars(parser.parse_args())
save = args["save"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

#load the parameter file
param_file = os.path.join("parameter_files", args["param_file"])
with open(param_file, "r") as file:
    parameters = json.load(file)

# load the data
dataset = get_dataset(parameters)
test_amount, val_amount = (
    int(dataset.__len__() * parameters["test_size"]),
    int(dataset.__len__() * parameters["val_size"]),
)

train_set, val_set, _ = torch.utils.data.random_split(
    dataset,
    [(dataset.__len__() - (test_amount + val_amount)), test_amount, val_amount],
)

print(f"number of pairs : {len(dataset)}")

max_samples_per_epoch = 5000  #maximum number of pairs to sample per epoch


indices = np.random.choice(len(dataset), max_samples_per_epoch, replace=False)
train_dataloader = DataLoader(
    train_set,
    batch_size=parameters["batch_size"],
    shuffle=True,
    follow_batch=["x_s", "x_t"],
)
val_dataloader = DataLoader(
    val_set,
    batch_size=parameters["batch_size"],
    shuffle=True,
    follow_batch=["x_s", "x_t"],
)

# paths for saving
path_trained_model = get_filename(parameters, "trained_model")
path_training_losses = get_filename(parameters, "training_losses")

# useful parameters
example_graph = dataset.__getitem__(0)
in_channels = example_graph.x_s.shape[1]

print(f"node feature dimension : {in_channels}")

# get model, loss, optimizer and scheduler
model = get_model(parameters, in_channels)
model = model.to(device)
loss = get_loss(parameters)
optimizer = get_optimizer(parameters, model)

train_losses = {"fgw": [], "gw": [], "w": [], "marginals": []}
val_losses = {"fgw": [], "gw": [], "w": [], "marginals": []}


print("training")
# start training
for epoch in range(parameters["n_epochs"]):
    L_train = train(
        model,
        optimizer,
        loss,
        train_dataloader,
        device,
        max_n_pairs=max_samples_per_epoch,
        rho_range=parameters["rho_range"],
    )

    train_losses["fgw"].append(L_train["total"])
    train_losses["gw"].append(L_train["gromov"])
    train_losses["w"].append(L_train["wasserstein"])
    train_losses["marginals"].append(L_train["marginals"])

    L_val = validation(
        model,
        loss,
        val_dataloader,
        device,
        max_n_pairs=max_samples_per_epoch,
        rho_range=parameters["rho_range"],
    )

    val_losses["fgw"].append(L_val["total"])
    val_losses["gw"].append(L_val["gromov"])
    val_losses["w"].append(L_val["wasserstein"])
    val_losses["marginals"].append(L_val["marginals"])

    print(
        f"Epoch {epoch}, FGW: {L_val['total']},  GW: {L_val['gromov']}, W: {L_val['wasserstein']},  marginals: {L_val['marginals']}"
    )

    if epoch % 50 == 0 or epoch == parameters["n_epochs"] - 1:
        if save:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path_trained_model,
            )
