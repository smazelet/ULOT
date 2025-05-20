import torch
from ulot.losses import FUGW
from ulot.architectures import ULOT_net
import os
from ulot.utils_data import SBM
from scipy.stats import beta


def train(model, optimizer, loss, dataloader, device, max_n_pairs=None, rho_range=9):
    """
    Trains the model.

    Parameters
    ----------
    model:
      Model to train
    optimizer:
      Optimizer for training
    loss:
      Loss function to use
    dataloader: torch_geometric.loader.DataLoader
      DataLoader for the training
    device: str
      Device to train on
    max_n_pairs: int
        Maximum number of pairs to train on during one epoch
    rho_range: int
        Range of rho values to sample from 1 to e^(-rho_range)
    """

    model.train()

    fugw_loss, gw_loss, w_loss, marginals = 0.0, 0.0, 0.0, 0.0
    all_rhos = []
    all_alphas = []
    total_n_pairs = 0
    n_batches = 0
    for graph_pair in dataloader:
        if max_n_pairs is not None and total_n_pairs > max_n_pairs:
            break
    n_batches += 1
    n_graphs = graph_pair.num_graphs
    total_n_pairs += n_graphs
    batch_indices_s = graph_pair.x_s_batch
    batch_indices_t = graph_pair.x_t_batch
    rhos = torch.exp(torch.rand(n_graphs, 1, device=device) * -rho_range)
    rand_num = torch.rand(n_graphs)
    indices_0 = torch.where(rand_num < 0.05)[0]
    indices_1 = torch.where(rand_num > 0.95)[0]
    alphas = torch.tensor(beta.rvs(0.5, 0.5, size=(n_graphs, 1))).to(device)
    alphas[indices_0] = 0.0
    alphas[indices_1] = 1.0
    batched_rhos_s = rhos[batch_indices_s]
    batched_rhos_t = rhos[batch_indices_t]
    batched_alphas_s = alphas[batch_indices_s]
    batched_alphas_t = alphas[batch_indices_t]
    all_rhos.append(rhos)
    all_alphas.append(alphas)
    graph_pair = graph_pair.to(device)
    optimizer.zero_grad()

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
        rhos=rhos,
        connectivity_s=graph_pair.connectivity_s,
        connectivity_t=graph_pair.connectivity_t,
        alphas=alphas,
    )
    L["total"].mean().backward()
    optimizer.step()

    fugw_loss += L["total"].mean().item()
    gw_loss += L["gromov"].mean().item()
    w_loss += L["wasserstein"].mean().item()
    marginals += L["marginals"].mean().item()
    return {
        "total": fugw_loss / n_batches,
        "gromov": gw_loss / n_batches,
        "wasserstein": w_loss / n_batches,
        "marginals": marginals / n_batches,
        "rhos": all_rhos,
    }


def validation(model, loss, dataloader, device, max_n_pairs=None, rho_range=9):
    """
    Validates the model.
    Parameters
    ----------
    model:
        Model to validate
    loss:
        Loss function to use
    dataloader: torch_geometric.loader.DataLoader
        DataLoader for the validation
    device: str
        Device to validate on
    max_n_pairs: int
        Maximum number of pairs to validate on during one epoch
    rho_range: int
        Range of rho values to sample from 1 to e^(-rho_range)
    """

    model.eval()

    fugw_loss, gw_loss, w_loss, marginals = 0.0, 0.0, 0.0, 0.0
    all_rhos = []
    all_alphas = []
    total_n_pairs = 0
    n_batches = 0
    for graph_pair in dataloader:
        if max_n_pairs is not None and total_n_pairs > max_n_pairs:
            break
        n_graphs = graph_pair.num_graphs
        n_batches += 1
        total_n_pairs += n_graphs
        batch_indices_s = graph_pair.x_s_batch
        batch_indices_t = graph_pair.x_t_batch
        rhos = torch.exp(torch.rand(n_graphs, 1, device=device) * -rho_range)
        rand_num = torch.rand(n_graphs)
        indices_0 = torch.where(rand_num < 0.05)[0]
        indices_1 = torch.where(rand_num > 0.95)[0]
        alphas = torch.tensor(beta.rvs(0.5, 0.5, size=(n_graphs, 1))).to(device)
        alphas[indices_0] = 0.0
        alphas[indices_1] = 1.0
        batched_rhos_s = rhos[batch_indices_s]
        batched_rhos_t = rhos[batch_indices_t]
        batched_alphas_s = alphas[batch_indices_s]
        batched_alphas_t = alphas[batch_indices_t]
        all_rhos.append(rhos)
        all_alphas.append(alphas)
        graph_pair = graph_pair.to(device)
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
            rhos=rhos,
            connectivity_s=graph_pair.connectivity_s,
            connectivity_t=graph_pair.connectivity_t,
            alphas=alphas,
        )

        fugw_loss += L["total"].mean().item()
        gw_loss += L["gromov"].mean().item()
        w_loss += L["wasserstein"].mean().item()
        marginals += L["marginals"].mean().item()
    return {
        "total": fugw_loss / n_batches,
        "gromov": gw_loss / n_batches,
        "wasserstein": w_loss / n_batches,
        "marginals": marginals / n_batches,
        "rhos": rhos,
    }


def get_filename(
    params,
    file_type,
):
    """
    Get the filename for the given file type and parameters.
    Parameters
    ----------
    params: dict
        Parameters for the file name
    file_type: str
        Type of file to get the name for
    Returns
    -------
    filename: str
        Filename for the given file type and parameters
    """

    if file_type == "true_losses":
        if params["dataset"] == "SBM":
            filename = (
                "results/true_losses/"
                + params["dataset"]
                + "_"
                + str(params["n_pairs"])
                + ".npy"
            )

    elif file_type == "training_losses" or file_type == "test_performances":
        sorted_params = {key: params[key] for key in sorted(params)}
        basedir = "results/" + sorted_params["model_type"] + "/" + file_type + "/"
        filename_parts = [
            f"{key[:3]}={value}"
            for key, value in sorted_params.items()
            if not (key == "model_type")
        ]
        filename = basedir + "_".join(filename_parts)
        os.makedirs(basedir, exist_ok=True)

    elif file_type == "trained_model":
        sorted_params = {key: params[key] for key in sorted(params)}
        basedir = "results/" + sorted_params["model_type"] + "/" + file_type + "/"
        filename_parts = [
            f"{key[:3]}={value}"
            for key, value in sorted_params.items()
            if not (key == "model_type")
        ]
        filename = basedir + "_".join(filename_parts) + ".pt"
        os.makedirs(basedir, exist_ok=True)

    return filename


def get_dataset(
    params,
):
    """
    Get the dataset for the given parameters.
    Parameters
    ----------
    params: dict
        Parameters for the dataset
    Returns
    -------
    dataset: torch_geometric.data.Dataset
        Dataset for the given parameters
    """

    dataset_name = params["dataset"]
    if dataset_name == "SBM":
        n_pairs = params.get("n_pairs")
        dataset = SBM(n_pairs)
    return dataset


def get_model(params, in_channels):
    """
    Get the model for the given parameters.
    Parameters
    ----------
    params: dict
        Parameters for the model
    in_channels: int
        Number of input channels
    Returns
    -------
    model: torch.nn.Module
        Model for the given parameters
    """
    hidden_channels = params.get("hidden_channels")
    num_layers = params.get("num_layers")
    out_channels = params.get("out_channels")
    hidden_channels_message = params.get("hidden_channels_message")
    temperature = params.get("temperature")
    alpha_enc = params.get("alpha_enc")

    model = ULOT_net(
        in_channels,
        hidden_channels,
        hidden_channels_message,
        out_channels,
        num_layers,
        temperature,
        alpha_enc,
    )

    return model


def get_loss(params):
    loss_name = params["loss_name"]
    model_type = params["model_type"]

    if loss_name == "FUGW" and model_type == "ULOT":
        loss = FUGW

    return loss


def get_optimizer(params, model):
    lr = params["lr"]
    if params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def batch_plans_to_list(P, mask1, mask2):
    bs, _, _ = P.shape
    plans = []
    for ind in range(bs):
        P_i = P[ind]
        plans.append(P_i[mask1[ind]][:, mask2[ind]])
    return plans
