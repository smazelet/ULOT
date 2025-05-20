import torch
import ot
from torch_geometric.utils import to_dense_batch


def gw_loss(P, batch1, batch2, mask1, mask2, connectivity_s=None, connectivity_t=None):
    """
    Gromov Wasserstein loss between graph 1 with n1 nodes and graph 2 with n2 nodes. The loss is the L2 loss.

    Parameters
    ----------
    P: torch.Tensor, shape (n1, n2)
      Transport plan
    batch1: torch.Tensor, shape (n1,)
      Batch vector for graph 1
    batch2: torch.Tensor, shape (n2,)
      Batch vector for graph 2
    mask1: torch.Tensor, shape (n1,)
      Mask vector for graph 1
    mask2: torch.Tensor, shape (n2,)
      Mask vector for graph 2
    connectivity_s: torch.Tensor, shape (n1, n1)
      Connectivity matrix for graph 1
    connectivity_t: torch.Tensor, shape (n2, n2)
      Connectivity distance matrix for graph 2

    """
    max_num_nodes1 = P.shape[1]
    max_num_nodes2 = P.shape[2]

    mask1 = mask1.float().unsqueeze(2)
    mask2 = mask2.float().unsqueeze(2)

    C1 = connectivity_s
    C2 = connectivity_t
    C1, mask1_C = to_dense_batch(C1, batch1)
    C2, mask2_C = to_dense_batch(C2, batch2)

    C1 = C1[:, :, :max_num_nodes1]
    C2 = C2[:, :, :max_num_nodes2]

    matrix_mask1 = mask1_C.unsqueeze(2) & mask1_C.unsqueeze(1)
    matrix_mask2 = mask2_C.unsqueeze(2) & mask2_C.unsqueeze(1)
    C1_masked = C1.masked_fill(~matrix_mask1, float("nan"))
    C2_masked = C2.masked_fill(~matrix_mask2, float("nan"))
    mean1 = C1_masked.view(C1.shape[0], -1).nanmean(dim=1)
    mean2 = C2_masked.view(C2.shape[0], -1).nanmean(dim=1)

    mean1 = mean1.view(-1, 1, 1)
    mean2 = mean2.view(-1, 1, 1)

    C1 = C1 / (10 * mean1)
    C2 = C2 / (10 * mean2)

    U1 = C1**2 * torch.bmm(mask1, mask1.mT)
    U2 = C2**2 * torch.bmm(mask2, mask2.mT)
    V1 = C1 * torch.bmm(mask1, mask1.mT)
    V2 = 2 * C2 * torch.bmm(mask2, mask2.mT)

    term1 = U1 @ P @ mask2 @ mask2.mT
    term2 = mask1 @ mask1.mT @ P @ U2.mT
    term3 = -V1 @ P @ V2.mT

    return torch.sum((term1 + term2 + term3) * P, dim=(1, 2))


def w_loss(
    P, x1, x2, batch1=None, batch2=None, mask1=None, mask2=None, normalize_cost=True
):
    """
    Wasserstein loss between graph 1 with n1 nodes and graph 2 with n2 nodes.

    Parameters
    ----------
    P: torch.Tensor, shape (n1, n2)
      Transport plan
    x1: torch.Tensor, shape (n1, d)
      Feature matrix for graph 1
    x2: torch.Tensor, shape (n2, d)
      Feature matrix for graph 2
    batch1: torch.Tensor, shape (n1,)
      Batch vector for graph 1
    batch2: torch.Tensor, shape (n2,)
      Batch vector for graph 2
    mask1: torch.Tensor, shape (n1,)
      Mask vector for graph 1
    mask2: torch.Tensor, shape (n2,)
      Mask vector for graph 2
    normalize_cost: bool
      Whether to normalize the cost matrix

    """
    if batch1 is None and batch2 is None:
        M = ot.dist(x1, x2)
        if normalize_cost:
            M = M / (10 * torch.mean(M))
        return torch.sum(M * P)
    else:
        x1, mask1 = to_dense_batch(x1, batch1)
        x2, mask2 = to_dense_batch(x2, batch2)
        matrix_mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)

        M = torch.cdist(x1, x2) ** 2
        M_masked = M.masked_fill(~matrix_mask, float("nan"))
        mean = M_masked.view(M.shape[0], -1).nanmean(dim=1)
        mean = mean.view(-1, 1, 1)
        if normalize_cost:
            M = M / (10 * mean)

        return torch.sum(M * P, dim=(1, 2))


def kl_div(p, q, mass=False, eps=1e-16, batch=False):
    """
    KL divergence between two distributions p and q.
    Parameters
    ----------
    p: torch.Tensor, shape (n1, n2)
      First distribution
    q: torch.Tensor, shape (n1, n2)
      Second distribution
    mass: bool
      Whether to add the mass term
    eps: float
      Small value to avoid division by zero
    batch: bool
      Whether to compute the KL divergence for each batch
    """
    if not batch:
        value = torch.sum(p * torch.log(p / q + eps))
        if mass:
            value = value + torch.sum(q - p)

    else:
        value = torch.sum(p * torch.log(p / q + eps), dim=1)
        if mass:
            value = value + torch.sum(q - p, dim=1)

    return value


def FUGW(
    P,
    x1,
    x2,
    alphas,
    rhos,
    batch1=None,
    batch2=None,
    mask1=None,
    mask2=None,
    geodesic=False,
    connectivity_s=None,
    connectivity_t=None,
):
    """
    FUGW loss between batched graphs
    Parameters
    ----------
    P: torch.Tensor
      Transport plan
    x1: torch.Tensor
      Feature matrix for graphs 1
    x2: torch.Tensor
      Feature matrix for graphs 2
    C1: torch.Tensor
      Adjacency matrix for graphs 1
    C2: torch.Tensor
      Adjacency matrix for graphs 2
    alphas: torch.Tensor
      Alpha vector for the batch
    rhos: torch.Tensor
      Rho vector for the batch
    batch1: torch.Tensor
      Batch vector for graphs 1
    batch2: torch.Tensor
      Batch vector for graphs 2
    mask1: torch.Tensor
      Mask vector for graphs 1
    mask2: torch.Tensor
      Mask vector for graphs 2
    geodesic: bool
      Whether to use geodesic distance
    connectivity_s: torch.Tensor
      Connectivity matrix for graphs 1
    connectivity_t: torch.Tensor
      Connectivity matrix for graphs 2
    """
    wasserstein_loss = w_loss(
        P, x1, x2, batch1=batch1, batch2=batch2, mask1=mask1, mask2=mask2
    )
    gromov_loss = gw_loss(
        P, batch1, batch2, mask1, mask2, connectivity_s, connectivity_t
    )

    # marginal constraints
    bs, n1, n2 = P.shape

    P_marginal_1 = P @ torch.ones(bs, n2, 1, device=P.device)
    P_marginal_2 = P.transpose(1, 2) @ torch.ones(bs, n1, 1, device=P.device)

    n_valid_nodes1 = torch.sum(mask1, dim=1)
    n_valid_nodes2 = torch.sum(mask2, dim=1)

    matrix_mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)

    p = torch.ones(bs, n1, device=P.device) / n_valid_nodes1.view(bs, 1)
    q = torch.ones(bs, n2, device=P.device) / n_valid_nodes2.view(bs, 1)

    sum_P = torch.sum(P * matrix_mask, dim=(1, 2))
    marginals = (
        2 * sum_P * kl_div(P_marginal_1.squeeze() * mask1, p, mass=False, batch=True)
        + 2 * sum_P * kl_div(P_marginal_2.squeeze() * mask2, q, mass=False, batch=True)
        + (1 - sum_P**2)
        + (1 - sum_P**2)
    )

    return {
        "total": (1 - alphas.squeeze()) * wasserstein_loss
        + alphas.squeeze() * gromov_loss
        + rhos.squeeze() * marginals,
        "wasserstein": wasserstein_loss,
        "gromov": gromov_loss,
        "marginals": marginals,
    }
