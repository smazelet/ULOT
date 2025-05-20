import torch


class NaNException(Exception):
    pass


class BaseSolver:
    def __init__(
        self,
        nits_bcd=100,
        nits_uot=1000,
        tol_bcd=1e-7,
        tol_uot=None,
        tol_loss=None,
        eval_bcd=1,
        eval_uot=10,
        # ibpp-specific parameters
        ibpp_eps_base=1,
        ibpp_nits_sinkhorn=1,
    ):
        """Init FUGW solver.

        Parameters
        ----------
        nits_bcd: int or None,
            Number of block-coordinate-descent iterations to run.
            If None, run until tol_bcd or tol_loss is reached.
            Default: 10
        nits_uot: int or None,
            Number of solver iteration to run at each BCD iteration
            If None, run until tol_uot is reached.
            Default: 1000
        tol_bcd: float or None,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold. If None, do not stop early.
            Default: None
        tol_uot: float or None,
            Stop the BCD procedure early if the absolute difference
            between two consecutive transport plans
            under this threshold. If None, do not stop early.
            Default: None
        tol_loss: float or None,
            Stop the BCD procedure early if the FUGW loss falls
            under this threshold. If None, do not stop early.
            Default: None
        eval_bcd: int,
            During .fit(), at every eval_bcd step:
            1. compute the FUGW loss and store it in an array
            2. consider stopping early if tol_loss is not None
            3. consider stopping early if tol_bcd is not None
            Default: 1
        eval_uot: int,
            During .fit(), at every eval_uot step:
            1. consider stopping early if tol_uot is not None
            Default: 10
        ibpp_eps_base: int,
            Regularization parameter specific to the ibpp solver.
            Default: 1
        ibpp_nits_sinkhorn: int,
            Number of sinkhorn iterations to run
            within each uot iteration of the ibpp solver.
            Default: 1

        Attributes
        ----------
        Same as parameters.
        """

        if tol_bcd is None and tol_loss is None and nits_bcd is None:
            raise ValueError(
                "At least one of nits_bcd, tol_bcd or tol_loss must be provided."
            )

        if tol_uot is None and nits_uot is None:
            raise ValueError("At least one of nits_uot or tol_uot must be provided.")

        self.nits_bcd = nits_bcd
        self.nits_uot = nits_uot
        self.tol_bcd = tol_bcd
        self.tol_uot = tol_uot
        self.tol_loss = tol_loss
        self.eval_bcd = eval_bcd
        self.eval_uot = eval_uot
        self.ibpp_eps_base = ibpp_eps_base
        self.ibpp_nits_sinkhorn = ibpp_nits_sinkhorn


def solver_sinkhorn(cost, init_duals, uot_params, tuple_weights, train_params):
    """
    Scaling algorithm (ie Sinkhorn algorithm).
    Code adapted from Séjourné et al 2020:
    https://github.com/thibsej/unbalanced_gromov_wasserstein.
    """

    ws, wt, ws_dot_wt = tuple_weights
    log_ws, log_wt = ws.log(), wt.log()
    u, v = init_duals
    rho_s, rho_t, eps = uot_params
    niters, tol, eval_freq = train_params

    tau_s = 1 if torch.isinf(rho_s) else rho_s / (rho_s + eps)
    tau_t = 1 if torch.isinf(rho_t) else rho_t / (rho_t + eps)

    for _ in range(niters):
        u_prev, v_prev = u.detach().clone(), v.detach().clone()
        if rho_t == 0:
            v = torch.zeros_like(v)
        else:
            v = -tau_t * ((u + log_ws)[:, None] - cost / eps).logsumexp(dim=0)

        if rho_s == 0:
            u = torch.zeros_like(u)
        else:
            u = -tau_s * ((v + log_wt)[None, :] - cost / eps).logsumexp(dim=1)

        err = max((u - u_prev).abs().max(), (v - v_prev).abs().max())
        if err < tol:
            break

    pi = ws_dot_wt * (u[:, None] + v[None, :] - cost / eps).exp()

    return (u, v), pi


def solver_mm(cost, init_pi, uot_params, tuple_weights, train_params):
    """Solve (regularized) UOT using the majorization-minimization algorithm.

    Allow epsilon to be 0 but rho_s and rho_t can't be infinity.

    Note that if the parameters are small so that numerically, the exponential
    of negative cost will contain zeros and this serves as sparsification
    of the optimal plan.

    If the parameters are large, then the resulting optimal plan is more dense
    than the one obtained from sinkhorn algorithm.
    But all parameters should not be too small, otherwise the kernel will
    contain too many zeros. Consequently, the optimal plan will contain NaN
    (because the Kronecker sum of two marginals will eventually contain zeros,
    and divided by zero will result in undesirable coupling).
    """
    niters, tol, eval_freq = train_params
    ws, wt = tuple_weights
    rho_s, rho_t, eps = uot_params

    sum_param = rho_s + rho_t + eps
    tau_s = rho_s / sum_param
    tau_t = rho_t / sum_param
    r = eps / sum_param
    K = (
        ws[:, None] ** (tau_s + r)
        * wt[None, :] ** (tau_t + r)
        * (-cost / sum_param).exp()
    )

    pi1, pi2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(niters):
        pi1_prev, pi2_prev = pi1.detach().clone(), pi2.detach().clone()
        pi = pi ** (tau_s + tau_t) * K / (pi1[:, None] ** tau_s * pi2[None, :] ** tau_t)
        pi1, pi2 = pi.sum(1), pi.sum(0)

        if idx % eval_freq == 0:
            err = max((pi1 - pi1_prev).abs().max(), (pi2 - pi2_prev).abs().max())
            if err < tol:
                break

    return pi


def solver_mm_l2(cost, init_pi, uot_params, tuple_weights, train_params, verbose=True):
    """
    Solve regularized UOT with L2-squared norm using
    the majorization-minimization algorithm. Allow epsilon to be 0
    but rho_s and rho_t can't be infinity.

    If $\rho$ is too small, then we obtain $0$ everywhere,
    which will result in NaN coupling.
    If $\rho$ is too large, then we lose the sparsity.
    So, need to choose $\rho$ adequately.
    """

    niters, tol, eval_freq = train_params
    ws, wt, ws_dot_wt = tuple_weights
    rho_s, rho_t, eps = uot_params

    a = rho_s * ws[:, None] + rho_t * wt[None, :] + eps * ws_dot_wt
    thres = torch.clamp(a - cost, min=0)

    if torch.count_nonzero(thres) == 0:
        print("Values for rho and/or eps are too low, plan will be empty.")

    pi1, pi2, pi = init_pi.sum(1), init_pi.sum(0), init_pi

    for idx in range(niters):
        pi1_prev, pi2_prev = pi1.detach().clone(), pi2.detach().clone()

        # Update plan and marginals
        denom = rho_s * pi1[:, None] + rho_t * pi2[None, :] + eps * pi
        pi = thres * pi / denom
        pi1, pi2 = pi.sum(1), pi.sum(0)

        if idx % eval_freq == 0:
            err = max((pi1 - pi1_prev).abs().max(), (pi2 - pi2_prev).abs().max())
            if err < tol:
                break

    return pi


def solver_ibpp(
    cost,
    init_pi,
    init_duals,
    uot_params,
    tuple_weights,
    train_params,
    verbose=True,
):
    niters, nits_sinkhorn, eps_base, tol, eval_freq = train_params
    rho_s, rho_t, eps = uot_params
    ws, wt, ws_dot_wt = tuple_weights
    u, v = init_duals
    m1, pi = init_pi.sum(1), init_pi

    sum_eps = eps_base + eps
    tau_s = 1 if rho_s == float("inf") else rho_s / (rho_s + sum_eps)
    tau_t = 1 if rho_t == float("inf") else rho_t / (rho_t + sum_eps)

    K = torch.exp(-cost / sum_eps)

    for idx in range(niters):
        m1_prev = m1.detach().clone()

        # IPOT
        G = K * pi if (eps_base / sum_eps) == 1 else K * pi ** (eps_base / sum_eps)

        for _ in range(nits_sinkhorn):
            v = (G.T @ (u * ws)) ** (-tau_t) if rho_t != 0 else torch.ones_like(v)
            u = (G @ (v * wt)) ** (-tau_s) if rho_s != 0 else torch.ones_like(u)
        pi = u[:, None] * G * v[None, :]

        m1 = pi.sum(1)
        #     if m1.isnan().any() or m1.isinf().any():
        #         raise ValueError(
        #             "There is NaN in coupling. "
        #             "You may want to increase ibpp_eps_base "
        #             f"(current value: {eps_base})."
        #         )

        if m1.isnan().any() or m1.isinf().any():
            return (u, v), pi

        if idx % eval_freq == 0:
            err = (m1 - m1_prev).abs().max()
            if err < tol:
                break

    # renormalize couplings
    pi = pi * ws_dot_wt

    return (u, v), pi


def compute_unnormalized_kl(p, q):
    """Compute unnormalized Kullback-Leibler divergence between two vectors.

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p

    Returns
    -------
    unnormalized_kl: float"""
    # By convention: 0 log 0 = 0
    entropy = torch.nan_to_num(p * (p / q).log(), nan=0.0, posinf=0.0, neginf=0.0).sum()
    return entropy


def compute_kl(p, q):
    """Compute Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p

    Returns
    -------
    kl: float
    """
    return compute_unnormalized_kl(p, q) - p.sum() + q.sum()


def compute_l2(p, q):
    """Compute L2 distance between two distributions.

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p

    Returns
    -------
    l2: float
    """
    return torch.sum((p - q) ** 2) / 2


def compute_divergence(p, q, divergence="kl"):
    """Compute div(p, q).

    Parameters
    ----------
    p: torch tensor
    q: torch tensor
        Should have the same size as p
    divergence: str
        Either "kl" or "l2".
        If "kl", compute KL(p, q).
        If "l2", compute || p - q ||^2 / 2.
        Default: "kl"

    Returns
    -------
    div: float
    """
    if divergence == "kl":
        return compute_kl(p, q)
    elif divergence == "l2":
        return compute_l2(p, q)


def compute_quad_kl(mu, nu, alpha, beta):
    """
    Calculate the KL divergence between two product measures:
    KL(mu otimes nu, alpha otimes beta) =
    m_mu * KL(nu, beta)
    + m_nu * KL(mu, alpha)
    + (m_mu - m_alpha) * (m_nu - m_beta)

    Parameters
    ----------
    mu: torch tensor
    nu: torch tensor
    alpha: torch tensor
        Should have the same size as mu
    beta: torch tensor
        Should have the same size as nu

    Returns
    ----------
    kl: float
        KL divergence between two product measures
    """

    m_mu = mu.sum()
    m_nu = nu.sum()
    m_alpha = alpha.sum()
    m_beta = beta.sum()
    const = (m_mu - m_alpha) * (m_nu - m_beta)
    kl = m_nu * compute_kl(mu, alpha) + m_mu * compute_kl(nu, beta) + const

    return kl


def compute_quad_l2(a, b, mu, nu):
    """Compute || a otimes b - mu otimes nu ||^2 / 2."""

    norm = (
        (a**2).sum() * (b**2).sum()
        - 2 * (a * mu).sum() * (b * nu).sum()
        + (mu**2).sum() * (nu**2).sum()
    )

    return norm / 2


def compute_quad_divergence(mu, nu, alpha, beta, divergence="kl"):
    """Compute div(mu otimes nu, alpha otimes beta).

    Parameters
    ----------
    mu: torch tensor
    nu: torch tensor
    alpha: torch tensor
        Should have the same size as mu
    beta: torch tensor
        Should have the same size as nu
    divergence: str
        Either "kl" or "l2".
        If "kl", compute KL(mu otimes nu, alpha otimes beta).
        If "l2", compute || mu otimes nu - alpha otimes beta ||^2 / 2.
        Default: "kl"

    Returns
    -------
    div: float
    """
    if divergence == "kl":
        return compute_quad_kl(mu, nu, alpha, beta)
    elif divergence == "l2":
        return compute_quad_l2(mu, nu, alpha, beta)
