import numpy as np
import torch
import torch.nn as nn
import ot

import matplotlib.pyplot as plt

# ── Wasserstein distance utilities ───────────────────────────────────────────

def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor,
                       n_projections: int = 64, p: int = 2) -> torch.Tensor:
    """
    Sliced Wasserstein-p distance between two point clouds in ambient R^d.
    NOT torus-aware: for a torus-correct distance use `wasserstein_torus` instead.

    x: (n, d), y: (m, d). If n != m the larger cloud is randomly subsampled to match
    (exact 1-D OT via sort-and-compare requires equal-size samples per projection).
    """
    device = x.device
    n = min(x.shape[0], y.shape[0])
    if x.shape[0] > n:
        x = x[torch.randperm(x.shape[0], device=device)[:n]]
    if y.shape[0] > n:
        y = y[torch.randperm(y.shape[0], device=device)[:n]]

    d = x.shape[-1]
    theta = torch.randn(n_projections, d, device=device)
    theta = theta / theta.norm(dim=-1, keepdim=True)

    x_proj, _ = torch.sort(x @ theta.T, dim=0)   # (n, n_projections)
    y_proj, _ = torch.sort(y @ theta.T, dim=0)
    sw_p = (x_proj - y_proj).abs().pow(p).mean(dim=0)  # per-direction W_p^p
    return sw_p.mean().pow(1.0 / p)


def torus_cost_matrix(x: torch.Tensor, y: torch.Tensor,
                      squared: bool = True) -> torch.Tensor:
    """
    Pairwise cost matrix using the flat-torus geodesic distance on [0,1)^d.
    Each coordinate wraps: d_i = min(|x_i - y_i|, 1 - |x_i - y_i|).

    x: (n, d), y: (m, d)  ->  (n, m)
    squared=True  -> squared distances (feed into emd2/sinkhorn2 for W_2, then sqrt)
    squared=False -> distances directly (use for W_1, no sqrt needed)
    """
    diff = (x.unsqueeze(1) - y.unsqueeze(0)).abs()   # (n, m, d)
    diff = torch.minimum(diff, 1.0 - diff)
    sqdist = (diff ** 2).sum(dim=-1)                  # (n, m)
    return sqdist if squared else sqdist.sqrt()


def wasserstein_torus(x: torch.Tensor, y: torch.Tensor,
                      p: int = 2, method: str = 'exact',
                      sinkhorn_reg: float = 1e-2) -> float:
    """
    Exact (or entropic-regularised) Wasserstein-p distance on the flat torus [0,1)^d,
    using the geodesic (wraparound) ground metric.

    method: 'exact'   -> ot.emd2, network-simplex LP; exact, O(n^3) worst case.
            'sinkhorn' -> ot.sinkhorn2, entropic-regularised; faster, small bias
                          tunable via sinkhorn_reg.
    """
    M = torus_cost_matrix(x, y, squared=(p == 2)).detach().cpu().numpy().astype(np.float64)
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]

    if method == 'exact':
        cost = ot.emd2(a, b, M)
    elif method == 'sinkhorn':
        cost = float(ot.sinkhorn2(a, b, M, reg=sinkhorn_reg))
    else:
        raise ValueError(f"Unknown OT method: {method!r}. Use 'exact' or 'sinkhorn'.")

    return float(np.sqrt(cost)) if p == 2 else float(cost)


# ── Visualisation ─────────────────────────────────────────────────────────────

def scatter_3d(x, title: str = "", elev: int = 20, azim: int = 45):
    """3-D scatter of a (n, 3) tensor or numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=5, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return fig

# ── Visualizations ────────────────────────────────────────────────

def visualize_score(trainer, t: float = 0.1, n: int = 5,
                    model: nn.Module = None):
    """Quiver plot of the learned vector field G(x)u(x,t) on an n^3 grid."""
    if model is None:
        model = trainer.best_ckpt if trainer.best_ckpt is not None else trainer.model
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.linspace(0.0, 1.0, n)
    grid = torch.tensor(
        [[xs[i], xs[j], xs[k]]
         for i in range(n) for j in range(n) for k in range(n)],
        dtype=torch.float32,
        device=trainer.device,
    )
    u_grid = model(grid, t * torch.ones(n ** 3, device=trainer.device))
    g_grid = trainer.g(grid)                                    # (m, k, d)
    v_grid = torch.einsum('mkd, mk -> md', g_grid, u_grid)
    grid   = grid.detach().cpu().numpy()
    v_grid = v_grid.detach().cpu().numpy()
    ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2],
              v_grid[:, 0], v_grid[:, 1], v_grid[:, 2], length=0.2)
    ax.set_title(f"Score function t={t}")
    return fig