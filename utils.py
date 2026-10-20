import numpy as np
import torch
import torch.nn as nn
import ot

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d


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


def animate_backward(trainer, path: str = 'backward.gif', n: int = 400,
                     model: nn.Module = None, gamma: float = 0.0,
                     view: str = 'plane', n_frames: int = 120, fps: int = 20,
                     hold_seconds: float = 2.0, dpi: int = 80,
                     seed: int = 0) -> str:
    """
    Save a GIF of the backward process of `trainer` (a NonholonomicDiffusionTrainer with
    d = 3), run from uniform noise at t = T down to t0. Returns `path`.

    view: 'plane' -- top-down (x1, x2) view, each sample drawn as an arrow along its heading
                     angle 2*pi*x3 (unicycle convention), coloured by heading.
          '3d'    -- 3-D scatter of (x1, x2, x3) with the camera slowly orbiting.
          'both'  -- the two views side by side.
    Frames are spaced evenly in log t, so the fast changes near t0 are not skipped. The
    final frame is held for `hold_seconds` before the GIF loops.
    """
    if trainer.d != 3:
        raise ValueError(f"animate_backward needs d = 3, got d = {trainer.d}")
    if view not in ('plane', '3d', 'both'):
        raise ValueError(f"Unknown view: {view!r}")
    if model is None:
        model = trainer.best_ckpt if trainer.best_ckpt is not None else trainer.model

    torch.manual_seed(seed)
    with torch.no_grad():
        x_init = torch.rand(n, 3, device=trainer.device)
        _, ts, xs = trainer.backward_process(x_init, model=model, gamma=gamma,
                                             return_trajectory=True)
    ts = np.array(ts)

    # Trajectory index nearest to each of n_frames log-spaced times (ts is decreasing).
    targets = np.exp(np.linspace(np.log(ts[0]), np.log(ts[-1]), n_frames))
    idx = np.unique(np.searchsorted(-ts, -targets).clip(0, len(ts) - 1))
    idx = np.append(idx, [idx[-1]] * int(hold_seconds * fps))
    frames = [xs[i].numpy() for i in idx]

    panels = {'plane': ['plane'], '3d': ['3d'], 'both': ['plane', '3d']}[view]
    fig = plt.figure(figsize=(5 * len(panels), 5.4))
    axes = {}
    for i, kind in enumerate(panels):
        if kind == 'plane':
            ax = fig.add_subplot(1, len(panels), i + 1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        else:
            ax = fig.add_subplot(1, len(panels), i + 1, projection='3d')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
            ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$x_3$')
        axes[kind] = ax

    x = frames[0]
    artists = {}
    if 'plane' in axes:
        heading = 2 * np.pi * x[:, 2]
        artists['plane'] = axes['plane'].quiver(
            x[:, 0], x[:, 1], np.cos(heading), np.sin(heading), x[:, 2],
            cmap='hsv', clim=(0, 1), angles='xy', scale_units='xy', scale=28,
            width=0.005, headwidth=3.5, headlength=4, headaxislength=3.5)
    if '3d' in axes:
        artists['3d'] = axes['3d'].scatter(x[:, 0], x[:, 1], x[:, 2], s=4, alpha=0.8,
                                           c=x[:, 2], cmap='hsv', vmin=0, vmax=1)
    title = fig.suptitle('')
    fig.tight_layout()

    def update(f):
        x = frames[f]
        if 'plane' in artists:
            heading = 2 * np.pi * x[:, 2]
            artists['plane'].set_offsets(x[:, :2])
            artists['plane'].set_UVC(np.cos(heading), np.sin(heading), x[:, 2])
        if '3d' in artists:
            artists['3d']._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
            artists['3d'].set_array(x[:, 2])
            axes['3d'].view_init(elev=20, azim=45 + 90 * f / len(frames))
        title.set_text(f"backward process   t = {ts[idx[f]]:.2e}")
        return list(artists.values()) + [title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
