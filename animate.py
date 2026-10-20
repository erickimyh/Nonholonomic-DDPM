import json
import matplotlib.pyplot as plt
from matplotlib import path, animation

import numpy as np
import torch
from torch import nn

from model import ScoreModel
from nonholonomic_diffusion import NonholonomicDiffusionTrainer
from run import G_REGISTRY


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


def load_trainer(run_dir: str) -> NonholonomicDiffusionTrainer:
    """Rebuild the trainer for a saved run from its args.json and best_model.pt."""
    with open(f'{run_dir}/args.json') as f:
        a = json.load(f)

    g_fn, d, k = G_REGISTRY[a['g']]
    model = ScoreModel(input_dim=d, output_dim=k,
                       hidden_dims=[64, 32, 32, 16],            # as in run.py
                       log_time=not a.get('no_log_time', False))
    model.load_state_dict(torch.load(f'{run_dir}/best_model.pt', map_location='cpu'))
    model.eval()

    return NonholonomicDiffusionTrainer(
        model, g_fn, d=d, k=k,
        T=a['T'], kappa=a['kappa'], schedule=a['schedule'], t0=a['t0'],
        loss_type=a['loss_type'], sigma=a['sigma'],
        loss_weight=a.get('loss_weight', 'uniform'),
        device='cpu',
    )


if __name__ == '__main__':
    run_dir = 'output/20261020_134621_unicycle_ibp_circle_N5000_e500_t0=0.0001'  
    trainer = load_trainer(run_dir)
    print(trainer.schedule, trainer.steps, 'steps, t0 =', trainer.t0)
    for gamma in [0.0, 1.0]:
        animate_backward(trainer, 
                         f'backward_gamma={gamma}.gif', 
                         gamma = gamma,
                         view = 'plane',
        )