#!/usr/bin/env python3
"""
experiment.py — entry point for nonholonomic diffusion experiments.

Usage examples
--------------
# IBP loss, sphere target, default hyperparameters:
python experiment.py --loss-type ibp

# DSM loss, more samples, offline W&B:
python experiment.py --loss-type dsm --N 5000 --epochs 2000 --wandb-offline

# Dirac-delta target, fast smoke test:
python experiment.py --target dirac --N 2000 --epochs 50 --eval-every 10

Results are written to output/<run-name>/ and simultaneously streamed to W&B.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from model import ScoreModel
from nonholonomic_diffusion import NonholonomicDiffusionTrainer
from utils import scatter_3d, visualize_score, sliced_wasserstein, wasserstein_torus


# ── Vector field registry ────────────────────────────────────────────────────
# Each entry maps a --g name to (function, d, k).
# Add new vector fields here; experiment code below is field-agnostic.

def g_unicycle(x: torch.Tensor) -> torch.Tensor:
    """Unicycle on T^3: g1=(cos 2πx3, sin 2πx3, 0), g2=(0,0,1)."""
    zero = torch.zeros_like(x[2])
    one  = torch.ones_like(x[2])
    g1 = torch.stack([torch.cos(2 * torch.pi * x[2]),
                      torch.sin(2 * torch.pi * x[2]), zero])
    g2 = torch.stack([zero, zero, one])
    return torch.stack([g1, g2])


G_REGISTRY = {
    'unicycle': (g_unicycle, 3, 2),
    # 'my_new_field': (g_my_new_field, d, k),
}


# ── Target distribution registry ─────────────────────────────────────────────

def make_sphere(N: int, **_) -> torch.Tensor:
    """Uniform samples on the sphere of radius 0.25 centred at (0.5, 0.5, 0.5)."""
    x = np.random.randn(N, 3)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = 0.25 * x + 0.5
    return torch.tensor(x, dtype=torch.float32)

def make_circle(N: int, **_) -> torch.Tensor:
    """Uniform samples on the circle in xy-plane of radius 0.25 centred at (0.5, 0.5, 0)."""
    x = np.random.randn(N, 2)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = 0.25 * x + 0.5
    x = np.concatenate([x, 0.5 * np.ones((N, 1))], axis=1)
    return torch.tensor(x, dtype=torch.float32)

def make_dirac(N: int, **_) -> torch.Tensor:
    """Point mass at (0.5, 0.5, 0.5)."""
    return 0.5 * torch.ones(N, 3)

def make_uniform(N: int, **_) -> torch.Tensor:
    """Uniform distribution on the torus [0,1)^3."""
    return torch.rand(N, 3)

TARGET_REGISTRY = {
    'sphere': make_sphere,
    'dirac':  make_dirac,
    'uniform': make_uniform,
    'circle': make_circle,
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate a nonholonomic score diffusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core experiment parameters (explicitly requested)
    p.add_argument('--N',          type=int,   default=5000,
                   help="Number of training samples drawn from the target distribution.")
    p.add_argument('--epochs',     type=int,   default=1000,
                   help="Number of training epochs.")
    p.add_argument('--T',          type=float, default=5.0,
                   help="Forward-process horizon.")
    p.add_argument('--kappa',      type=float, default=0.01,
                   help="Maximum time step.")
    p.add_argument('--schedule',   type=str,   default='exponential',
                   choices=['constant', 'linear', 'exponential'],
                   help="Step-size schedule near t=0: constant, linear ramp in step index, "
                        "or exponential decay.")
    p.add_argument('--loss-weight', type=str,   default='uniform',
                   choices=['uniform', 'time-weighted'],
                   help="Per-sample loss weight over time: uniform in time (Riemann sum), "
                        " or t dt (Song et al.).")
    p.add_argument('--g',          type=str,   default='unicycle',
                   choices=list(G_REGISTRY),
                   help="Vector field controlling the nonholonomic constraint.")
    p.add_argument('--loss-type',  type=str,   default='ibp',
                   choices=['ibp', 'dsm'],
                   help="Score matching objective: IBP (integration by parts) or DSM (denoising).")
    p.add_argument('--sigma',      type=float, default=0.1,
                   help="DSM noise standard deviation (ignored for IBP).")

    # Target distribution
    p.add_argument('--target',     type=str,   default='sphere',
                   choices=list(TARGET_REGISTRY),
                   help="Target distribution to learn.")

    # Training hyperparameters
    p.add_argument('--lr',         type=float, default=1e-3,  help="Adam learning rate.")
    p.add_argument('--batch-size', type=int,   default=64,    help="Minibatch size.")
    p.add_argument('--t0',         type=float, default=1e-4,
                   help="Early-stopping time. t is the accumulated noise variance, so "
                        "samples are blurred by std ~sqrt(t0) per driven coordinate.")
    p.add_argument('--no-log-time', action='store_true',
                   help="Embed t instead of log(t) in the score model.")
    p.add_argument('--no-lr-schedule', action='store_true',
                   help="Disable cosine annealing LR schedule.")

    # Evaluation
    p.add_argument('--eval-every',  type=int,   default=0,
                   help="Compute sliced W2 every N epochs during training (0 = disable).")
    p.add_argument('--eval-n',      type=int,   default=200,
                   help="Number of backward-process samples used per training eval.")
    p.add_argument('--eval-stride', type=int,   default=5,
                   help="Integration stride for cheap training-time backward process.")
    p.add_argument('--ot-method',   type=str,   default='exact',
                   choices=['exact', 'sinkhorn'],
                   help="OT solver for final torus-aware W2 evaluation. "
                        "'exact' uses POT network-simplex LP (precise, O(n^3) worst case); "
                        "'sinkhorn' is faster with a small controllable bias.")

    # W&B / output
    p.add_argument('--verbose', action='store_true',
                   help="Print training progress to console (in addition to W&B).")
    p.add_argument('--wandb-project', type=str, default='nonholonomic-diffusion',
                   help="W&B project name.")
    p.add_argument('--wandb-offline', action='store_true',
                   help="Run W&B in offline mode (no network required).")
    p.add_argument('--run-name',   type=str,   default=None,
                   help="Override the auto-generated run name.")

    return p.parse_args()


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    """Timestamped logger that writes to stdout AND a file simultaneously."""
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Run name and output directory ─────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or (
        f"{ts}_{args.g}_{args.loss_type}_{args.target}_N{args.N}_e{args.epochs}_t0={args.t0}"
    )
    out_dir = Path('output') / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir / 'run.log')
    log.info(f"Run name : {run_name}")
    log.info(f"Output   : {out_dir.resolve()}")
    log.info(f"Args     : {vars(args)}")

    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    wb_run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        dir=str(out_dir),
    )
    log.info(f"W&B URL  : {wb_run.url}")

    # ── Vector field + dimensions ─────────────────────────────────────────────
    g_fn, d, k = G_REGISTRY[args.g]
    log.info(f"Vector field: {args.g}  (d={d}, k={k})")
    wandb.config.update({'d': d, 'k': k}, allow_val_change=True)

    # ── Target distribution ───────────────────────────────────────────────────
    x0 = TARGET_REGISTRY[args.target](N=args.N)
    log.info(f"Target: {args.target}  ({args.N} samples, shape={tuple(x0.shape)})")

    # ── Model ────────────────────────────────────────────────────────────────
    model = ScoreModel(
        input_dim=d,
        output_dim=k,
        log_time=not args.no_log_time,
        hidden_dims=[64, 32, 32, 16],
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")
    wandb.config.update({'n_params': n_params}, allow_val_change=True)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = NonholonomicDiffusionTrainer(
        model, g_fn,
        d=d, k=k,
        T=args.T,
        kappa=args.kappa,
        schedule=args.schedule,
        t0=args.t0,
        loss_weight=args.loss_weight,
        sigma=args.sigma,
        loss_type=args.loss_type,
        learning_rate=args.lr,
        per_sample_time=True,
    )
    log.info(f"Time grid: {args.schedule}, {trainer.steps} steps, "
             f"T={args.T}, kappa={args.kappa}, t0={args.t0} (index {trainer.j0})")
    t_stop = float(trainer.t[trainer.j0])
    if t_stop > 1.5 * args.t0:
        log.warning(f"t0={args.t0} is not resolved by the '{args.schedule}' grid; "
                    f"the backward process stops at t={t_stop:.2e}. Use a smaller "
                    f"--kappa or the linear/exponential schedule.")
    wandb.config.update({'steps': trainer.steps}, allow_val_change=True)

    # ── Epoch callback: W&B logging + timestamped console lines ───────────────
    def on_epoch_end(epoch: int, loss: float, history: dict) -> None:
        metrics = {'train/loss': loss, 'epoch': epoch}
        # sliced W2 is appended to history before this callback is called
        sw2_logged_this_epoch = (
            history['sliced_w2_epoch'] and history['sliced_w2_epoch'][-1] == epoch
        )
        if sw2_logged_this_epoch:
            metrics['eval/sliced_w2'] = history['sliced_w2'][-1]
        wandb.log(metrics, step=epoch)

        # Console: every 10 epochs, or whenever a W2 eval fired
        if epoch % 10 == 0 or sw2_logged_this_epoch:
            sw_str = (f"  SW2={history['sliced_w2'][-1]:.4f}"
                      if sw2_logged_this_epoch else "")
            log.info(f"Epoch {epoch:>5}/{args.epochs}  loss={loss:+.4f}{sw_str}")

    # ── Training ──────────────────────────────────────────────────────────────
    log.info("Training started.")
    t_train_start = time.time()

    trainer.train(
        x0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,                   
        eval_every=args.eval_every or None,
        eval_n=args.eval_n,
        eval_stride=args.eval_stride,
        use_lr_schedule=not args.no_lr_schedule,
        on_epoch_end=on_epoch_end,
    )

    t_train_elapsed = time.time() - t_train_start
    log.info(
        f"Training complete.  "
        f"elapsed={t_train_elapsed/60:.1f} min  "
        f"best_epoch={trainer.min_epoch}  "
        f"best_loss={trainer.min_loss:.4f}"
    )
    wandb.summary['best_epoch'] = trainer.min_epoch
    wandb.summary['best_loss']  = float(trainer.min_loss)
    wandb.summary['train_time_min'] = round(t_train_elapsed / 60, 2)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    ckpt_path = out_dir / 'best_model.pt'
    torch.save(trainer.best_ckpt.state_dict(), ckpt_path)
    log.info(f"Checkpoint saved -> {ckpt_path}")

    # ── Training curves plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(trainer.history['loss'])
    axes[0].set_title(f"Loss ({args.loss_type.upper()})")
    axes[0].set_xlabel('epoch')

    if trainer.history['sliced_w2']:
        axes[1].plot(trainer.history['sliced_w2_epoch'],
                     trainer.history['sliced_w2'], marker='o', markersize=3)
        axes[1].set_title("Sliced W2 vs target (training monitor, ambient-Euclidean)")
        axes[1].set_xlabel('epoch')
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    curves_path = out_dir / 'training_curves.png'
    fig.savefig(curves_path, dpi=120)
    plt.close(fig)
    wandb.log({'charts/training_curves': wandb.Image(str(curves_path))})
    log.info(f"Training curves saved -> {curves_path}")

    # ── Backward process sweep + torus W2 ────────────────────────────────────
    # Evaluate Wasserstein distance between target distribution and backward-process samples at t=t0.
    # Log this evaluation for both deterministic (gamma=0) and stochastic (gamma=1) backward processes.
    log.info(f"Running backward process sweep (OT method: {args.ot_method})...")
    final_metrics: dict = {}

    for gamma in [0.0, 1.0]:
        xT = torch.rand(2000, d)
        x_gen = trainer.backward_process(
            xT, model=trainer.best_ckpt, gamma=gamma
        ).detach()

        w2 = wasserstein_torus(x_gen, x0, p=2, method=args.ot_method)
        final_metrics[f'eval_gamma={gamma}/w2_torus'] = w2
        log.info(f"gamma={gamma} | W2_torus ({args.ot_method}) = {w2:.4f}")

        fig_scatter = scatter_3d(x_gen, title=f"gamma={gamma} | W2 (torus) ≈ {w2:.4f}")
        scatter_path = out_dir / f'backward_gamma={gamma}.png'
        fig_scatter.savefig(scatter_path, dpi=120)
        plt.close(fig_scatter)
        wandb.log({f'backward/gamma={gamma}': wandb.Image(str(scatter_path))})

    wandb.log(final_metrics)
    wandb.summary.update(final_metrics)

    # ── Target distribution reference plot ────────────────────────────────────
    fig_target = scatter_3d(x0, title=f"Target distribution ({args.target})")
    target_path = out_dir / 'target_distribution.png'
    fig_target.savefig(target_path, dpi=120)
    plt.close(fig_target)
    wandb.log({'charts/target_distribution': wandb.Image(str(target_path))})

    # ── Persist results as JSON ───────────────────────────────────────────────
    results = {
        'run_name':      run_name,
        'args':          vars(args),
        'best_epoch':    trainer.min_epoch,
        'best_loss':     float(trainer.min_loss),
        'train_time_min': round(t_train_elapsed / 60, 2),
        'final_w2':      final_metrics,
    }
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved -> {results_path}")

    wandb.finish()
    log.info("Done.")


if __name__ == '__main__':
    main()
