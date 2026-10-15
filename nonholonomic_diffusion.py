import copy
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from utils import sliced_wasserstein


class NonholonomicDiffusionTrainer:
    def __init__(self,
                 model: nn.Module,
                 g: Callable[[torch.Tensor], torch.Tensor],
                 d: int = 3,
                 k: int = 2,
                 T: float = 5.0,
                 steps: int = 100,
                 loss_type: str = 'ibp',
                 sigma: float = 0.1,
                 t0: float = 0.01,
                 b0: float = 0.1,
                 b1: float = 10.0,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 per_sample_time: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Model takes in spacetime coordinates (x,t) and outputs k-dimensional vector field
        (frame components of the score) approximating the score function.

        Spatial domain is a torus represented by [0,1)^d.
        g represents the k generating vector fields, mapping a space tensor x of shape (d,) to (k, d).
        g must be defined in such a way that we can call torch.vmap(g).

        Time is discretised uniformly; reparametrisation encoded in the schedule beta(t).

        sigma: DSM noise level.
        per_sample_time: if True each sample in a training batch is diffused to its OWN
            randomly-sampled stopping time, reducing gradient variance substantially.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=learning_rate, weight_decay=weight_decay)
        self.g = torch.vmap(g)
        self.d = d
        self.k = k
        self.steps = steps
        self.loss_type = loss_type
        self.sigma = sigma
        self.per_sample_time = per_sample_time

        self.t = torch.linspace(0.0, 1.0, steps + 1, device=device)
        self.dt = 1.0 / steps
        self.t0 = t0
        self.j_min = int(t0 * steps)

        def beta_(t):
            return b0 + t * (b1 - b0)
        self.beta = beta_

        self.history: Dict = {
            'loss': [],
            'sliced_w2': [],
            'sliced_w2_epoch': [],
        }

        self.best_ckpt = None
        self.min_loss = 1e10
        self.min_epoch = None

    # ── Forward process ───────────────────────────────────────────────────────

    def forward_process(self, x: torch.Tensor, steps) -> torch.Tensor:
        """
        x: (m, d) or (d,) promoted to (1, d).
        steps: int (shared for whole batch) or (m,) long tensor (per-sample).
        """
        x = x.clone()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        m = x.shape[0]

        if isinstance(steps, int):
            steps_vec = torch.full((m,), steps, device=x.device, dtype=torch.long)
        else:
            steps_vec = steps.to(device=x.device, dtype=torch.long)
            if steps_vec.numel() == 1:
                steps_vec = steps_vec.expand(m)

        max_steps = int(steps_vec.max().item()) if steps_vec.numel() > 0 else 0
        for j in range(max_steps):
            active = (steps_vec > j).to(x.dtype).unsqueeze(-1)   # (m, 1)
            noise = torch.randn(x.shape[:-1] + (self.k,), device=x.device)
            dx = (torch.sqrt(self.beta(self.t[j]) * self.dt)
                  * torch.einsum('mkd,mk->md', self.g(x), noise))
            x = x + active * dx
            x = torch.remainder(x, 1)
        return x

    # ── Backward process ──────────────────────────────────────────────────────

    def backward_process(self, x: torch.Tensor, model: nn.Module = None,
                         t0: float = 0.0, gamma: float = 0.0,
                         stride: int = 1) -> torch.Tensor:
        """
        x: (m, d) initial points (typically uniform noise).
        gamma: diffusion parameter (0 = probability flow ODE, 1 = full reverse SDE).
        stride: coarsen integration by taking a step every `stride` grid points; useful
                for cheap evaluation during training (stride=1 = exact).
        """
        if model is None:
            model = self.model

        x = torch.FloatTensor(x).to(self.device)
        j0 = max(int(t0 * self.steps), self.j_min)
        js = list(range(j0, self.steps, stride))[::-1]
        dt_eff = stride * self.dt

        for j in js:
            t = self.t[j]
            u = model(x, t * torch.ones(x.shape[0], device=self.device))  # (m, k)
            if self.loss_type == 'dsm':
                u = u / self.sigma
            gx = self.g(x)                                                  # (m, k, d)
            v = torch.einsum('mkd, mk -> md', gx, u)
            noise = torch.einsum('mkd,mk -> md', gx, torch.randn_like(u))
            x = (x
                 + ((1.0 + gamma) / 2) * v * self.beta(t) * dt_eff
                 + torch.sqrt(gamma * self.beta(t) * dt_eff) * noise)
            x = torch.remainder(x, 1)
        return x

    # ── IBP divergence term ───────────────────────────────────────────────────

    def compute_div(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        u = self.model(x, t)
        gx = self.g(x)
        gu = torch.einsum('mkd, mk -> md', gx, u)
        div = 0
        for i in range(self.d):
            grad_outputs = torch.zeros_like(gu)
            grad_outputs[:, i] = 1
            gradient = torch.autograd.grad(
                outputs=gu, inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True,
            )[0]
            div = div + gradient[:, i]
        return div

    # ── Single batch step ─────────────────────────────────────────────────────

    def train_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (m, d) batch from target distribution."""
        self.model.train()
        m = x.shape[0]

        if self.per_sample_time:
            j = torch.randint(self.j_min, self.steps, (m,), device=self.device)
        else:
            j = torch.randint(self.j_min, self.steps, (1,), device=self.device)
        x = self.forward_process(x, j)
        t = self.t[j]

        if self.loss_type == 'dsm':
            eps = torch.randn_like(x)                              # (m, d)
            y = x + self.sigma * eps
            u_eps = self.model(y, t)                               # (m, k)
            # target s_j ~ -(1/sigma) g_j(y)·eps, shape (m, k)
            score_eps = -torch.einsum('mkd, md -> mk', self.g(y), eps) / self.sigma
            loss = torch.mean(0.5 * torch.norm(u_eps - score_eps, dim=-1) ** 2)

        elif self.loss_type == 'ibp':
            score = self.model(x, t)                               # (m, k)
            L2_score = 0.5 * torch.norm(score, dim=-1) ** 2       # (m,)
            div_score = self.compute_div(x, t)
            loss = torch.mean(L2_score + div_score)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self,
              x_train: torch.Tensor,
              epochs: int = 1000,
              batch_size: int = 64,
              verbose: bool = True,
              eval_every: Optional[int] = None,
              eval_n: int = 200,
              eval_stride: int = 5,
              eval_t0: Optional[float] = None,
              eval_x_ref: Optional[torch.Tensor] = None,
              use_lr_schedule: bool = False,
              on_epoch_end: Optional[Callable] = None) -> None:
        """
        Train for `epochs` epochs.

        eval_every: compute sliced Wasserstein-2 (ambient-Euclidean, cheap) every N epochs
            and store in history. NOT torus-aware -- fine as a training monitor when eval_t0
            is small (samples near target, away from periodic boundary). Use wasserstein_torus
            post-hoc for a definitive torus-correct number.

        on_epoch_end: optional callable(epoch: int, loss: float, history: dict) invoked at
            the end of every epoch, AFTER the sliced-W2 eval (if any). Use this to drive
            external logging (e.g. W&B) without importing it here.
        """
        x_train = torch.FloatTensor(x_train).to(self.device)
        dataset = TensorDataset(x_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if eval_x_ref is None:
            eval_x_ref = x_train
        if eval_t0 is None:
            eval_t0 = self.t0

        scheduler = None
        if use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                epoch_loss += self.train_batch(batch[0])
                n_batches += 1
            epoch_loss = epoch_loss / n_batches

            self.history['loss'].append(epoch_loss.detach().cpu().numpy())
            if epoch_loss < self.min_loss:
                self.best_ckpt = copy.deepcopy(self.model)
                self.min_loss = float(epoch_loss)
                self.min_epoch = epoch

            if scheduler is not None:
                scheduler.step()

            if eval_every and epoch % eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    x_init = torch.rand(eval_n, self.d, device=self.device)
                    x_gen = self.backward_process(x_init, t0=eval_t0,
                                                  gamma=0.0, stride=eval_stride)
                    idx = torch.randperm(eval_x_ref.shape[0],
                                         device=eval_x_ref.device)[:eval_n]
                    sw = sliced_wasserstein(x_gen, eval_x_ref[idx], p=2)
                self.history['sliced_w2'].append(sw.item())
                self.history['sliced_w2_epoch'].append(epoch)
                self.model.train()

            # External logging hook -- called after eval so history is up to date
            if on_epoch_end is not None:
                on_epoch_end(epoch, float(epoch_loss), self.history)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {float(epoch_loss):.4f}")
                if eval_every and epoch % eval_every == 0 and self.history['sliced_w2']:
                    print(f"  [eval] sliced W2 ≈ {self.history['sliced_w2'][-1]:.4f}")

        print("Training complete.")
