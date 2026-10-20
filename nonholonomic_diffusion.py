import copy
import math
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
                 kappa: float = 0.01,
                 schedule: str = 'linear',
                 loss_type: str = 'ibp',
                 sigma: float = 0.1,
                 t0: float = 1e-4,
                 loss_weight: str = 'uniform',
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

        Forward SDE is dX = G(X) dW on [0, T], discretised on a fixed grid self.t with step
        sizes self.dt[j] = t[j+1] - t[j] <= kappa. `schedule` sets how steps shrink near t=0
        (see make_time_grid); self.steps is the resulting number of steps.

        sigma: DSM noise level.
        loss_weight: per-sample weight on the loss at sampled step j (j is uniform over steps).
            'uniform'    : dt[j], i.e. uniform in time: the loss is a Riemann sum in time of
                           the per-time loss.
            'inverse_dt' : 1, i.e. time weighted by 1/dt (emphasises fine-step regions).
            'inverse_t2' : dt[j] / t[j]^2.
            dt-based weights are normalised by the length of the training interval, so
            'uniform' weights average to 1.
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
        self.loss_type = loss_type
        self.sigma = sigma
        self.per_sample_time = per_sample_time
        self.loss_weight = loss_weight

        self.kappa = kappa
        self.schedule = schedule
        self.t = self.make_time_grid(T, kappa, schedule, t0).to(device)   # (steps+1,)
        self.dt = self.t[1:] - self.t[:-1]                                  # (steps,)
        self.steps = len(self.dt)
        self.t0 = t0
        self.j0 = self._index_at(t0)
        self.t0_ = self.t[self.j0].item()  # actual first time

        if loss_weight == 'uniform':
            w = self.dt
        elif loss_weight == 'time-weighted':
            w = self.dt * self.t[:-1]
        self.loss_weight = w / w.mean()

        self.history: Dict = {
            'loss': [],
            'sliced_w2': [],
            'sliced_w2_epoch': [],
        }

        self.best_ckpt = None
        self.min_loss = 1e10
        self.min_epoch = None

    # ── Time grid ─────────────────────────────────────────────────────────────

    @staticmethod
    def make_time_grid(T: float, kappa: float, schedule: str, t0: float) -> torch.Tensor:
        """
        Grid 0 = t[0] < ... < t[steps] = T with every step <= kappa.
          'constant'    : dt = kappa throughout.
          'linear'      : dt grows linearly with step index up to kappa, reached at t ~ 1.
          'exponential' : Benton et al. (2023). Steps <= kappa from 0 to t0, then dt[j] = kappa * t[j]
                          (geometric growth) until t ~ 1.
        All schedules then take steps of at most kappa up to T.
        """
        ts = [0.0]
        if schedule == 'linear':
            n_ramp = max(1, round(2 / kappa))          
            for j in range(n_ramp):
                ts.append(ts[-1] + kappa * (j + 1) / n_ramp)
        elif schedule == 'exponential':
            # reach t0 in steps <= kappa (needs t0 > 0), then grow geometrically
            ts = np.linspace(0.0, t0, math.ceil(t0 / kappa - 1e-6) + 1).tolist()
            while ts[-1] * (1 + kappa) < 1.0:
                ts.append(ts[-1] * (1 + kappa))
        elif schedule != 'constant':
            raise ValueError(f"Unknown schedule: {schedule!r}")
        ts = [t for t in ts if t < T]                  # in case T is inside the ramp
        n_rest = math.ceil((T - ts[-1]) / kappa - 1e-6)
        ts += np.linspace(ts[-1], T, n_rest + 1)[1:].tolist()
        return torch.tensor(ts, dtype=torch.float32)

    def _index_at(self, t: float) -> int:
        """Index of the first grid point >= t."""
        return int(torch.searchsorted(self.t, torch.tensor(t, device=self.t.device)))

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
            dx = torch.sqrt(self.dt[j]) * torch.einsum('mkd,mk->md', self.g(x), noise)
            x = x + active * dx
            x = torch.remainder(x, 1)
        return x

    # ── Backward process ──────────────────────────────────────────────────────

    def backward_process(self, x: torch.Tensor, model: nn.Module = None,
                         t0: float = 0.0, gamma: float = 0.0,
                         stride: int = 1, return_trajectory: bool = False):
        """
        x: (m, d) initial points (typically uniform noise).
        return_trajectory: if True, return (x, ts, xs) where xs[i] is the (m, d) state at
                time ts[i], from the start time T down to the final time, inclusive.
        gamma: diffusion parameter (0 = probability flow ODE, 1 = full reverse SDE).
        stride: coarsen integration by taking a step every `stride` grid points; useful
                for cheap evaluation during training (stride=1 = full grid).
        Each step is forward Euler in reverse time: the model is evaluated at the time
        being stepped from, t[j + stride], and x moves to t[j].
        """
        if model is None:
            model = self.model

        x = torch.FloatTensor(x).to(self.device)
        j0 = max(self._index_at(t0), self.j0)
        js = list(range(j0, self.steps, stride))[::-1]
        ts, xs = [float(self.t[-1])], [x.detach().cpu()]

        for j in js:
            j_next = min(j + stride, self.steps)
            t = self.t[j_next]                      # x is currently at time t[j_next]
            dt = self.t[j_next] - self.t[j]
            u = model(x, t * torch.ones(x.shape[0], device=self.device))  # (m, k)
            if self.loss_type == 'dsm':
                u = u / self.sigma
            gx = self.g(x)                                                  # (m, k, d)
            v = torch.einsum('mkd, mk -> md', gx, u)
            noise = torch.einsum('mkd,mk -> md', gx, torch.randn_like(u))
            x = (x
                 + ((1.0 + gamma) / 2) * v * dt
                 + torch.sqrt(gamma * dt) * noise)
            x = torch.remainder(x, 1)
            if return_trajectory:
                ts.append(float(self.t[j]))
                xs.append(x.detach().cpu())
        if return_trajectory:
            return x, ts, xs
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
            j = torch.randint(self.j0, self.steps, (m,), device=self.device)
        else:
            j = torch.randint(self.j0, self.steps, (1,), device=self.device)
        x = self.forward_process(x, j)
        t = self.t[j]

        if self.loss_type == 'dsm':
            eps = torch.randn_like(x)                              # (m, d)
            y = x + self.sigma * eps
            u_eps = self.model(y, t)                               # (m, k)
            # target s_j ~ -(1/sigma) g_j(y)·eps, shape (m, k)
            score_eps = -torch.einsum('mkd, md -> mk', self.g(y), eps) / self.sigma
            loss = 0.5 * torch.norm(u_eps - score_eps, dim=-1) ** 2   # (m,)

        elif self.loss_type == 'ibp':
            score = self.model(x, t)                               # (m, k)
            L2_score = 0.5 * torch.norm(score, dim=-1) ** 2       # (m,)
            div_score = self.compute_div(x, t)
            loss = L2_score + div_score                            # (m,)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        loss = torch.mean(loss * self.loss_weight[j])

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
