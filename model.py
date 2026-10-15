import math
from typing import List

import torch
import torch.nn as nn


class ScoreModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 2, # d
                 output_dim: int = 2, # d (standard) or k (nonholonomic)
                 hidden_dims: List[int] = [64, 32, 16, 16], 
                 time_emb_dim: int = 16,
        ):
        """
            input_dim: Spatial dimension
            output_dim: Dimension of vector field output (d for standard diffusion, k for nonholonomic diffusion)
            hidden_layers: List of hidden layer sizes
            activations: List of activation functions for each hidden layer
        """
        super(ScoreModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.time_emb_dim = time_emb_dim

        # Layers
        layers = []
        dims = [2 * input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dims[-1], output_dim)
        self.act = nn.SiLU()

        # FiLM projections of time embedding into each hidden layer
        # learns (gamma, delta) to modulate each hidden layer h by (gamma * h + delta)
        self.film = nn.ModuleList([
            nn.Linear(time_emb_dim, 2 * d) for d in hidden_dims
        ])
        
        self._initialize_weights()

    # enforce periodicity
    def _x_embedding(self, x):
        return torch.cat([
            torch.cos(2 * torch.pi * x),
            torch.sin(2 * torch.pi * x)
        ], dim=-1)

    # sinusoidal time embedding
    def _time_embedding(self, t):
        half = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]   # (m, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (m, time_emb_dim)
        return emb

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        x = self._x_embedding(x)
        t_emb = self._time_embedding(t)

        # pass through hidden layers, modulating by projection of time embedding at each hidden layer
        for layer, film in zip(self.layers, self.film):
            x = layer(x)
            gamma, delta = film(t_emb).chunk(2, dim=-1)
            x = gamma * x + delta
            x = self.act(x)
        return self.out(x)
