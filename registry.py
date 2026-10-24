import torch
import numpy as np

''' Control vector field registry '''

def g_unicycle(x: torch.Tensor) -> torch.Tensor:
    """Unicycle on T^3: g1=(cos 2πx3, sin 2πx3, 0), g2=(0,0,1)."""
    zero = torch.zeros_like(x[2])
    one  = torch.ones_like(x[2])
    g1 = torch.stack([torch.cos(2 * torch.pi * x[2]),
                      torch.sin(2 * torch.pi * x[2]), zero])
    g2 = torch.stack([zero, zero, one])
    return torch.stack([g1, g2])

# (function, d, k)
G_REGISTRY = {
    'unicycle': (g_unicycle, 3, 2),
}


''' Target distribution registry '''

def make_sphere(N: int, d: int = 3) -> torch.Tensor:
    """Uniform samples on the sphere of radius 0.25 centred at (0.5, 0.5, 0.5)."""
    x = np.random.randn(N, d)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = 0.25 * x + 0.5
    return torch.tensor(x, dtype=torch.float32)

def make_circle(N: int) -> torch.Tensor:
    """Uniform samples on the circle in xy-plane of radius 0.25 centred at (0.5, 0.5, 0.5)."""
    x = np.random.randn(N, 2)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = 0.25 * x + 0.5
    x = np.concatenate([x, 0.5 * np.ones((N, 1))], axis=1)
    return torch.tensor(x, dtype=torch.float32)

def make_dirac(N: int) -> torch.Tensor:
    """Point mass at (0.5, 0.5, 0.5)."""
    return 0.5 * torch.ones(N, 3)

TARGET_REGISTRY = {
    'sphere': make_sphere,
    'dirac':  make_dirac,
    'circle': make_circle,
}