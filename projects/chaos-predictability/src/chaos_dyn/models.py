"""Two learned predictors for s(t_k), k = 1..N, given s(0) (and per-trajectory constants c).

StepMLP   (autoregressive): s_{k+1} = s_k + g_theta(s_k, c); rolled out N times.
DirectMLP (direct):         [s_1 .. s_N] = s_0 + h_theta(s_0, c); one forward pass.

Both normalise inputs and outputs with statistics fixed from the training set.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sine": Sine,  # plain sin with default Linear init (inputs are normalised); not SIREN init
}


def mlp(d_in: int, d_out: int, hidden: int, depth: int, activation: str = "gelu") -> nn.Sequential:
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _safe_std(x: np.ndarray, axis) -> np.ndarray:
    sd = x.std(axis=axis)
    return np.where(sd < 1e-8, 1.0, sd)


class StepMLP(nn.Module):
    def __init__(
        self,
        trajs: np.ndarray,
        cond: np.ndarray,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "gelu",
    ):
        super().__init__()
        d, c = trajs.shape[-1], cond.shape[-1]
        delta = np.diff(trajs, axis=1)
        self.register_buffer("s_mu", torch.tensor(trajs.mean((0, 1)), dtype=torch.float32))
        self.register_buffer("s_sd", torch.tensor(_safe_std(trajs, (0, 1)), dtype=torch.float32))
        self.register_buffer("d_mu", torch.tensor(delta.mean((0, 1)), dtype=torch.float32))
        self.register_buffer("d_sd", torch.tensor(_safe_std(delta, (0, 1)), dtype=torch.float32))
        c_mu = cond.mean(0) if c else np.zeros(0)
        c_sd = _safe_std(cond, 0) if c else np.ones(0)
        self.register_buffer("c_mu", torch.tensor(c_mu, dtype=torch.float32))
        self.register_buffer("c_sd", torch.tensor(c_sd, dtype=torch.float32))
        self.net = mlp(d + c, d, hidden, depth, activation)

    def delta_normalised(self, s: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([(s - self.s_mu) / self.s_sd, (cond - self.c_mu) / self.c_sd], -1)
        return self.net(x)

    def forward(self, s: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return s + self.delta_normalised(s, cond) * self.d_sd + self.d_mu

    def rollout(self, s0: torch.Tensor, cond: torch.Tensor, n_steps: int) -> torch.Tensor:
        out = [s0]
        s = s0
        for _ in range(n_steps):
            s = self(s, cond)
            out.append(s)
        return torch.stack(out, 1)


class DirectMLP(nn.Module):
    def __init__(
        self,
        trajs: np.ndarray,
        cond: np.ndarray,
        hidden: int = 512,
        depth: int = 3,
        activation: str = "gelu",
    ):
        super().__init__()
        n1, d = trajs.shape[1], trajs.shape[2]
        c = cond.shape[-1]
        self.n_steps = n1 - 1
        s0 = trajs[:, 0]
        disp = trajs[:, 1:] - s0[:, None]  # (M, N, d): normalised per time step
        self.register_buffer("s_mu", torch.tensor(s0.mean(0), dtype=torch.float32))
        self.register_buffer("s_sd", torch.tensor(_safe_std(s0, 0), dtype=torch.float32))
        self.register_buffer("o_mu", torch.tensor(disp.mean(0), dtype=torch.float32))
        self.register_buffer("o_sd", torch.tensor(_safe_std(disp, 0), dtype=torch.float32))
        c_mu = cond.mean(0) if c else np.zeros(0)
        c_sd = _safe_std(cond, 0) if c else np.ones(0)
        self.register_buffer("c_mu", torch.tensor(c_mu, dtype=torch.float32))
        self.register_buffer("c_sd", torch.tensor(c_sd, dtype=torch.float32))
        self.net = mlp(d + c, self.n_steps * d, hidden, depth, activation)

    def disp_normalised(self, s0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([(s0 - self.s_mu) / self.s_sd, (cond - self.c_mu) / self.c_sd], -1)
        return self.net(x).view(len(s0), self.n_steps, -1)

    def forward(self, s0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        disp = self.disp_normalised(s0, cond) * self.o_sd + self.o_mu
        return torch.cat([s0[:, None], s0[:, None] + disp], 1)

    def rollout(self, s0: torch.Tensor, cond: torch.Tensor, n_steps: int) -> torch.Tensor:
        return self(s0, cond)[:, : n_steps + 1]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
