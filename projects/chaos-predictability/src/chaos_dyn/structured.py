"""Step models whose form matches the equations (hypothesis H9).

All three expose the StepMLP interface: ``model(s, cond) -> s_next``.

PolyStep         delta = linear map of all monomials of the normalised state up to ``degree``.
                 Fitted in closed form by least squares.
QuadStep         small network of multiplicative units, h = (A x) * (B x) + C x. One layer is
                 a general quadratic map, two layers reach degree 4.
VectorFieldStep  a quadratic *vector field* g(s) advanced one dt by RK4. With the same number
                 of substeps as the data generator this family contains the exact map.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from chaos_dyn.models import StepMLP


def exponents(d: int, degree: int) -> list[tuple[int, ...]]:
    """Index tuples of every monomial of total degree 1..degree in d variables."""
    out: list[tuple[int, ...]] = []
    for p in range(1, degree + 1):
        out += list(itertools.combinations_with_replacement(range(d), p))
    return out


class Monomials(nn.Module):
    def __init__(self, d: int, degree: int):
        super().__init__()
        self.idx = exponents(d, degree)
        self.n_out = len(self.idx) + 1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cols = [torch.ones_like(z[..., 0])]
        for t in self.idx:
            c = z[..., t[0]]
            for j in t[1:]:
                c = c * z[..., j]
            cols.append(c)
        return torch.stack(cols, -1)


class PolyStep(StepMLP):
    def __init__(self, trajs: np.ndarray, cond: np.ndarray, degree: int = 2):
        super().__init__(trajs, cond, 4, 1)
        feats = Monomials(trajs.shape[-1], degree)
        self.net = nn.Sequential(feats, nn.Linear(feats.n_out, trajs.shape[-1], bias=False))

    def fit_lstsq(self, s: np.ndarray, s_next: np.ndarray) -> None:
        """Closed-form fit in float64 on pairs (s, s_next)."""
        self.double()
        with torch.no_grad():
            z = (torch.as_tensor(s) - self.s_mu) / self.s_sd
            y = (torch.as_tensor(s_next - s) - self.d_mu) / self.d_sd
            w = torch.linalg.lstsq(self.net[0](z), y).solution
            self.net[1].weight.copy_(w.T)


class QuadLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.a, self.b, self.c = (
            nn.Linear(d_in, d_out),
            nn.Linear(d_in, d_out),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) * self.b(x) + self.c(x)


class QuadStep(StepMLP):
    def __init__(self, trajs: np.ndarray, cond: np.ndarray, hidden: int = 32, depth: int = 1):
        super().__init__(trajs, cond, 4, 1)
        d = trajs.shape[-1]
        layers: list[nn.Module] = []
        d_in = d
        for _ in range(depth):
            layers.append(QuadLayer(d_in, hidden))
            d_in = hidden
        layers.append(nn.Linear(d_in, d))
        self.net = nn.Sequential(*layers)


class VectorFieldStep(nn.Module):
    """s_next = RK4 over dt (in ``substeps`` equal substeps) of ds/dt = W . monomials_2(z)."""

    def __init__(self, trajs: np.ndarray, dt: float, substeps: int = 1, degree: int = 2):
        super().__init__()
        d = trajs.shape[-1]
        self.dt, self.substeps = dt, substeps
        self.register_buffer("s_mu", torch.tensor(trajs.mean((0, 1))))
        self.register_buffer("s_sd", torch.tensor(trajs.std((0, 1))))
        self.register_buffer("d_sd", torch.tensor(np.diff(trajs, axis=1).std((0, 1))))
        self.feats = Monomials(d, degree)
        self.lin = nn.Linear(self.feats.n_out, d, bias=False)

    def step(self, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """One dt of RK4 with coefficient matrix w (d, n_monomials)."""

        def g(x):
            return self.feats((x - self.s_mu) / self.s_sd) @ w.T

        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = g(s)
            k2 = g(s + 0.5 * h * k1)
            k3 = g(s + 0.5 * h * k2)
            k4 = g(s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s

    def forward(self, s: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.step(s, self.lin.weight)

    def fit(
        self, trajs: np.ndarray, n_pairs: int = 4000, seed: int = 0, max_iter: int = 15
    ) -> float:
        """Float64 fit: least squares on centred differences for a start, then Gauss-Newton on
        the one-step residual (30 unknowns, so the Jacobian is cheap in forward mode)."""
        self.double()
        rng = np.random.default_rng(seed)
        i = rng.integers(0, trajs.shape[0], n_pairs)
        k = rng.integers(1, trajs.shape[1] - 1, n_pairs)
        s, s1 = torch.as_tensor(trajs[i, k]), torch.as_tensor(trajs[i, k + 1])

        def resid(w):
            return ((self.step(s, w) - s1) / self.d_sd).reshape(-1)

        with torch.no_grad():
            v = torch.as_tensor((trajs[i, k + 1] - trajs[i, k - 1]) / (2 * self.dt))
            w = torch.linalg.lstsq(
                self.feats((s - self.s_mu) / self.s_sd), v
            ).solution.T.contiguous()
            best = float((resid(w) ** 2).mean())
        for _ in range(max_iter):
            jac = torch.func.jacfwd(resid)(w).reshape(-1, w.numel())
            with torch.no_grad():
                w_new = w + torch.linalg.lstsq(jac, -resid(w)[:, None]).solution.reshape(w.shape)
                loss = float((resid(w_new) ** 2).mean())
            if not loss < best:
                break
            w, best = w_new, loss
        with torch.no_grad():
            self.lin.weight.copy_(w)
        return best
