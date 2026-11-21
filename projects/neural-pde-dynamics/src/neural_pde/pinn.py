"""Neural PDE solvers with hard-constrained trial functions (Lagaris et al. 1998).

The network never has to *learn* the boundary or initial conditions. They are built into the
trial function, so the loss is just the PDE residual at collocation points.

A. Laplace on the unit square (port of the original AMATH 581 project). Eqs. 6-8 in notes/derivation.typ.
B. Viscous Burgers on the periodic domain [0, 2 pi) as a PINN. Eq. 9.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64


def mlp(d_in: int, d_out: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers).to(DTYPE)


def _grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]


# ------------------------------------------------------------------------------ Laplace


def laplace_exact(x, y):
    """u = sin(pi x) sinh(pi y) / sinh(pi). Eq. 7. Works on numpy arrays or torch tensors."""
    lib = torch if isinstance(x, torch.Tensor) else np
    return lib.sin(np.pi * x) * lib.sinh(np.pi * y) / np.sinh(np.pi)


class LaplaceTrial(nn.Module):
    """psi(x, y) = y sin(pi x) + x (1 - x) y (1 - y) N(x, y). Implements eq. 8.

    psi = 0 on x = 0, x = 1, y = 0 and psi = sin(pi x) on y = 1 for *any* network N.
    """

    def __init__(self, hidden: int = 32, depth: int = 2):
        super().__init__()
        self.net = mlp(2, 1, hidden, depth)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :1], xy[:, 1:]
        return y * torch.sin(np.pi * x) + x * (1 - x) * y * (1 - y) * self.net(xy)


def laplace_residual(model: LaplaceTrial, xy: torch.Tensor) -> torch.Tensor:
    """psi_xx + psi_yy at the points xy (eq. 6 with f = 0)."""
    xy = xy.clone().requires_grad_(True)
    psi = model(xy)
    g = _grad(psi, xy)
    psi_xx = _grad(g[:, :1], xy)[:, :1]
    psi_yy = _grad(g[:, 1:], xy)[:, 1:]
    return psi_xx + psi_yy


def grid(n: int, interior: bool = True) -> torch.Tensor:
    s = np.linspace(0, 1, n + 2)[1:-1] if interior else np.linspace(0, 1, n)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=DTYPE)


def train_laplace(
    model: LaplaceTrial,
    n_colloc: int = 24,
    adam_iters: int = 2000,
    lbfgs_iters: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Minimise mean(residual^2) on an interior grid: Adam to get close, then L-BFGS to polish.

    (The original version applied the learning rate twice, W -= lr * (lr * grad), for 10
    iterations, an effective step of 1e-6 per iteration. That is why it never converged.)
    """
    torch.manual_seed(seed)
    xy = grid(n_colloc)
    hist: list[float] = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters)
    t0 = time.time()
    for _ in range(adam_iters):
        loss = laplace_residual(model, xy).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
    if lbfgs_iters:
        _lbfgs(model, lambda: laplace_residual(model, xy).pow(2).mean(), lbfgs_iters, hist)
    if verbose:
        print(f"laplace: final loss {hist[-1]:.2e} in {time.time() - t0:.0f}s")
    return {"loss": hist}


def laplace_fd_solve(n: int) -> tuple[np.ndarray, np.ndarray]:
    """5-point finite differences on an (n+2)^2 grid, solved directly. Returns (s, U), U[i, j] = u(s_i, s_j).

    Replaces the original Jacobi iteration (same discrete answer, no iteration-count tuning).
    """
    s = np.linspace(0, 1, n + 2)
    h = s[1] - s[0]
    T = (np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / h**2
    eye = np.eye(n)
    A = np.kron(T, eye) + np.kron(eye, T)  # unknown ordering: index = i * n + j (x-major)
    b = np.zeros((n, n))
    b[:, -1] -= np.sin(np.pi * s[1:-1]) / h**2  # boundary y = 1 enters the last y-column
    U = np.zeros((n + 2, n + 2))
    U[:, -1] = np.sin(np.pi * s)
    U[1:-1, 1:-1] = np.linalg.solve(A, b.ravel()).reshape(n, n)
    return s, U


# ------------------------------------------------------------------------------ Burgers


class BurgersPINN(nn.Module):
    """u(x, t) = u0(x) + t N(sin x, cos x, t / t_final). Implements eq. 9.

    The sin/cos features make u exactly 2*pi-periodic in x, and the t factor makes
    u(x, 0) = u0(x) exactly. Only the PDE residual is left to learn.
    """

    def __init__(
        self,
        u0: Callable[[torch.Tensor], torch.Tensor],
        t_final: float,
        hidden: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.u0 = u0
        self.t_final = t_final
        self.net = mlp(3, 1, hidden, depth)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([torch.sin(x), torch.cos(x), t / self.t_final], -1)
        return self.u0(x) + t * self.net(feats)


def burgers_residual(model: BurgersPINN, x: torch.Tensor, t: torch.Tensor, nu: float):
    """u_t + u u_x - nu u_xx (eq. 1)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t + u * u_x - nu * u_xx


def train_burgers(
    model: BurgersPINN,
    nu: float,
    n_colloc: int = 4096,
    adam_iters: int = 5000,
    lbfgs_iters: int = 500,
    lr: float = 2e-3,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Adam on freshly resampled random collocation points each step, then L-BFGS on a fixed set."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    def sample(n):
        x = 2 * np.pi * torch.rand(n, 1, generator=gen, dtype=DTYPE)
        t = model.t_final * torch.rand(n, 1, generator=gen, dtype=DTYPE)
        return x, t

    hist: list[float] = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters)
    t0 = time.time()
    for it in range(adam_iters):
        x, t = sample(n_colloc)
        loss = burgers_residual(model, x, t, nu).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
        if verbose and (it + 1) % 1000 == 0:
            print(f"  burgers adam {it + 1}: {np.mean(hist[-1000:]):.2e} ({time.time() - t0:.0f}s)")
    if lbfgs_iters:
        x, t = sample(2 * n_colloc)
        _lbfgs(model, lambda: burgers_residual(model, x, t, nu).pow(2).mean(), lbfgs_iters, hist)
    if verbose:
        print(f"burgers: final loss {hist[-1]:.2e} in {time.time() - t0:.0f}s")
    return {"loss": hist}


@torch.no_grad()
def evaluate_on_grid(model: BurgersPINN, x: np.ndarray, times: np.ndarray) -> np.ndarray:
    """u on the tensor grid times x x, shape (len(times), len(x))."""
    X, T = np.meshgrid(x, times)
    xt = torch.tensor(X.reshape(-1, 1), dtype=DTYPE)
    tt = torch.tensor(T.reshape(-1, 1), dtype=DTYPE)
    return model(xt, tt).numpy().reshape(T.shape)


def _lbfgs(model: nn.Module, closure_loss, iters: int, hist: list[float]) -> None:
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=iters,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    def closure():
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        hist.append(float(loss))
        return loss

    opt.step(closure)
