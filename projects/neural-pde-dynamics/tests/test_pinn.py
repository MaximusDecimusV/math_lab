"""Checks on the neural PDE solvers. Fast checks run by default; full-accuracy runs are `slow`."""

import numpy as np
import pytest
import torch

from neural_pde import cole_hopf_solution
from neural_pde.pinn import (
    BurgersPINN,
    LaplaceTrial,
    burgers_residual,
    evaluate_on_grid,
    grid,
    laplace_exact,
    laplace_fd_solve,
    train_burgers,
    train_laplace,
)

NU = 0.1


def cole_hopf_u0(x):
    return 2 * NU * torch.sin(x) / (2 + torch.cos(x))


def laplace_max_error(model):
    xy = grid(41, interior=False)
    with torch.no_grad():
        u = model(xy).numpy().ravel()
    return np.max(np.abs(u - laplace_exact(xy[:, 0].numpy(), xy[:, 1].numpy())))


def test_laplace_trial_satisfies_bcs_untrained():
    model = LaplaceTrial()
    s = torch.linspace(0, 1, 11, dtype=torch.float64)[:, None]
    z, o = torch.zeros_like(s), torch.ones_like(s)
    with torch.no_grad():
        assert torch.allclose(model(torch.cat([z, s], 1)), z)
        assert torch.allclose(model(torch.cat([o, s], 1)), z, atol=1e-12)
        assert torch.allclose(model(torch.cat([s, z], 1)), z)
        assert torch.allclose(model(torch.cat([s, o], 1)), torch.sin(np.pi * s), atol=1e-12)


def test_laplace_fd_converges_second_order():
    errs = []
    for n in (15, 31):
        s, U = laplace_fd_solve(n)
        X, Y = np.meshgrid(s, s, indexing="ij")
        errs.append(np.max(np.abs(U - laplace_exact(X, Y))))
    assert 3.5 < errs[0] / errs[1] < 4.5


def test_laplace_short_training():
    model = LaplaceTrial()
    train_laplace(model, n_colloc=12, adam_iters=300, lbfgs_iters=50)
    assert laplace_max_error(model) < 1e-2


def test_burgers_pinn_hard_constraints():
    model = BurgersPINN(cole_hopf_u0, t_final=1.0)
    x = torch.linspace(0, 2 * np.pi, 9, dtype=torch.float64)[:, None]
    with torch.no_grad():
        assert torch.allclose(model(x, torch.zeros_like(x)), cole_hopf_u0(x))
        t = torch.full_like(x, 0.7)
        assert torch.allclose(model(x[:1], t[:1]), model(x[-1:], t[-1:]), atol=1e-12)


def test_burgers_residual_zero_on_exact_solution():
    """The residual operator is right: plug Cole-Hopf in through a stand-in 'model'."""

    class Exact(torch.nn.Module):
        def forward(self, x, t):
            d = torch.exp(-NU * t)
            return 2 * NU * d * torch.sin(x) / (2 + d * torch.cos(x))

    x = 2 * np.pi * torch.rand(64, 1, dtype=torch.float64)
    t = torch.rand(64, 1, dtype=torch.float64)
    assert burgers_residual(Exact(), x, t, NU).abs().max() < 1e-12


def test_burgers_short_training_reduces_residual():
    model = BurgersPINN(cole_hopf_u0, t_final=1.0, hidden=32, depth=3)
    hist = train_burgers(model, NU, n_colloc=512, adam_iters=200, lbfgs_iters=0)["loss"]
    assert np.mean(hist[-20:]) < 0.1 * np.mean(hist[:20])


@pytest.mark.slow
def test_laplace_full_accuracy():
    model = LaplaceTrial()
    train_laplace(model)
    assert laplace_max_error(model) < 1e-3


@pytest.mark.slow
def test_burgers_pinn_matches_cole_hopf():
    model = BurgersPINN(cole_hopf_u0, t_final=1.0)
    train_burgers(model, NU, n_colloc=2048, adam_iters=2000, lbfgs_iters=300)
    x = 2 * np.pi * np.arange(128) / 128
    ts = np.linspace(0, 1, 11)
    exact = np.array([cole_hopf_solution(x, t, NU) for t in ts])
    assert np.max(np.abs(evaluate_on_grid(model, x, ts) - exact)) < 1e-3
