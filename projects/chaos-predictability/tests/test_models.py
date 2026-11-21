import numpy as np
import torch

from chaos_dyn.data import Dataset
from chaos_dyn.integrate import rk4_solve
from chaos_dyn.models import DirectMLP, StepMLP
from chaos_dyn.systems import lorenz
from chaos_dyn.train import predict, train_direct_model, train_step_model


def tiny_lorenz(n=64, steps=20):
    rng = np.random.default_rng(0)
    s0 = rk4_solve(lorenz, rng.normal(0, 5, (n, 3)) + [0, 0, 25], 500, 0.01)[:, -1]
    return Dataset(rk4_solve(lorenz, s0, steps, 0.01), np.zeros((n, 0)), 0.01, {})


def test_shapes():
    ds = tiny_lorenz()
    for model in (StepMLP(ds.trajs, ds.cond, 32, 2), DirectMLP(ds.trajs, ds.cond, 32, 2)):
        assert predict(model, ds).shape == ds.trajs.shape


def test_direct_model_returns_ic_exactly():
    ds = tiny_lorenz()
    out = predict(DirectMLP(ds.trajs, ds.cond, 32, 2), ds)
    assert np.allclose(out[:, 0], ds.trajs[:, 0], atol=1e-5)


def test_training_reduces_loss():
    torch.manual_seed(0)
    ds = tiny_lorenz()
    losses = train_step_model(
        StepMLP(ds.trajs, ds.cond, 64, 2), ds, iters=300, batch=256, log_every=0
    )
    assert np.mean(losses[-30:]) < 0.1 * np.mean(losses[:30])
    losses = train_step_model(
        StepMLP(ds.trajs, ds.cond, 64, 2), ds, iters=100, batch=64, unroll=4, log_every=0
    )
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    losses = train_direct_model(
        DirectMLP(ds.trajs, ds.cond, 64, 2), ds, iters=300, batch=64, log_every=0
    )
    assert np.mean(losses[-30:]) < 0.5 * np.mean(losses[:30])
