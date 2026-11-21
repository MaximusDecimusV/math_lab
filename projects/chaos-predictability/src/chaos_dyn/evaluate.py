"""Error-vs-step curves, breakdown steps, the chaos ("twin trajectory") baseline, and
invariant drift. All errors are dimensionless: divided by a length scale of the system."""

from __future__ import annotations

import numpy as np

from chaos_dyn.data import Dataset, solve_many
from chaos_dyn.integrate import rk4_solve
from chaos_dyn.systems import (
    lorenz,
    split_state,
    system_scale,
    three_body_angular_momentum,
    three_body_energy,
)

LORENZ_SCALE = 15.0  # RMS distance of attractor points from their mean (14.8, measured)


def lorenz_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|pred - ref| / LORENZ_SCALE, shape (M, T)."""
    return np.linalg.norm(pred - ref, axis=-1) / LORENZ_SCALE


def three_body_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMS position error over bodies / system scale of the initial condition, shape (M, T)."""
    qp, _ = split_state(pred)
    qr, _ = split_state(ref)
    rms = np.sqrt(np.mean(np.sum((qp - qr) ** 2, -1), -1))
    return rms / system_scale(ref[:, :1])


def breakdown_step(err: np.ndarray, tol: float) -> np.ndarray:
    """First step k with err[:, k] > tol; n_steps + 1 if it never happens."""
    over = err > tol
    first = np.argmax(over, axis=1)
    return np.where(over.any(axis=1), first, err.shape[1])


def band(err: np.ndarray, q=(10, 50, 90)) -> np.ndarray:
    """Percentiles across trajectories, shape (len(q), T). Geometric data, so use log-safe values."""
    return np.percentile(err, q, axis=0)


def one_step_error(model_step_pred: np.ndarray, ref: np.ndarray, err_fn) -> float:
    """Median normalised error after a single step."""
    return float(np.median(err_fn(model_step_pred[:, :2], ref[:, :2])[:, 1]))


def twin_baseline_lorenz(test: Dataset, delta0: float, seed: int = 0, params=(10, 28, 8 / 3)):
    """Exact dynamics from an IC displaced by delta0 (in normalised units, random direction).

    This is the best any predictor with initial error delta0 could do: the divergence is
    entirely due to chaos, not to model error."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(test.trajs[:, 0].shape)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    s0 = test.trajs[:, 0] + delta0 * LORENZ_SCALE * u
    n = test.trajs.shape[1] - 1
    twin = rk4_solve(lorenz, s0, n, test.dt, params, substeps=10)
    return lorenz_error(twin, test.trajs)


def twin_baseline_three_body(test: Dataset, delta0: float, seed: int = 0, workers: int = 4):
    """Same idea for three-body: displace positions by delta0 * system scale, re-solve exactly."""
    rng = np.random.default_rng(seed)
    s0 = test.trajs[:, 0].copy()
    u = rng.standard_normal(s0[:, :6].shape)
    u /= np.sqrt(np.mean(u.reshape(-1, 3, 2) ** 2, axis=(1, 2)) * 2)[:, None]  # unit RMS per body
    s0[:, :6] += delta0 * system_scale(s0)[:, None] * u
    n = test.trajs.shape[1] - 1
    m = test.cond[0]
    twins = solve_many(s0, m, n, test.dt, workers=workers)
    return three_body_error(twins, test.trajs)


def three_body_drift(pred: np.ndarray, m: np.ndarray) -> dict[str, np.ndarray]:
    """|E(t) - E(0)| / |E(0)| and |L(t) - L(0)| / scale, shapes (M, T)."""
    E = three_body_energy(pred, m)
    L = three_body_angular_momentum(pred, m)
    return {
        "energy": np.abs(E - E[:, :1]) / np.abs(E[:, :1]),
        "angular_momentum": np.abs(L - L[:, :1]),
    }
