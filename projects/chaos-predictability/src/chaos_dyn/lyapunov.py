"""Maximal Lyapunov exponent by Benettin renormalisation. Implements eq. <lyap-def>.

Follow a reference trajectory and a companion displaced by d0. Every tau time units,
record log(|delta| / d0) and rescale the displacement back to length d0 along its current
direction. The running average of the logs divided by elapsed time converges to lambda_max.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from chaos_dyn.integrate import RHS, rk4_step


def _pair_rhs(rhs: RHS, d: int):
    def f(t, y, *args):
        return np.concatenate([rhs(t, y[:d], *args), rhs(t, y[d:], *args)])

    return f


def max_lyapunov(
    rhs: RHS,
    s0: np.ndarray,
    t_total: float,
    args: tuple = (),
    tau: float = 0.5,
    d0: float = 1e-8,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    transient: float = 0.0,
    rng: np.random.Generator | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Benettin estimate of the largest Lyapunov exponent.

    Returns (lambda_estimate, running_estimates). ``weights`` scales each coordinate when
    measuring distance (useful when positions and velocities have different units).
    ``transient`` time is integrated first and discarded (to land on the attractor).
    """
    rng = rng or np.random.default_rng(0)
    s = np.asarray(s0, dtype=float)
    d = s.size
    w = np.ones(d) if weights is None else np.asarray(weights, dtype=float)
    pair = _pair_rhs(rhs, d)

    if transient > 0:
        s = solve_ivp(rhs, (0, transient), s, method="DOP853", args=args, rtol=rtol, atol=atol).y[
            :, -1
        ]

    u = rng.standard_normal(d)
    u /= np.linalg.norm(u * w)
    y = np.concatenate([s, s + d0 * u])
    n_chunks = round(t_total / tau)
    logs = np.empty(n_chunks)
    for k in range(n_chunks):
        sol = solve_ivp(pair, (0, tau), y, method="DOP853", args=args, rtol=rtol, atol=atol)
        y = sol.y[:, -1]
        delta = y[d:] - y[:d]
        dist = np.linalg.norm(delta * w)
        logs[k] = np.log(dist / d0)
        y[d:] = y[:d] + delta * (d0 / dist)
    running = np.cumsum(logs) / (tau * np.arange(1, n_chunks + 1))
    return float(running[-1]), running


def max_lyapunov_rk4(
    rhs: RHS,
    s0: np.ndarray,
    t_total: float,
    dt: float = 0.005,
    args: tuple = (),
    renorm_every: int = 20,
    d0: float = 1e-8,
    transient: float = 20.0,
) -> float:
    """Same estimate with fixed-step RK4, batched over many s0 at once (shape (B, d)).

    Much faster for smooth, non-singular systems like Lorenz. Returns an array (B,).
    """
    s = np.atleast_2d(np.asarray(s0, dtype=float))
    for _ in range(round(transient / dt)):
        s = rk4_step(rhs, s, dt, args)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(s.shape)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    p = s + d0 * u
    n = round(t_total / dt)
    total = np.zeros(s.shape[0])
    for k in range(1, n + 1):
        s = rk4_step(rhs, s, dt, args)
        p = rk4_step(rhs, p, dt, args)
        if k % renorm_every == 0:
            delta = p - s
            dist = np.linalg.norm(delta, axis=-1, keepdims=True)
            total += np.log(dist[:, 0] / d0)
            p = s + delta * (d0 / dist)
    return total / (n * dt)


def predictability_horizon(lam: float, delta0: float, tol: float) -> float:
    """t_p = ln(tol / delta0) / lambda. Implements eq. <horizon>."""
    return np.log(tol / delta0) / lam


def predictability_horizon_accumulated(lam: float, eps: float, tol: float, dt: float) -> float:
    """Horizon when a fresh error eps is made at *every* step of size dt (eq. <horizon-acc>).

    Errors injected at steps k = 0..n-1 each grow by e^(lambda (n-k) dt); summing the
    geometric series gives eps (e^(lambda t) - 1) / (e^(lambda dt) - 1), i.e. an effective
    initial error delta_eff = eps / (e^(lambda dt) - 1), roughly eps / (lambda dt).
    """
    return predictability_horizon(lam, eps / np.expm1(lam * dt), tol)


def _ftle_job(job):
    rhs, s0, t_total, args, tau, weights = job
    return max_lyapunov(rhs, s0, t_total, args=args, tau=tau, weights=weights)[0]


def max_lyapunov_many(rhs: RHS, s0s, t_total, args=(), tau=0.5, weights=None, workers=4):
    """Finite-time max Lyapunov exponent for many ICs in parallel (one Benettin run each)."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(rhs, s, t_total, args, tau, weights) for s in s0s]
    with ProcessPoolExecutor(workers) as ex:
        return np.array(list(ex.map(_ftle_job, jobs, chunksize=8)))
