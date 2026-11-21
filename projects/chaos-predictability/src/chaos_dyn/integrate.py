"""Reference integrators.

There is no closed-form solution for Lorenz, the double pendulum, or the general
three-body problem, so "ground truth" means a high-order adaptive integrator run at
tolerances far below any error we want to measure (checked in tests/test_systems.py).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

RHS = Callable[..., np.ndarray]


def reference_solve(
    rhs: RHS,
    s0: np.ndarray,
    n_steps: int,
    dt: float,
    args: tuple = (),
    rtol: float = 1e-12,
    atol: float = 1e-12,
    events=None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """DOP853 (8th-order Dormand-Prince) sampled on the grid t_k = k dt, k = 0..n_steps.

    Returns (times, states, sol). ``states`` has shape (n_steps + 1, d) unless a terminal
    event fires, in which case it is truncated and ``sol.status == 1``.
    """
    t_eval = dt * np.arange(n_steps + 1)
    sol = solve_ivp(
        rhs,
        (0.0, t_eval[-1]),
        np.asarray(s0, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        args=args,
        rtol=rtol,
        atol=atol,
        events=events,
    )
    if sol.status == -1:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T, sol


def rk4_step(rhs: RHS, s: np.ndarray, dt: float, args: tuple = ()) -> np.ndarray:
    """Classical fixed-step RK4. Vectorises over leading axes."""
    k1 = rhs(0.0, s, *args)
    k2 = rhs(0.0, s + 0.5 * dt * k1, *args)
    k3 = rhs(0.0, s + 0.5 * dt * k2, *args)
    k4 = rhs(0.0, s + dt * k3, *args)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_solve(
    rhs: RHS, s0: np.ndarray, n_steps: int, dt: float, args: tuple = (), substeps: int = 1
) -> np.ndarray:
    """Fixed-step RK4 on a batch of states (..., d). Returns (..., n_steps + 1, d).

    Only for autonomous, non-stiff, non-singular systems (Lorenz). Used to generate
    large Lorenz datasets quickly; validated against reference_solve in the tests.
    """
    h = dt / substeps
    s = np.asarray(s0, dtype=float)
    out = [s]
    for _ in range(n_steps):
        for _ in range(substeps):
            s = rk4_step(rhs, s, h, args)
        out.append(s)
    return np.stack(out, axis=-2)
