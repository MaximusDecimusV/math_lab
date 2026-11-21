"""Right-hand sides and invariants for the chaotic ODEs used in this project.

Every RHS has the signature ``f(t, s, *params) -> ds/dt`` so it plugs straight into
``scipy.integrate.solve_ivp``. Equation labels refer to notes/nonlinear-dynamics.typ.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- Lorenz

LORENZ_CLASSIC = (10.0, 28.0, 8.0 / 3.0)  # (sigma, r, b)


def lorenz(t: float, s: np.ndarray, sigma: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0):
    """Lorenz system. Implements eq. <lorenz>.

    x' = sigma (y - x),  y' = r x - y - x z,  z' = x y - b z.
    Works on a single state (3,) or a batch (..., 3).
    """
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([sigma * (y - x), r * x - y - x * z, x * y - b * z], axis=-1)


def lorenz_fixed_points(r: float = 28.0, b: float = 8.0 / 3.0) -> np.ndarray:
    """Origin plus C+/C- = (+-sqrt(b(r-1)), +-sqrt(b(r-1)), r-1) for r > 1. Eq. <lorenz-fp>."""
    pts = [np.zeros(3)]
    if r > 1:
        c = np.sqrt(b * (r - 1))
        pts += [np.array([c, c, r - 1]), np.array([-c, -c, r - 1])]
    return np.array(pts)


def lorenz_hopf_r(sigma: float = 10.0, b: float = 8.0 / 3.0) -> float:
    """Subcritical Hopf of C+/C-: r_H = sigma (sigma + b + 3) / (sigma - b - 1). Eq. <lorenz-hopf>."""
    return sigma * (sigma + b + 3) / (sigma - b - 1)


# ------------------------------------------------------------------ double pendulum


def double_pendulum(
    t: float,
    s: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
):
    """Planar double pendulum, state (theta1, theta2, omega1, omega2). Implements eq. <dp-eom>.

    Angles measured from the downward vertical; point masses on massless rods.
    """
    th1, th2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = th1 - th2
    den = 2 * m1 + m2 - m2 * np.cos(2 * d)
    a1 = (
        -g * (2 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2 * th2)
        - 2 * np.sin(d) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(d))
    ) / (l1 * den)
    a2 = (
        2
        * np.sin(d)
        * (w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + w2**2 * l2 * m2 * np.cos(d))
    ) / (l2 * den)
    return np.stack([w1, w2, a1, a2], axis=-1)


def double_pendulum_energy(s, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """Total energy T + V (conserved). Eq. <dp-energy>."""
    th1, th2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    kin = 0.5 * (m1 + m2) * l1**2 * w1**2 + 0.5 * m2 * l2**2 * w2**2
    kin = kin + m2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
    pot = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return kin + pot


def double_pendulum_xy(s, l1=1.0, l2=1.0):
    """Cartesian positions of both bobs, shape (..., 2, 2)."""
    th1, th2 = s[..., 0], s[..., 1]
    x1, y1 = l1 * np.sin(th1), -l1 * np.cos(th1)
    x2, y2 = x1 + l2 * np.sin(th2), y1 - l2 * np.cos(th2)
    return np.stack([np.stack([x1, y1], -1), np.stack([x2, y2], -1)], -2)


# ------------------------------------------------------------------- three-body
#
# Planar, G = 1. State layout (12,): positions (x1, y1, x2, y2, x3, y3), then
# velocities (vx1, vy1, vx2, vy2, vx3, vy3).

N_BODIES = 3
DIM = 2
STATE_DIM = 2 * N_BODIES * DIM  # 12


def split_state(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 12) -> positions (..., 3, 2), velocities (..., 3, 2)."""
    q = s[..., : N_BODIES * DIM].reshape(*s.shape[:-1], N_BODIES, DIM)
    v = s[..., N_BODIES * DIM :].reshape(*s.shape[:-1], N_BODIES, DIM)
    return q, v


def join_state(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    lead = q.shape[:-2]
    return np.concatenate([q.reshape(*lead, -1), v.reshape(*lead, -1)], axis=-1)


def three_body(t: float, s: np.ndarray, m: np.ndarray):
    """Newtonian gravity for three point masses, G = 1. Implements eq. <nbody>.

    q_i'' = sum_{j != i} m_j (q_j - q_i) / |q_j - q_i|^3.
    """
    m = np.asarray(m, dtype=float)
    q, v = split_state(s)
    diff = q[..., None, :, :] - q[..., :, None, :]  # diff[i, j] = q_j - q_i
    # +eye avoids 0/0 on the diagonal; diff is zero there, so it contributes nothing.
    inv_r3 = (np.sum(diff**2, axis=-1) + np.eye(N_BODIES)) ** -1.5
    acc = np.einsum("...ijk,...ij,j->...ik", diff, inv_r3, m)
    return join_state(v, acc)


def three_body_energy(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """E = sum m_i |v_i|^2 / 2 - sum_{i<j} m_i m_j / r_ij. Eq. <nbody-invariants>."""
    m = np.asarray(m, dtype=float)
    q, v = split_state(s)
    kin = 0.5 * np.sum(m * np.sum(v**2, axis=-1), axis=-1)
    pot = 0.0
    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            pot = pot - m[i] * m[j] / np.linalg.norm(q[..., i, :] - q[..., j, :], axis=-1)
    return kin + pot


def three_body_momentum(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Total linear momentum P = sum m_i v_i, shape (..., 2). Eq. <nbody-invariants>."""
    _, v = split_state(s)
    return np.einsum("...ik,i->...k", v, np.asarray(m, dtype=float))


def three_body_angular_momentum(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Total angular momentum L = sum m_i (x_i vy_i - y_i vx_i). Eq. <nbody-invariants>."""
    q, v = split_state(s)
    cross = q[..., 0] * v[..., 1] - q[..., 1] * v[..., 0]
    return np.sum(np.asarray(m, dtype=float) * cross, axis=-1)


def min_pair_distance(s: np.ndarray) -> np.ndarray:
    """Smallest of the three pairwise distances, shape (...)."""
    q, _ = split_state(s)
    d = [np.linalg.norm(q[..., i, :] - q[..., j, :], axis=-1) for i, j in ((0, 1), (0, 2), (1, 2))]
    return np.min(np.stack(d, -1), axis=-1)


def to_com_frame(s: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Shift so the centre of mass sits at the origin with zero total momentum."""
    m = np.asarray(m, dtype=float)
    q, v = split_state(s)
    q = q - np.einsum("...ik,i->...k", q, m)[..., None, :] / m.sum()
    v = v - np.einsum("...ik,i->...k", v, m)[..., None, :] / m.sum()
    return join_state(q, v)


def system_scale(s: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    """RMS distance of the bodies from their centroid: the length used to normalise errors."""
    q, _ = split_state(s)
    c = q.mean(axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.sum((q - c) ** 2, axis=-1), axis=-1))
