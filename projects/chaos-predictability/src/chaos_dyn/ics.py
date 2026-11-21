"""Initial-condition families for the planar three-body problem (G = 1).

All samplers return states already in the centre-of-mass frame (zero total momentum).
"""

from __future__ import annotations

import numpy as np

from chaos_dyn.systems import join_state, min_pair_distance, three_body_energy, to_com_frame

EQUAL_MASSES = np.ones(3)
PYTHAGOREAN_MASSES = np.array([3.0, 4.0, 5.0])

# Chenciner & Montgomery (2000) figure-eight, initial data from Simo / Moore.
FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight() -> np.ndarray:
    """Equal-mass figure-eight choreography. Eq. <figure-eight>."""
    x1 = np.array([0.97000436, -0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    q = np.stack([x1, -x1, np.zeros(2)])
    v = np.stack([-v3 / 2, -v3 / 2, v3])
    return join_state(q, v)


def pythagorean() -> np.ndarray:
    """Burrau's Pythagorean problem: masses 3, 4, 5 at rest on a 3-4-5 triangle."""
    q = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]])
    return join_state(q, np.zeros((3, 2)))


def lagrange_triangle(omega: float = 1.0) -> np.ndarray:
    """Equal masses on an equilateral triangle in rigid rotation (Lagrange's solution).

    For side a, rigid rotation needs omega^2 = 3 m / a^3 (G = 1), so a = (3 / omega^2)^(1/3).
    """
    a = (3.0 / omega**2) ** (1 / 3)
    rad = a / np.sqrt(3)
    ang = 2 * np.pi * np.arange(3) / 3
    q = rad * np.stack([np.cos(ang), np.sin(ang)], -1)
    v = omega * rad * np.stack([-np.sin(ang), np.cos(ang)], -1)
    return join_state(q, v)


def perturb(
    s0: np.ndarray, rng: np.random.Generator, n: int, scale: float, m: np.ndarray = EQUAL_MASSES
) -> np.ndarray:
    """n copies of s0 with i.i.d. Gaussian noise of std ``scale`` on every coordinate."""
    s = s0[None, :] + scale * rng.standard_normal((n, s0.size))
    return to_com_frame(s, m)


def random_bound(
    rng: np.random.Generator,
    n: int,
    box: float = 1.0,
    v_std: float = 0.3,
    min_sep: float = 0.3,
    m: np.ndarray = EQUAL_MASSES,
) -> np.ndarray:
    """Random equal-mass configurations: positions uniform in [-box, box]^2, Gaussian
    velocities, rejected unless the system is bound (E < 0) and no pair starts closer
    than ``min_sep``."""
    out = []
    while sum(len(o) for o in out) < n:
        q = rng.uniform(-box, box, size=(4 * n, 3, 2))
        v = v_std * rng.standard_normal((4 * n, 3, 2))
        s = to_com_frame(join_state(q, v), m)
        keep = (three_body_energy(s, m) < 0) & (min_pair_distance(s) > min_sep)
        out.append(s[keep])
    return np.concatenate(out)[:n]
