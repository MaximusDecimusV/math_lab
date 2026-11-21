"""Reference-solution checks. If these fail, no learned-model error means anything."""

import numpy as np
import pytest

from chaos_dyn import ics
from chaos_dyn.integrate import reference_solve, rk4_solve
from chaos_dyn.systems import (
    double_pendulum,
    double_pendulum_energy,
    lorenz,
    lorenz_fixed_points,
    lorenz_hopf_r,
    three_body,
    three_body_angular_momentum,
    three_body_energy,
    three_body_momentum,
)

M = ics.EQUAL_MASSES


def test_figure_eight_is_periodic():
    s0 = ics.figure_eight()
    n = 1000
    _, S, _ = reference_solve(three_body, s0, n, ics.FIGURE_EIGHT_PERIOD / n, args=(M,))
    # The published ICs carry ~8 significant digits, so return error ~1e-8 is the floor.
    assert np.max(np.abs(S[-1] - s0)) < 1e-6


@pytest.mark.parametrize("family", ["figure_eight", "lagrange"])
def test_three_body_invariants(family):
    base = {"figure_eight": ics.figure_eight, "lagrange": ics.lagrange_triangle}[family]()
    s0 = ics.perturb(base, np.random.default_rng(0), 1, 0.03)[0]
    _, S, _ = reference_solve(three_body, s0, 500, 0.02, args=(M,))
    E, L = three_body_energy(S, M), three_body_angular_momentum(S, M)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-9
    assert np.max(np.abs(L - L[0])) < 1e-9
    assert np.max(np.abs(three_body_momentum(S, M))) < 1e-12


def test_reference_converged_in_tolerance():
    """Tightening rtol 1e-12 -> 1e-13 changes the Lagrange-family answer by << model error."""
    s0 = ics.perturb(ics.lagrange_triangle(), np.random.default_rng(1), 1, 0.03)[0]
    _, a, _ = reference_solve(three_body, s0, 500, 0.02, args=(M,), rtol=1e-12, atol=1e-12)
    _, b, _ = reference_solve(three_body, s0, 500, 0.02, args=(M,), rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(a - b)) < 1e-7


def test_lagrange_triangle_is_a_relative_equilibrium():
    s0 = ics.lagrange_triangle()
    _, S, _ = reference_solve(three_body, s0, 100, 0.02, args=(M,))  # t = 2: before instability
    r = np.linalg.norm(S[:, :2], axis=1)
    assert np.max(np.abs(r - r[0])) < 1e-6


def test_double_pendulum_energy():
    s0 = np.array([2.0, 2.5, 0.0, 0.0])
    _, S, _ = reference_solve(double_pendulum, s0, 1000, 0.01)
    E = double_pendulum_energy(S)
    assert np.max(np.abs(E - E[0])) < 1e-8


def test_lorenz_fixed_points_and_hopf():
    for p in lorenz_fixed_points(28.0):
        assert np.allclose(lorenz(0, p), 0, atol=1e-12)
    assert abs(lorenz_hopf_r() - 24.7368) < 1e-3


def test_lorenz_rk4_matches_dop853():
    s0 = np.array([-8.0, 8.0, 27.0])
    _, ref, _ = reference_solve(lorenz, s0, 500, 0.01)
    fast = rk4_solve(lorenz, s0, 500, 0.01, substeps=10)
    assert np.max(np.abs(fast - ref)) < 1e-5  # vs attractor size 15 and model errors >= 1e-4
