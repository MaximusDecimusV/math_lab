"""Mathematical checks on the reference solver. If these fail, no surrogate result means anything."""

import numpy as np
import pytest

from neural_pde import BurgersSpectral, cole_hopf_solution, energy, mass

NU = 0.1


@pytest.fixture
def solver() -> BurgersSpectral:
    return BurgersSpectral(n=128, nu=NU)


def test_matches_cole_hopf(solver):
    u0 = cole_hopf_solution(solver.x, 0.0, NU)
    _, snaps = solver.solve(u0, t_final=1.0, dt=1e-3)
    exact = cole_hopf_solution(solver.x, 1.0, NU)
    assert np.max(np.abs(snaps[-1] - exact)) < 1e-8


def test_mass_conserved(solver):
    rng = np.random.default_rng(0)
    u0 = np.sin(solver.x) + 0.3 * np.cos(3 * solver.x + rng.uniform(0, 2 * np.pi)) + 0.2
    _, snaps = solver.solve(u0, t_final=2.0, dt=1e-3, save_every=100)
    masses = np.array([mass(u) for u in snaps])
    assert np.max(np.abs(masses - masses[0])) < 1e-12


def test_energy_non_increasing(solver):
    u0 = np.sin(solver.x) + 0.5 * np.sin(2 * solver.x)
    _, snaps = solver.solve(u0, t_final=2.0, dt=1e-3, save_every=50)
    energies = np.array([energy(u) for u in snaps])
    assert np.all(np.diff(energies) <= 1e-14)
    assert energies[-1] < energies[0]


def test_fourth_order_in_time():
    # Compare against a fine-dt run of the same spatial discretization, so spatial
    # error cancels and only temporal error remains.
    s = BurgersSpectral(n=64, nu=NU)
    u0 = np.sin(s.x) + 0.5 * np.sin(2 * s.x)
    reference = s.solve(u0, 2.0, 1e-3)[1][-1]
    dts = [0.05, 0.025, 0.0125]
    errs = [np.max(np.abs(s.solve(u0, 2.0, dt)[1][-1] - reference)) for dt in dts]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 3.8), f"observed orders {orders}, errors {errs}"
