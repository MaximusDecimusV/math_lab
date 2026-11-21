import numpy as np

from chaos_dyn.lyapunov import max_lyapunov, max_lyapunov_rk4, predictability_horizon
from chaos_dyn.systems import lorenz


def test_lorenz_max_lyapunov():
    lam, _ = max_lyapunov(lorenz, np.array([1.0, 1.0, 1.0]), 150.0, transient=20.0)
    assert 0.85 < lam < 0.95  # literature: 0.9056


def test_lorenz_max_lyapunov_rk4_batched():
    lam = max_lyapunov_rk4(lorenz, np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 20.0]]), 200.0)
    assert np.all((0.85 < lam) & (lam < 0.97))


def test_stable_fixed_point_has_negative_exponent():
    # r = 10 < r_H: trajectories spiral into C+ or C-, so lambda_max < 0.
    lam, _ = max_lyapunov(
        lorenz, np.array([1.0, 1.0, 1.0]), 50.0, args=(10, 10, 8 / 3), transient=50.0
    )
    assert lam < 0


def test_horizon_formula():
    assert np.isclose(predictability_horizon(1.0, 1e-4, 1e-1), np.log(1e3))
