"""Pseudospectral reference solver for viscous Burgers on the periodic domain [0, 2*pi).

    u_t + u u_x = nu u_xx                                   (eq. 1 in notes/derivation.typ)

Written in conservative form u_t + (u^2 / 2)_x = nu u_xx. Space is Fourier
spectral with 2/3-rule dealiasing; time is integrating-factor RK4 (Lawson), which
treats diffusion exactly so the step size is limited by the nonlinearity only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BurgersSpectral:
    n: int = 128
    nu: float = 0.1
    x: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    _dealias: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.x = 2 * np.pi * np.arange(self.n) / self.n
        self.k = np.fft.rfftfreq(self.n, d=1.0 / self.n)
        self._dealias = self.k < (2.0 / 3.0) * (self.n // 2)

    def nonlinear(self, u_hat: np.ndarray) -> np.ndarray:
        """Fourier transform of -(u^2/2)_x, dealiased. Implements the N(u) term in eq. 4."""
        u = np.fft.irfft(u_hat * self._dealias, n=self.n)
        return -0.5j * self.k * np.fft.rfft(u * u) * self._dealias

    def step(self, u_hat: np.ndarray, dt: float) -> np.ndarray:
        """One Lawson / integrating-factor RK4 step. Implements eq. 5."""
        lin = -self.nu * self.k**2
        e = np.exp(lin * dt)
        e2 = np.exp(lin * dt / 2)
        a = self.nonlinear(u_hat)
        b = self.nonlinear(e2 * (u_hat + 0.5 * dt * a))
        c = self.nonlinear(e2 * u_hat + 0.5 * dt * b)
        d = self.nonlinear(e * u_hat + dt * e2 * c)
        return e * u_hat + (dt / 6.0) * (e * a + 2.0 * e2 * (b + c) + d)

    def solve(
        self, u0: np.ndarray, t_final: float, dt: float, save_every: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from u0 to t_final. Returns (times, snapshots) with snapshots shaped (T, n)."""
        n_steps = round(t_final / dt)
        if not np.isclose(n_steps * dt, t_final):
            raise ValueError("t_final must be an integer multiple of dt")
        save_every = save_every or n_steps
        u_hat = np.fft.rfft(u0)
        times, snaps = [0.0], [u0.copy()]
        for i in range(1, n_steps + 1):
            u_hat = self.step(u_hat, dt)
            if i % save_every == 0:
                times.append(i * dt)
                snaps.append(np.fft.irfft(u_hat, n=self.n))
        return np.array(times), np.array(snaps)


def cole_hopf_solution(x: np.ndarray, t: float, nu: float, b: float = 2.0) -> np.ndarray:
    """Exact solution from phi = b + exp(-nu t) cos x via u = -2 nu (log phi)_x. Implements eq. 3.

    Requires b > 1 so that phi > 0.
    """
    decay = np.exp(-nu * t)
    return 2.0 * nu * decay * np.sin(x) / (b + decay * np.cos(x))


def mass(u: np.ndarray) -> float:
    """Spatial mean of u; conserved exactly by eq. 1 on a periodic domain."""
    return float(np.mean(u))


def energy(u: np.ndarray) -> float:
    """Mean of u^2 / 2; non-increasing under eq. 1 (eq. 2)."""
    return float(0.5 * np.mean(u * u))
