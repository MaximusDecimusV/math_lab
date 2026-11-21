# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A tour from fixed points to chaos
#
# One figure per section of `notes/nonlinear-dynamics.typ`. Everything is computed with the
# reference integrators in `chaos_dyn`, so the same code produces the training data later.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from chaos_dyn import ics
from chaos_dyn.integrate import reference_solve
from chaos_dyn.lyapunov import max_lyapunov
from chaos_dyn.systems import (
    double_pendulum,
    double_pendulum_energy,
    lorenz,
    lorenz_hopf_r,
    split_state,
    three_body,
)

FIG = Path.cwd().parent / "figures" if Path.cwd().name == "notebooks" else Path("figures")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=110, bbox_inches="tight")


# %% [markdown]
# ## 1-2. One dimension: fixed points and a pitchfork
# Left: the phase line of $\dot x = x - x^3$. Right: the pitchfork $\dot x = r x - x^3$.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.linspace(-1.6, 1.6, 400)
a.plot(x, x - x**3, color="k")
a.axhline(0, color="0.6", lw=0.8)
for xs, stable in [(-1, True), (0, False), (1, True)]:
    a.plot(xs, 0, "o", ms=9, mfc="k" if stable else "w", mec="k")
for xs in (-1.4, -0.5, 0.5, 1.4):
    a.annotate(
        "",
        (xs + 0.15 * np.sign(xs - xs**3), 0),
        (xs, 0),
        arrowprops={"arrowstyle": "->", "color": "C3", "lw": 2},
    )
a.set(xlabel="x", ylabel=r"$\dot x$", title=r"$\dot x = x - x^3$: filled = stable, open = unstable")

r = np.linspace(-1, 1, 200)
b.plot(r[r <= 0], 0 * r[r <= 0], "k", label="stable")
b.plot(r[r >= 0], 0 * r[r >= 0], "k--", label="unstable")
b.plot(r[r >= 0], np.sqrt(r[r >= 0]), "k")
b.plot(r[r >= 0], -np.sqrt(r[r >= 0]), "k")
b.set(xlabel="r", ylabel="x*", title=r"Supercritical pitchfork $\dot x = r x - x^3$")
b.legend(frameon=False)
fig.tight_layout()
save(fig, "tour-1d")

# %% [markdown]
# ## 3-5. Two dimensions: a conservative center and a limit cycle
# Lotka-Volterra $\dot x = x(1-y),\ \dot y = \mu y (x-1)$ conserves
# $E = \ln y - y + \mu(\ln x - x)$, so orbits are closed. Van der Pol ($\mu = 1$) has an
# unstable spiral at the origin and one attracting limit cycle. Poincaré-Bendixson says
# this is as complicated as a planar flow gets.

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
mu = 0.5
X, Y = np.meshgrid(np.linspace(0.05, 3, 30), np.linspace(0.05, 3, 30))
a.streamplot(X, Y, X * (1 - Y), mu * Y * (X - 1), color="0.75", density=1.1)
for x0 in (1.2, 1.6, 2.0, 2.4):
    s = solve_ivp(
        lambda t, s: [s[0] * (1 - s[1]), mu * s[1] * (s[0] - 1)],
        (0, 30),
        [x0, 1.0],
        rtol=1e-10,
        max_step=0.05,
    )
    a.plot(*s.y, lw=1.6)
a.plot(1, 1, "ko")
a.set(xlabel="rabbits x", ylabel="foxes y", title="Lotka-Volterra: closed orbits (conservative)")

X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
b.streamplot(X, Y, Y, -X + Y * (1 - X**2), color="0.75", density=1.1)
for s0 in ([0.1, 0.0], [3.0, 2.5]):
    s = solve_ivp(lambda t, s: [s[1], -s[0] + s[1] * (1 - s[0] ** 2)], (0, 30), s0, max_step=0.02)
    b.plot(*s.y, lw=1.6)
b.plot(0, 0, "o", mfc="w", mec="k")
b.set(xlabel="x", ylabel="y", title="Van der Pol: attracting limit cycle (dissipative)")
fig.tight_layout()
save(fig, "tour-2d")

# %% [markdown]
# ## 6. Maps: period-doubling to chaos in the logistic map
# Top: long-run values of $x_{n+1} = r x_n (1 - x_n)$. Bottom: the Lyapunov exponent
# $\lambda = \lim \frac1n \sum \ln|r(1 - 2x_n)|$. Chaos is exactly where $\lambda > 0$, and the
# periodic windows (period 3 at $r = 1 + \sqrt 8$) dip below zero.

# %%
rs = np.linspace(2.5, 4.0, 3000)
xv = 0.5 * np.ones_like(rs)
for _ in range(1000):
    xv = rs * xv * (1 - xv)
pts, lyap = [], np.zeros_like(rs)
for _ in range(400):
    xv = rs * xv * (1 - xv)
    lyap += np.log(np.abs(rs * (1 - 2 * xv)) + 1e-300)
    pts.append(xv.copy())
lyap /= 400
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
a.plot(np.repeat(rs, 400), np.array(pts).T.ravel(), ",k", alpha=0.25)
a.set(ylabel="x", title="Logistic map bifurcation diagram")
b.plot(rs, lyap, "k", lw=0.7)
b.axhline(0, color="C3", lw=0.8)
b.set(xlabel="r", ylabel=r"$\lambda$", ylim=(-2, 1))
fig.tight_layout()
save(fig, "tour-logistic")

# %% [markdown]
# ## 7. Lorenz: a strange attractor, and two trajectories $10^{-8}$ apart
# The log of the separation grows linearly with slope $\lambda_{max} \approx 0.906$ until it
# saturates at the size of the attractor. That saturation time is the predictability horizon.

# %%
sigma, rr, bb = 10.0, 28.0, 8 / 3
print(f"r_H = {lorenz_hopf_r(sigma, bb):.3f}")
s0 = solve_ivp(lorenz, (0, 30), [1, 1, 1], rtol=1e-10).y[:, -1]
n, dt = 4000, 0.01
_, A, _ = reference_solve(lorenz, s0, n, dt)
_, B, _ = reference_solve(lorenz, s0 + np.array([1e-8, 0, 0]), n, dt)
lam, _ = max_lyapunov(lorenz, s0, 200.0)
t = dt * np.arange(n + 1)
sep = np.linalg.norm(A - B, axis=1)

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*A[:2500].T, lw=0.5, color="C0")
ax.plot(*B[:2500].T, lw=0.5, color="C3", alpha=0.8)
ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz, r = 28: two runs, 1e-8 apart")
ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(t, sep, "k", lw=0.8)
ax2.semilogy(
    t, 1e-8 * np.exp(lam * t), "C3--", label=rf"$10^{{-8}} e^{{\lambda t}}$, $\lambda$ = {lam:.3f}"
)
ax2.set(
    xlabel="t",
    ylabel="|separation|",
    ylim=(1e-9, 1e2),
    title="Exponential divergence, then saturation",
)
ax2.legend(frameon=False)
fig.tight_layout()
save(fig, "tour-lorenz")

# %% [markdown]
# ## 8. Double pendulum: regular at low energy, chaotic at high energy
# Each panel shows two pendulums whose initial angles differ by $10^{-8}$ rad.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, th, label in [
    (axes[0], 0.3, "low energy (0.3 rad)"),
    (axes[1], 2.0, "high energy (2.0 rad)"),
]:
    s0 = np.array([th, th, 0.0, 0.0])
    _, A, _ = reference_solve(double_pendulum, s0, 3000, 0.01)
    _, B, _ = reference_solve(double_pendulum, s0 + [1e-8, 0, 0, 0], 3000, 0.01)
    t = 0.01 * np.arange(3001)
    ax.semilogy(t, np.abs(A[:, 1] - B[:, 1]) + 1e-16, "k", lw=0.8)
    drift = np.max(np.abs(double_pendulum_energy(A) - double_pendulum_energy(A[0])))
    ax.set(xlabel="t (s)", title=f"{label}; energy drift {drift:.0e}")
axes[0].set_ylabel(r"$|\theta_2^A - \theta_2^B|$")
fig.tight_layout()
save(fig, "tour-double-pendulum")

# %% [markdown]
# ## 9. Three bodies: a stable choreography vs a broken-up Lagrange triangle
# Both are perturbed by $10^{-3}$. The figure-eight (linearly stable) keeps its shape.
# The equal-mass Lagrange triangle (linearly unstable) dissolves into chaotic motion.

# %%
m = ics.EQUAL_MASSES
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, (name, base) in zip(
    axes[:2], [("figure-eight", ics.figure_eight()), ("Lagrange", ics.lagrange_triangle())]
):
    s0 = ics.perturb(base, rng, 1, 1e-3)[0]
    _, S, _ = reference_solve(three_body, s0, 1500, 0.02, args=(m,))
    q, _ = split_state(S)
    for i in range(3):
        ax.plot(q[:, i, 0], q[:, i, 1], lw=0.8)
    ax.set(aspect="equal", title=f"{name} + 1e-3 noise, t in [0, 30]")
    _, T, _ = reference_solve(
        three_body, s0 + 1e-8 * rng.standard_normal(12), 1500, 0.02, args=(m,)
    )
    axes[2].semilogy(
        0.02 * np.arange(1501), np.linalg.norm(S[:, :6] - T[:, :6], axis=1), label=name
    )
axes[2].set(xlabel="t", ylabel="|position separation|", title="Twins 1e-8 apart")
axes[2].legend(frameon=False)
fig.tight_layout()
save(fig, "tour-three-body")
