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
# # Neural PDE solvers: Laplace (original, fixed) and a Burgers PINN
#
# Trial functions build in the boundary and initial conditions, so the loss is just the PDE
# residual (notes/derivation.typ, section 2). Config: `experiments/2026-11-20-pinn-laplace-burgers/config.yaml`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from neural_pde import BurgersSpectral, cole_hopf_solution, energy, mass
from neural_pde.pinn import (
    BurgersPINN,
    LaplaceTrial,
    evaluate_on_grid,
    grid,
    laplace_exact,
    laplace_fd_solve,
    train_burgers,
    train_laplace,
)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
cfg = yaml.safe_load((ROOT / "experiments/2026-11-20-pinn-laplace-burgers/config.yaml").read_text())
plt.rcParams.update({"figure.dpi": 110})
results = {}

# %% [markdown]
# ## A. Laplace on the unit square
# The original run applied the learning rate twice (`W -= lr * (lr * grad)`, an effective step of
# 1e-6) for 10 iterations. That is, in effect, an *untrained* trial function. Compare that
# with a properly trained one, and with a finite-difference solve on the same grid.

# %%
c = cfg["laplace"]
torch.manual_seed(cfg["seed"])
untrained = LaplaceTrial(c["hidden"], c["depth"])
model = LaplaceTrial(c["hidden"], c["depth"])
model.load_state_dict(untrained.state_dict())
hist = train_laplace(model, c["n_colloc"], c["adam_iters"], c["lbfgs_iters"], c["lr"], cfg["seed"])[
    "loss"
]

n_eval = 51
xy = grid(n_eval, interior=False)
X, Y = xy[:, 0].numpy().reshape(n_eval, n_eval), xy[:, 1].numpy().reshape(n_eval, n_eval)
exact = laplace_exact(X, Y)
with torch.no_grad():
    u_nn = model(xy).numpy().reshape(n_eval, n_eval)
    u_untrained = untrained(xy).numpy().reshape(n_eval, n_eval)
s, U_fd = laplace_fd_solve(c["n_colloc"])
Sx, Sy = np.meshgrid(s, s, indexing="ij")
results["laplace"] = {
    "max_err_trained": float(np.abs(u_nn - exact).max()),
    "max_err_untrained_(original-equivalent)": float(np.abs(u_untrained - exact).max()),
    f"max_err_fd_{c['n_colloc']}x{c['n_colloc']}": float(
        np.abs(U_fd - laplace_exact(Sx, Sy)).max()
    ),
}
print(results["laplace"])

fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for a, Z, title in [
    (ax[0], u_nn, "neural solution"),
    (ax[1], exact, "exact"),
    (ax[2], np.abs(u_nn - exact), "|neural - exact|"),
]:
    im = a.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=a)
    a.set(title=title, xlabel="x", ylabel="y", aspect="equal")
ax[3].semilogy(hist)
ax[3].axvline(c["adam_iters"], color="0.5", ls="--")
ax[3].set(title="residual loss (Adam | L-BFGS)", xlabel="iteration")
fig.tight_layout()
fig.savefig(ROOT / "figures/pinn-laplace.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## B1. Burgers, $\nu = 0.1$, against the exact Cole-Hopf solution


# %%
def run_burgers(case, u0_torch):
    c = cfg[case]
    torch.manual_seed(cfg["seed"])
    m = BurgersPINN(u0_torch, c["t_final"], **c["model"])
    h = train_burgers(m, c["nu"], **c["train"], seed=cfg["seed"], verbose=True)["loss"]
    return m, h


nu = cfg["burgers_smooth"]["nu"]
pinn, hist_s = run_burgers("burgers_smooth", lambda x: 2 * nu * torch.sin(x) / (2 + torch.cos(x)))
ref = BurgersSpectral(cfg["reference"]["n"], nu)
times = np.linspace(0, cfg["burgers_smooth"]["t_final"], 21)
U_pinn = evaluate_on_grid(pinn, ref.x, times)
U_exact = np.array([cole_hopf_solution(ref.x, t, nu) for t in times])
results["burgers_smooth"] = {
    "max_abs_err_vs_cole_hopf": float(np.abs(U_pinn - U_exact).max()),
    "max_|u|": float(np.abs(U_exact).max()),
    "mass_drift": float(np.ptp([mass(u) for u in U_pinn])),
    "energy_increase_steps": int(np.sum(np.diff([energy(u) for u in U_pinn]) > 0)),
}
print(results["burgers_smooth"])

# %% [markdown]
# ## B2. Burgers, $\nu = 0.02$: a steep front, checked against the spectral solver

# %%
cs = cfg["burgers_steep"]
pinn2, hist_st = run_burgers("burgers_steep", lambda x: torch.sin(x) + 0.5 * torch.sin(2 * x))
ref2 = BurgersSpectral(cfg["reference"]["n"], cs["nu"])
u0 = np.sin(ref2.x) + 0.5 * np.sin(2 * ref2.x)
dt = cfg["reference"]["dt"]
save_every = round(cs["t_final"] / 20 / dt)
t2, U_ref2 = ref2.solve(u0, cs["t_final"], dt, save_every=save_every)
U_pinn2 = evaluate_on_grid(pinn2, ref2.x, t2)
err2 = np.abs(U_pinn2 - U_ref2)
results["burgers_steep"] = {
    "max_abs_err_vs_spectral": float(err2.max()),
    "max_abs_err_t<=0.3": float(err2[t2 <= 0.3].max()),
    "rel_L2_err_final": float(
        np.linalg.norm(U_pinn2[-1] - U_ref2[-1]) / np.linalg.norm(U_ref2[-1])
    ),
    "mass_drift": float(np.ptp([mass(u) for u in U_pinn2])),
}
print(results["burgers_steep"])

# %%
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for row, (U_p, U_r, tt, lab) in enumerate(
    [
        (U_pinn, U_exact, times, r"$\nu$=0.1 vs Cole-Hopf"),
        (U_pinn2, U_ref2, t2, r"$\nu$=0.02 vs spectral"),
    ]
):
    for a, Z, title in [
        (ax[row, 0], U_p, "PINN u(x,t)"),
        (ax[row, 2], np.abs(U_p - U_r), "|PINN - reference|"),
    ]:
        im = a.pcolormesh(
            ref.x, tt, Z, shading="auto", cmap="RdBu_r" if "PINN u" in title else "magma"
        )
        fig.colorbar(im, ax=a)
        a.set(title=f"{title}  ({lab})", xlabel="x", ylabel="t")
    for k in (0, len(tt) // 2, len(tt) - 1):
        ax[row, 1].plot(ref.x, U_r[k], "k", lw=1)
        ax[row, 1].plot(ref.x, U_p[k], "--", lw=1.5, label=f"t={tt[k]:.2f}")
    ax[row, 1].set(title="slices: reference (black) vs PINN (dashed)", xlabel="x")
    ax[row, 1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/pinn-burgers.png", dpi=110, bbox_inches="tight")
print(json.dumps(results, indent=1))
