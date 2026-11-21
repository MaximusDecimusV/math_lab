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
# # Lorenz warm-up: how far can a learned predictor see?
#
# A sanity check before three-body. $\lambda_{max} \approx 0.906$ is known, so we can test the
# horizon formula $t_p \approx \lambda^{-1} \ln(\Delta / \delta_0)$ (eq. <horizon>) directly.
# Config: `experiments/2026-11-20-lorenz-horizon/config.yaml`.

# %%
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from chaos_dyn.data import Dataset, LorenzSpec, lorenz_dataset
from chaos_dyn.evaluate import band, breakdown_step, lorenz_error, twin_baseline_lorenz
from chaos_dyn.integrate import rk4_solve
from chaos_dyn.lyapunov import (
    max_lyapunov_rk4,
    predictability_horizon,
    predictability_horizon_accumulated,
)
from chaos_dyn.models import DirectMLP, StepMLP, n_params
from chaos_dyn.systems import lorenz
from chaos_dyn.train import load_or_train, predict, train_direct_model, train_step_model

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EXP = ROOT / "experiments/2026-11-20-lorenz-horizon"
OUT = ROOT / "outputs/lorenz-horizon"
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
params = tuple(cfg["params"].values())
LAM = 0.906  # literature value; tests/test_lyapunov.py reproduces it

# %%
ds = lorenz_dataset(LorenzSpec(**cfg["data"], seed=cfg["seed"], params=params))
train, val, test = ds.split(cfg["split"], seed=cfg["seed"])
dt, N = ds.dt, ds.trajs.shape[1] - 1
print(
    f"{len(train.trajs)} train / {len(test.trajs)} test trajectories, {N} steps of dt={dt}"
    f" = {N * dt * LAM:.1f} Lyapunov times"
)

# %% [markdown]
# ## Train the three predictors
# * **step**: one-step map $s_k \to s_{k+1}$, rolled out autoregressively.
# * **step+unroll**: the same, fine-tuned with the loss taken over an 8-step rollout.
# * **direct**: IC to all 500 future states in one forward pass.

# %%
c = cfg["step_model"]
step = load_or_train(
    StepMLP(train.trajs, train.cond, c["hidden"], c["depth"]),
    OUT / "step.pt",
    lambda m: train_step_model(m, train, c["iters"], c["batch"], c["lr"], seed=cfg["seed"]),
)
cu = cfg["step_model_unroll"]
unroll = load_or_train(
    copy.deepcopy(step),
    OUT / "step_unroll.pt",
    lambda m: train_step_model(
        m,
        train,
        cu["iters"],
        cu["batch"],
        cu["lr"],
        unroll=cu["unroll"],
        seed=cfg["seed"] + 1,
        log_every=500,
    ),
)
cd = cfg["direct_model"]
direct = load_or_train(
    DirectMLP(train.trajs, train.cond, cd["hidden"], cd["depth"]),
    OUT / "direct.pt",
    lambda m: train_direct_model(m, train, cd["iters"], cd["batch"], cd["lr"], seed=cfg["seed"]),
)
print({k: n_params(m) for k, m in [("step", step), ("direct", direct)]})

# %% [markdown]
# ## Evaluate on longer rollouts than the training windows
# Training windows are 500 steps (about 4.5 Lyapunov times). To see *where* the step models
# break down, extend the exact reference for each test IC to `eval_steps` and roll the step
# models out that far. The direct model can only ever output its 500 trained steps.

# %%
NL = cfg["eval_steps"]
long = Dataset(rk4_solve(lorenz, test.trajs[:, 0], NL, dt, params, substeps=10), test.cond, dt, {})
models = {"step": step, "step+unroll": unroll, "direct": direct}
preds = {k: predict(m, long) for k, m in models.items()}
errs = {k: lorenz_error(p, long.trajs[:, : p.shape[1]]) for k, p in preds.items()}
eps = {k: float(np.median(e[:, 1])) for k, e in errs.items()}
print(
    "median one-step error (fraction of attractor size):", {k: f"{v:.1e}" for k, v in eps.items()}
)

# %% [markdown]
# ## Chaos baseline: exact dynamics started $\delta_0 = \varepsilon_{\rm step}$ away
# If a predictor were *perfect* apart from its first-step error, it would track this twin.
# Anything worse than the twin is error the model adds at every step.

# %%
twin = twin_baseline_lorenz(long, eps["step"], seed=cfg["seed"], params=params)
twin_tiny = twin_baseline_lorenz(long, 1e-8, seed=cfg["seed"], params=params)

t_lyap = dt * np.arange(NL + 1) * LAM
fig, ax = plt.subplots(figsize=(10, 5))
for (k, e), col in zip(errs.items(), ["C0", "C2", "C1"]):
    q = band(e)
    tl = t_lyap[: e.shape[1]]
    ax.fill_between(tl, q[0], q[2], color=col, alpha=0.15)
    ax.semilogy(tl, q[1], color=col, lw=2, label=f"{k} (median, 10-90%)")
for e, ls, lab in [
    (twin, "k--", rf"exact twin, $\delta_0$ = {eps['step']:.0e}"),
    (twin_tiny, "k:", r"exact twin, $\delta_0$ = 1e-8"),
]:
    ax.semilogy(t_lyap, np.median(e, 0), ls, lw=1.5, label=lab)
acc = eps["step"] * np.expm1(t_lyap) / np.expm1(LAM * dt)
ax.semilogy(
    t_lyap,
    acc,
    color="0.5",
    lw=0.8,
    label=r"$\varepsilon\,(e^{\lambda t}-1)/(e^{\lambda\Delta t}-1)$",
)
ax.axvline(N * dt * LAM, color="0.6", lw=0.8, ls="-.")
ax.text(N * dt * LAM, 1e-8, " end of training window", color="0.4", fontsize=8)
for tol in cfg["tolerances"]:
    ax.axhline(tol, color="C3", lw=0.6, alpha=0.6)
ax.set(
    xlabel=r"time in Lyapunov times ($\lambda t$)",
    ylabel="error / attractor size",
    ylim=(1e-9, 3),
    title="Lorenz: learned predictors vs the chaos floor",
)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/lorenz-horizon-error.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## Breakdown step vs the horizon formula

# %%
rows = []
for tol in cfg["tolerances"]:
    row = {"tol": tol}
    for k, e in {**errs, "twin": twin}.items():
        row[k] = float(np.median(breakdown_step(e, tol)))
    row["formula: one-off eps"] = predictability_horizon(LAM, eps["step"], tol) / dt
    row["formula: eps every step"] = (
        predictability_horizon_accumulated(LAM, eps["step"], tol, dt) / dt
    )
    rows.append(row)
print(f"(steps; {NL + 1} = never within the {NL}-step rollout; direct capped at {N + 1})")
print(f"{'tol':>6} " + " ".join(f"{k:>18}" for k in rows[0] if k != "tol"))
for r in rows:
    print(f"{r['tol']:>6} " + " ".join(f"{v:>18.0f}" for k, v in r.items() if k != "tol"))

# %% [markdown]
# ## Per-IC: does local instability predict breakdown?
# Finite-time Lyapunov exponent of each test IC over the first 5 time units vs its breakdown
# time (tol 0.1, step model).

# %%
ftle = max_lyapunov_rk4(lorenz, test.trajs[:, 0], N * dt, dt=0.005, args=params, transient=0.0)
bd = breakdown_step(errs["step"], 0.1) * dt
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(ftle, bd, s=8, alpha=0.6)
ax.set(
    xlabel="finite-time Lyapunov exponent of the IC",
    ylabel="step-model breakdown time (tol 0.1)",
    title=f"corr = {np.corrcoef(ftle, bd)[0, 1]:.2f}",
)
fig.tight_layout()
fig.savefig(ROOT / "figures/lorenz-horizon-ftle.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## Change the ICs: start off the attractor
# All training ICs lie on the attractor. Start from random points in a big box instead. The
# model has never seen the transient fall onto the attractor.

# %%
rng = np.random.default_rng(cfg["seed"] + 7)
s0 = rng.uniform([-30, -30, 0], [30, 30, 60], (cfg["ood_n"], 3))
ood = Dataset(
    rk4_solve(lorenz, s0, N, dt, params, substeps=10), np.zeros((cfg["ood_n"], 0)), dt, {}
)
ood_err = {k: lorenz_error(predict(m, ood), ood.trajs) for k, m in models.items()}
summary = {
    "one_step_eps": eps,
    "breakdown_steps": rows,
    "ood_off_attractor_breakdown_tol0.1": {
        k: float(np.median(breakdown_step(e, 0.1))) for k, e in ood_err.items()
    },
    "ftle_breakdown_corr": float(np.corrcoef(ftle, bd)[0, 1]),
}
print(json.dumps(summary, indent=1))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
