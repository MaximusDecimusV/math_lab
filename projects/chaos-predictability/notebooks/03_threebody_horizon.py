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
# # Three bodies: when does a learned predictor stop knowing where the planets are?
#
# Planar three-body, G = 1, equal masses. Training family: Lagrange's equilateral triangle
# (an exact but *unstable* solution) perturbed by Gaussian noise of std 0.03. It breaks up
# chaotically within a few time units. Reference ("analytic") solution: DOP853 at
# rtol = atol = 1e-12, validated in `tests/test_systems.py`.
#
# Questions:
# 1. After how many steps does each predictor's error blow up?
# 2. How much of that is chaos (unavoidable) vs the model (fixable)?
# 3. What happens when the ICs move away from the training distribution?
#
# Config: `experiments/2026-11-20-threebody-horizon/config.yaml`.

# %%
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from chaos_dyn.data import Dataset, ThreeBodySpec, solve_many, three_body_dataset
from chaos_dyn.evaluate import (
    band,
    breakdown_step,
    three_body_drift,
    three_body_error,
    twin_baseline_three_body,
)
from chaos_dyn.lyapunov import (
    max_lyapunov_many,
    predictability_horizon,
    predictability_horizon_accumulated,
)
from chaos_dyn.models import DirectMLP, StepMLP, n_params
from chaos_dyn.systems import split_state, three_body
from chaos_dyn.train import load_or_train, predict, train_direct_model, train_step_model

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EXP = ROOT / "experiments/2026-11-20-threebody-horizon"
OUT = ROOT / "outputs/threebody-horizon"
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
SEED = cfg["seed"]

# %%
lag = three_body_dataset(ThreeBodySpec(**cfg["train_family"], seed=SEED))
fig8 = three_body_dataset(ThreeBodySpec(**cfg["contrast_family"], seed=SEED))
ood = {
    f"{o['family']} {o['scale']}": three_body_dataset(ThreeBodySpec(**{**cfg["train_family"], **o}))
    for o in cfg["ood"]
}
train, val, test = lag.split(cfg["split"], seed=SEED)
f8_train, _, f8_test = fig8.split(cfg["split"], seed=SEED)
dt, N = lag.dt, lag.trajs.shape[1] - 1
print(
    f"Lagrange: {len(train.trajs)} train / {len(test.trajs)} test"
    f" ({lag.meta['n_dropped_close_encounter']} of {cfg['train_family']['n_traj']} dropped for close encounters)"
)
print(f"{N} steps x dt {dt} = {N * dt:.0f} time units")

# %% [markdown]
# ## How chaotic is each IC? Finite-time Lyapunov exponents

# %%
ftle_path = OUT / "ftle_test.npy"
if ftle_path.exists():
    ftle = np.load(ftle_path)
else:
    ftle = max_lyapunov_many(
        three_body, test.trajs[:, 0], cfg["ftle_horizon"], args=(test.cond[0],)
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(ftle_path, ftle)
f8_ftle = max_lyapunov_many(
    three_body, f8_test.trajs[:100, 0], cfg["ftle_horizon"], args=(f8_test.cond[0],)
)
LAM = float(np.median(ftle))
print(
    f"Lagrange family: median FTLE {LAM:.2f} -> Lyapunov time {1 / LAM:.1f} "
    f"= {1 / LAM / dt:.0f} steps; whole window = {N * dt * LAM:.1f} Lyapunov times"
)
print(f"figure-eight family: median FTLE {np.median(f8_ftle):.2f}")

# %% [markdown]
# ## Train: step (autoregressive), step+unroll, direct (IC to 500 steps)

# %%
c, cu, cd = cfg["step_model"], cfg["step_model_unroll"], cfg["direct_model"]


def fit_step(data, name):
    return load_or_train(
        StepMLP(data.trajs, data.cond, c["hidden"], c["depth"]),
        OUT / f"{name}.pt",
        lambda m: train_step_model(m, data, c["iters"], c["batch"], c["lr"], seed=SEED),
    )


step = fit_step(train, "step")
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
        seed=SEED + 1,
        log_every=500,
    ),
)
direct = load_or_train(
    DirectMLP(train.trajs, train.cond, cd["hidden"], cd["depth"]),
    OUT / "direct.pt",
    lambda m: train_direct_model(m, train, cd["iters"], cd["batch"], cd["lr"], seed=SEED),
)
f8_step = fit_step(f8_train, "f8_step")
print({k: n_params(m) for k, m in [("step", step), ("direct", direct)]})

# %% [markdown]
# ## Evaluate on longer rollouts than the training windows
# Training windows are 500 steps. Extend the exact reference for each test IC to
# `eval_steps` and roll the step models out that far. The direct model only ever outputs
# its 500 trained steps. Extended references are *not* filtered for close encounters.

# %%
NL = cfg["eval_steps"]


def extend(d):
    return Dataset(solve_many(d.trajs[:, 0], d.cond[0], NL, dt), d.cond, dt, {})


long, f8_long = extend(test), extend(f8_test)
models = {"step": step, "step+unroll": unroll, "direct": direct}
preds = {k: predict(m, long) for k, m in models.items()}
errs = {k: three_body_error(p, long.trajs[:, : p.shape[1]]) for k, p in preds.items()}
eps = {k: float(np.median(e[:, 1])) for k, e in errs.items()}
print("median one-step error / system scale:", {k: f"{v:.1e}" for k, v in eps.items()})
twin = twin_baseline_three_body(long, eps["step"], seed=SEED)
twin_tiny = twin_baseline_three_body(long, 1e-8, seed=SEED)
f8_err = three_body_error(predict(f8_step, f8_long), f8_long.trajs)


def S(e):
    return np.arange(e.shape[-1])


# %% [markdown]
# ## Error vs time: learned predictors, the chaos floor, and a regular family

# %%
fig, ax = plt.subplots(figsize=(11, 5.5))
for (k, e), col in zip(errs.items(), ["C0", "C2", "C1"]):
    q = band(e)
    ax.fill_between(S(e), q[0], q[2], color=col, alpha=0.12)
    ax.semilogy(S(e), q[1], color=col, lw=2, label=f"{k} on Lagrange")
ax.semilogy(
    S(twin), np.median(twin, 0), "k--", lw=1.5, label=rf"exact twin, $\delta_0$ = {eps['step']:.0e}"
)
ax.semilogy(
    S(twin_tiny), np.median(twin_tiny, 0), "k:", lw=1.5, label=r"exact twin, $\delta_0$ = 1e-8"
)
ax.semilogy(
    S(f8_err), np.median(f8_err, 0), color="C4", lw=2, label="step on figure-eight (regular)"
)
for tol in cfg["tolerances"]:
    ax.axhline(tol, color="C3", lw=0.6, alpha=0.6)
ax.axvline(N, color="0.6", lw=0.8, ls="-.")
ax.text(N, 1e-9, " end of training window", color="0.4", fontsize=8)
top = ax.secondary_xaxis("top", functions=(lambda s: s * dt * LAM, lambda l: l / (dt * LAM)))
top.set_xlabel("Lyapunov times (median FTLE)")
ax.set(
    xlabel=f"step (dt = {dt})",
    ylabel="RMS position error / system scale",
    ylim=(1e-10, 5),
    title="Three-body: learned rollouts vs the chaos floor",
)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/threebody-horizon-error.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## Breakdown steps (median over test ICs)

# %%
table = []
for tol in cfg["tolerances"]:
    row = {"tol": tol}
    for k, e in {**errs, "twin": twin, "figure-eight step": f8_err}.items():
        row[k] = float(np.median(breakdown_step(e, tol)))
    row["formula: one-off eps"] = float(predictability_horizon(LAM, eps["step"], tol) / dt)
    row["formula: eps every step"] = float(
        predictability_horizon_accumulated(LAM, eps["step"], tol, dt) / dt
    )
    table.append(row)
print(f"{'tol':>5} " + " ".join(f"{k:>18}" for k in table[0] if k != "tol"))
for r in table:
    print(f"{r['tol']:>5} " + " ".join(f"{v:>18.0f}" for k, v in r.items() if k != "tol"))
print(f"({NL + 1} = never broke down within {NL} steps; direct is capped at {N + 1})")

# %% [markdown]
# ## Does local chaos predict when each IC breaks down?
# Each point is one test IC. Curves: eq. <horizon> with the one-step error as a one-off initial
# error, and eq. <horizon-acc> with that error injected at every step.

# %%
tol = 0.1
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, k in zip(ax, ["step", "direct"]):
    bd = breakdown_step(errs[k], tol)
    a.scatter(ftle, bd, s=10, alpha=0.6, label="test ICs")
    lam_grid = np.linspace(max(ftle.min(), 0.05), ftle.max(), 100)
    cap = errs[k].shape[1]
    a.plot(
        lam_grid,
        np.minimum(predictability_horizon(lam_grid, eps[k], tol) / dt, cap),
        "k:",
        label="one-off error",
    )
    a.plot(
        lam_grid,
        np.minimum(predictability_horizon_accumulated(lam_grid, eps[k], tol, dt) / dt, cap),
        "k--",
        label="error every step",
    )
    a.set(
        xlabel="finite-time Lyapunov exponent of IC",
        ylabel=f"breakdown step (tol {tol})",
        title=f"{k}: corr = {np.corrcoef(ftle, bd)[0, 1]:.2f}",
    )
    a.legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures/threebody-horizon-ftle.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## Change the initial conditions
# Lagrange-trained models on: bigger perturbations (0.06, 0.10), a different family
# (figure-eight), and different masses (Pythagorean 3-4-5, with masses as an input the model
# never saw vary).

# %%
sets = {
    "Lagrange 0.03 (train dist.)": test,
    **{f"Lagrange {k.split()[1]}": v for k, v in ood.items() if "lagrange" in k},
    "figure-eight 0.03": f8_test,
    "Pythagorean": ood["pythagorean 0.01"],
}
ood_rows = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for (name, d), col in zip(sets.items(), plt.cm.viridis(np.linspace(0, 0.9, len(sets)))):
    row = {}
    for k in ("step", "direct"):
        e = three_body_error(predict(models[k], d), d.trajs)
        row[k] = {
            "eps": float(np.median(e[:, 1])),
            "breakdown_tol0.1": float(np.median(breakdown_step(e, 0.1))),
        }
        if k == "step":
            ax[0].semilogy(S(e), np.median(e, 0), color=col, lw=2, label=name)
    ood_rows[name] = row
ax[0].axhline(0.1, color="C3", lw=0.6)
ax[0].set(
    xlabel="step",
    ylabel="median error / scale",
    title="step model as ICs move away from training",
    ylim=(1e-5, 10),
)
ax[0].legend(frameon=False, fontsize=8)
names = list(ood_rows)
xx = np.arange(len(names))
for j, k in enumerate(("step", "direct")):
    ax[1].bar(xx + 0.4 * j - 0.2, [ood_rows[n][k]["breakdown_tol0.1"] for n in names], 0.4, label=k)
ax[1].set_xticks(xx, [n.replace(" (", "\n(") for n in names], fontsize=8)
ax[1].set(ylabel="median breakdown step (tol 0.1)", title="breakdown vs IC distribution")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures/threebody-horizon-ood.png", dpi=110, bbox_inches="tight")
print(json.dumps(ood_rows, indent=1))

# %% [markdown]
# ## Physics: do learned rollouts conserve energy and angular momentum?
# The reference conserves them to 1e-10. Nothing in the networks enforces it.

# %%
m = test.cond[0]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k in ("step", "step+unroll", "direct"):
    dr = three_body_drift(preds[k], m)
    ax[0].semilogy(S(dr["energy"]), np.median(dr["energy"], 0), label=k)
    ax[1].semilogy(S(dr["energy"]), np.median(dr["angular_momentum"], 0), label=k)
ref_dr = three_body_drift(long.trajs, m)
ax[0].semilogy(S(ref_dr["energy"]), np.median(ref_dr["energy"], 0) + 1e-17, "k", label="reference")
ax[1].semilogy(
    S(ref_dr["energy"]), np.median(ref_dr["angular_momentum"], 0) + 1e-17, "k", label="reference"
)
ax[0].set(xlabel="step", ylabel="|E(t) - E(0)| / |E(0)|", title="energy drift (median)")
ax[1].set(xlabel="step", ylabel="|L(t) - L(0)|", title="angular momentum drift (median)")
ax[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures/threebody-horizon-invariants.png", dpi=110, bbox_inches="tight")

# %% [markdown]
# ## One trajectory, three predictions
# Orbits of the three bodies for a typical test IC. Markers show where the step model
# crosses 10% error.

# %%
i = int(np.argsort(breakdown_step(errs["step"], 0.1))[len(test.trajs) // 2])  # median IC
k_bd = int(breakdown_step(errs["step"], 0.1)[i])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
qr, _ = split_state(long.trajs[i])
for a, k in zip(ax, ("reference", "step", "direct")):
    q = qr if k == "reference" else split_state(preds[k][i])[0]
    for b in range(3):
        a.plot(qr[:, b, 0], qr[:, b, 1], color=f"C{b}", lw=0.6, alpha=0.25)
        a.plot(q[:, b, 0], q[:, b, 1], color=f"C{b}", lw=1.3)
        if k == "step" and k_bd <= NL:
            a.plot(q[k_bd, b, 0], q[k_bd, b, 1], "kx", ms=8)
    a.set(title=k if k == "reference" else f"{k} (faint = reference)", aspect="equal")
fig.suptitle(f"test IC {i}: FTLE {ftle[i]:.2f}, step-model breakdown at step {k_bd}")
fig.tight_layout()
fig.savefig(ROOT / "figures/threebody-horizon-example.png", dpi=110, bbox_inches="tight")

# %%
summary = {
    "median_ftle_lagrange": LAM,
    "median_ftle_figure_eight": float(np.median(f8_ftle)),
    "one_step_eps": eps,
    "breakdown_table": table,
    "ood": ood_rows,
    "ftle_breakdown_corr": {
        k: float(np.corrcoef(ftle, breakdown_step(errs[k], 0.1))[0, 1]) for k in errs
    },
    "energy_drift_at_step_100_median": {
        k: float(np.median(three_body_drift(p, m)["energy"][:, 100])) for k, p in preds.items()
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
