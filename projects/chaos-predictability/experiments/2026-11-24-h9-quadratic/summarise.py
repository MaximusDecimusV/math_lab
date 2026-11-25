"""H9 table, plus breakdown against one-step error for every Lorenz step model so far.

python experiments/2026-11-24-h9-quadratic/summarise.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chaos_dyn.lyapunov import predictability_horizon_accumulated
from chaos_dyn.sweep import LAM, ROOT

dt = 0.01
load = lambda name: [
    json.loads(Path(f).read_text())
    for f in sorted(glob.glob(str(ROOT / f"outputs/{name}/*.json")))
    if "-s" in f or name == "h9-quadratic"
]
h9 = [r for r in load("h9-quadratic") if "breakdown" in r]
mlp = load("h1-h2-width") + [r for r in load("h6-activations") if r["activation"] != "gelu"]
bd = lambda r, tol="0.1": r["breakdown"][tol]["median"]

print(
    f"{'run':16} {'params':>6} {'eps':>9} {'p99':>9} {'bd1%':>5} {'bd10%':>6} {'bd50%':>6} {'twin':>5} {'lyap t':>6} {'lambda':>7} {'W':>6} {'bnd':>4} {'fit s':>6}"
)
for r in h9:
    print(
        f"{r['run']:16} {r['n_params']:6d} {r['eps_ic_median']:9.2e} {r['eps_all']['p99']:9.2e} {bd(r, '0.01'):5.0f} {bd(r):6.0f} {bd(r, '0.5'):6.0f} "
        f"{r['breakdown']['0.1']['twin_median']:5.0f} {bd(r) * dt * LAM:6.1f} {r['lambda_learned']['mean']:7.4f} {r['long_run']['z_wasserstein']:6.3f} {r['long_run']['frac_bounded']:4.2f} {r['train_seconds']:6.1f}"
    )

pos = [r for r in h9 + mlp if r["eps_ic_median"] > 0]
le = np.log([r["eps_ic_median"] for r in pos])
y = np.array([bd(r) for r in pos])
slope, icpt = np.polyfit(le, y, 1)
pred = np.array(
    [predictability_horizon_accumulated(LAM, r["eps_ic_median"], 0.1, dt) / dt for r in pos]
)
summary = {
    "n_models_eps_positive": len(pos),
    "eps_range": [float(np.exp(le.min())), float(np.exp(le.max()))],
    "slope_steps_per_efold_eps": float(slope),
    "slope_theory": float(-1 / (LAM * dt)),
    "r2_bd10_vs_ln_eps": float(np.corrcoef(le, y)[0, 1] ** 2),
    "breakdown_minus_every_step_formula": {
        "mean": float((y - pred).mean()),
        "sd": float((y - pred).std()),
        "min": float((y - pred).min()),
        "max": float((y - pred).max()),
    },
    "h9": {
        r["run"]: {
            "n_params": r["n_params"],
            "eps": r["eps_ic_median"],
            "bd10": bd(r),
            "lyapunov_times": bd(r) * dt * LAM,
        }
        for r in h9
    },
}
(ROOT / "outputs/h9-quadratic/summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "h9"}, indent=1))

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(9, 5.2))
xs = np.geomspace(1e-17, 3e-3, 200)
ax.plot(
    xs,
    [predictability_horizon_accumulated(LAM, x, 0.1, dt) / dt for x in xs],
    color="0.6",
    lw=1,
    label=r"formula, $\varepsilon$ injected every step",
)
groups = [
    ("MLP, widths 32 to 1024 (GELU)", "C0", "o", [r for r in mlp if r["activation"] == "gelu"]),
    ("MLP 256, other activations", "C4", "o", [r for r in mlp if r["activation"] != "gelu"]),
    ("polynomial features, degree 2 to 6", "C1", "s", [r for r in h9 if r["kind"] == "poly"]),
    ("quad-unit network, 1 and 2 layers", "C2", "^", [r for r in h9 if r["kind"] == "quad"]),
    ("quadratic vector field in RK4", "C3", "D", [r for r in h9 if r["kind"] == "vf"]),
]
for lab, col, mk, rs in groups:
    ax.scatter(
        [max(r["eps_ic_median"], 1e-16) for r in rs],
        [bd(r) for r in rs],
        color=col,
        marker=mk,
        s=36,
        label=lab,
        zorder=3,
    )
for r in h9:
    if r["kind"] in ("poly", "vf"):
        ax.annotate(
            r["run"],
            (max(r["eps_ic_median"], 1e-16), bd(r)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="0.3",
        )
ax.axvline(1e-7, color="0.5", ls="--", lw=0.8)
ax.text(1.5e-7, 2700, "float32 resolution", fontsize=8, color="0.4")
ax.set(
    xscale="log",
    xlabel=r"one-step error $\varepsilon$ / attractor size (exact models drawn at $10^{-16}$)",
    ylabel="breakdown step (10% error)",
    title="Lorenz: breakdown against one-step error, every step model so far",
)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "figures/h9-quadratic.png", dpi=110, bbox_inches="tight")
