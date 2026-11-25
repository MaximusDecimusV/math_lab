"""Collect outputs/h1-h2-width/*.json into summary.json, a table and a figure.

python experiments/2026-11-24-h1-h2-width/summarise.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from chaos_dyn.sweep import LAM, ROOT

EXP = Path(__file__).resolve().parent
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
OUT = ROOT / "outputs" / cfg["name"]
rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / "*-s*.json")))]
dt = 0.01
f32 = [r for r in rows if r["dtype"] == "float32"]
f64 = [r for r in rows if r["dtype"] == "float64"]
widths = cfg["grid"]["hidden"]
get = {
    "eps": lambda r: r["eps_ic_median"],
    "eps_p99": lambda r: r["eps_all"]["p99"],
    "bd1": lambda r: r["breakdown"]["0.01"]["median"],
    "bd10": lambda r: r["breakdown"]["0.1"]["median"],
    "bd50": lambda r: r["breakdown"]["0.5"]["median"],
    "twin10": lambda r: r["breakdown"]["0.1"]["twin_median"],
    "lam": lambda r: r["lambda_learned"]["mean"],
    "W": lambda r: r["long_run"]["z_wasserstein"],
    "bounded": lambda r: r["long_run"]["frac_bounded"],
    "cpu_min": lambda r: r.get("train_cpu_seconds", r.get("train_seconds_1thread", np.nan)) / 60,
}
table = {}
for w in widths:
    rs = [r for r in f32 if r["hidden"] == w]
    table[w] = {
        k: [float(np.mean([g(r) for r in rs])), float(np.std([g(r) for r in rs], ddof=1))]
        for k, g in get.items()
    }
    table[w]["n_params"] = rs[0]["n_params"]

lnN = np.log([r["n_params"] for r in f32])
lne = np.log([get["eps"](r) for r in f32])
bd = np.array([get["bd10"](r) for r in f32])
big = lnN >= np.log(30000)
fit = {
    "alpha_all_widths": float(-np.polyfit(lnN, lne, 1)[0]),
    "alpha_width_32_to_128": float(
        -np.polyfit(lnN[lnN < np.log(40000)], lne[lnN < np.log(40000)], 1)[0]
    ),
    "alpha_width_128_to_1024": float(-np.polyfit(lnN[big], lne[big], 1)[0]),
    "bd10_steps_per_doubling_measured": float(np.polyfit(lnN, bd, 1)[0] * np.log(2)),
    "bd10_steps_per_doubling_32_to_128": float(
        np.polyfit(lnN[lnN < np.log(40000)], bd[lnN < np.log(40000)], 1)[0] * np.log(2)
    ),
    "bd10_steps_per_doubling_128_to_1024": float(np.polyfit(lnN[big], bd[big], 1)[0] * np.log(2)),
    "bd10_steps_per_efold_eps_measured": float(np.polyfit(lne, bd, 1)[0]),
    "bd10_steps_per_efold_eps_theory": float(-1 / (LAM * dt)),
    "r2_bd10_vs_lnN": float(np.corrcoef(lnN, bd)[0, 1] ** 2),
    "r2_bd10_vs_ln_eps": float(np.corrcoef(lne, bd)[0, 1] ** 2),
}
fit["bd10_steps_per_doubling_predicted_alpha_over_lambda"] = float(
    fit["alpha_all_widths"] * np.log(2) / (LAM * dt)
)
h2 = {r["run"]: {k: float(g(r)) for k, g in get.items()} for r in f64}
(OUT / "summary.json").write_text(
    json.dumps({"by_width_float32": table, "fits": fit, "float64": h2}, indent=1)
)

print(
    f"{'width':>5} {'params':>8} {'eps':>9} {'p99':>9} {'bd1%':>5} {'bd10%':>9} {'bd50%':>5} {'twin':>5} {'gap':>4} {'lambda':>7} {'W':>6} {'cpu min':>7}"
)
for w in widths:
    t = table[w]
    print(
        f"{w:5d} {t['n_params']:8d} {t['eps'][0]:9.2e} {t['eps_p99'][0]:9.2e} {t['bd1'][0]:5.0f} {t['bd10'][0]:5.0f}±{t['bd10'][1]:<3.0f} {t['bd50'][0]:5.0f} {t['twin10'][0]:5.0f} {t['twin10'][0] - t['bd10'][0]:4.0f} {t['lam'][0]:7.4f} {t['W'][0]:6.3f} {t['cpu_min'][0]:7.1f}"
    )
for k, v in h2.items():
    print(
        f"{k}: eps {v['eps']:.2e} p99 {v['eps_p99']:.2e} bd10 {v['bd10']:.0f} twin {v['twin10']:.0f} lambda {v['lam']:.4f} cpu min {v['cpu_min']:.0f}"
    )
print(json.dumps(fit, indent=1))

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
N = np.array([table[w]["n_params"] for w in widths])
ax.scatter(np.exp(lnN), np.exp(lne), color="C0", s=30, label="float32, 3 seeds")
ax.scatter(
    [r["n_params"] for r in f64],
    [get["eps"](r) for r in f64],
    color="C3",
    marker="s",
    s=40,
    label="float64",
)
ax.plot(N, [table[w]["eps"][0] for w in widths], color="C0", lw=1)
ax.axhline(1e-7, color="0.5", ls="--", lw=0.8)
ax.text(N[0], 1.15e-7, "float32 resolution (H2)", fontsize=8, color="0.4")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="parameters",
    ylabel=r"one-step error $\varepsilon$ / attractor size",
    title=r"$\varepsilon$ flattens two decades above the float32 floor",
)
ax.legend(frameon=False, fontsize=8)
ax2.scatter(np.exp(lnN), bd, color="C0", s=30, label="step model, float32")
ax2.scatter(
    [r["n_params"] for r in f64],
    [get["bd10"](r) for r in f64],
    color="C3",
    marker="s",
    s=40,
    label="float64",
)
ax2.plot(N, [table[w]["bd10"][0] for w in widths], color="C0", lw=1)
ax2.plot(
    N,
    [table[w]["twin10"][0] for w in widths],
    "k--",
    lw=1,
    label=r"exact twin, $\delta_0=\varepsilon$",
)
a256 = table[256]
ax2.plot(
    N,
    [a256["bd10"][0] - np.log(table[w]["eps"][0] / a256["eps"][0]) / (LAM * dt) for w in widths],
    color="0.5",
    lw=1,
    label=r"predicted from $\varepsilon$ alone (anchored at 256)",
)
ax2.set(
    xscale="log",
    xlabel="parameters",
    ylabel="breakdown step (10% error)",
    title="H1: breakdown vs network size",
)
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/h1-h2-width.png", dpi=110, bbox_inches="tight")
