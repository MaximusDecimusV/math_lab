"""Collect outputs/h6-activations/*.json into summary.json, a table and a figure.

python experiments/2026-11-24-h6-activations/summarise.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import wasserstein_distance

from chaos_dyn.integrate import rk4_solve
from chaos_dyn.sweep import LAM, ROOT, benettin_map, load_data
from chaos_dyn.systems import lorenz

EXP = Path(__file__).resolve().parent
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
OUT = ROOT / "outputs" / cfg["name"]
rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / "*-s*.json")))]
base, params, train, test = load_data(cfg)
dt = test.dt

# reference values from the exact map, with the same estimators and the same ICs
ref_path = OUT / "reference.json"
if ref_path.exists():
    ref = json.loads(ref_path.read_text())
else:
    ll, lr = cfg["learned_lyapunov"], cfg["long_run"]
    exact = lambda s: rk4_solve(lorenz, s, 1, dt, params, substeps=10)[..., -1, :]
    lam = benettin_map(
        exact, test.trajs[: ll["n_ics"], 0], ll["n_steps"], dt, ll["renorm_every"], ll["d0"]
    )
    long = rk4_solve(lorenz, test.trajs[: lr["n_ics"], 0], lr["n_steps"], dt, params, substeps=10)
    z = long[:, lr["discard"] + 1 :: 5, 2].ravel()
    ref = {
        "lambda_exact_map": float(lam.mean()),
        "lambda_exact_map_se": float(lam.std(ddof=1) / np.sqrt(len(lam))),
        "z_wasserstein_exact": float(wasserstein_distance(z, train.trajs[::4, ::5, 2].ravel())),
    }
    ref_path.write_text(json.dumps(ref, indent=1))
lam_ref = ref["lambda_exact_map"]

acts = cfg["grid"]["activation"]
get = {
    "eps": lambda r: r["eps_ic_median"],
    "eps_p99": lambda r: r["eps_all"]["p99"],
    "eps_max": lambda r: r["eps_all"]["max"],
    "bd1": lambda r: r["breakdown"]["0.01"]["median"],
    "bd10": lambda r: r["breakdown"]["0.1"]["median"],
    "bd50": lambda r: r["breakdown"]["0.5"]["median"],
    "twin10": lambda r: r["breakdown"]["0.1"]["twin_median"],
    "formula10": lambda r: r["breakdown"]["0.1"]["formula_every_step"],
    "lam": lambda r: r["lambda_learned"]["mean"],
    "W": lambda r: r["long_run"]["z_wasserstein"],
    "bounded": lambda r: r["long_run"]["frac_bounded"],
    "train_s": lambda r: r["train_seconds_1thread"],
}
table = {}
for a in acts:
    rs = [r for r in rows if r["activation"] == a]
    table[a] = {
        k: [float(np.mean([g(r) for r in rs])), float(np.std([g(r) for r in rs], ddof=1))]
        for k, g in get.items()
    }
    table[a]["n"] = len(rs)

# Does eps alone set breakdown? Compare each activation with the GELU line of slope -1/(lam dt).
g = table["gelu"]
for a in acts:
    t = table[a]
    pred = g["bd10"][0] - np.log(t["eps"][0] / g["eps"][0]) / (LAM * dt)
    t["bd10_predicted_from_gelu_eps_only"] = float(pred)
    t["bd10_residual"] = float(t["bd10"][0] - pred)
    t["gap_to_twin10"] = float(t["twin10"][0] - t["bd10"][0])

# pooled regressions (preview of H7; 15 runs in 5 tight clusters, so treat as descriptive)
y = np.array([get["bd10"](r) for r in rows])


def r2(*cols):
    X = np.column_stack([np.ones(len(y)), *cols])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - res.var() / y.var())


le = np.log([get["eps"](r) for r in rows])
dl = np.abs(np.array([get["lam"](r) for r in rows]) - lam_ref)
reg = {
    "r2_ln_eps_median": r2(le),
    "r2_ln_eps_p99": r2(np.log([get["eps_p99"](r) for r in rows])),
    "r2_ln_eps_max": r2(np.log([get["eps_max"](r) for r in rows])),
    "r2_ln_eps_median_plus_abs_dlambda": r2(le, dl),
    "r2_ln_eps_median_plus_ln_p99": r2(le, np.log([get["eps_p99"](r) for r in rows])),
    "slope_bd10_per_ln_eps": float(np.polyfit(le, y, 1)[0]),
    "slope_theory": float(-1 / (LAM * dt)),
}
(OUT / "summary.json").write_text(
    json.dumps({"reference": ref, "by_activation": table, "pooled": reg}, indent=1)
)

print(
    f"reference: lambda {lam_ref:.4f} +- {ref['lambda_exact_map_se']:.4f}, z-Wasserstein of exact rollouts {ref['z_wasserstein_exact']:.3f}"
)
print(
    f"{'act':5} {'eps':>9} {'p99':>9} {'bd1%':>5} {'bd10%':>9} {'bd50%':>5} {'twin':>5} {'gap':>4} {'eps-only':>8} {'resid':>6} {'lambda':>15} {'W':>6} {'bounded':>7}"
)
for a in acts:
    t = table[a]
    print(
        f"{a:5} {t['eps'][0]:9.2e} {t['eps_p99'][0]:9.2e} {t['bd1'][0]:5.0f} {t['bd10'][0]:5.0f}±{t['bd10'][1]:<3.0f} {t['bd50'][0]:5.0f} {t['twin10'][0]:5.0f}"
        f" {t['gap_to_twin10']:4.0f} {t['bd10_predicted_from_gelu_eps_only']:8.0f} {t['bd10_residual']:+6.0f} {t['lam'][0]:.4f}±{t['lam'][1]:.4f} {t['W'][0]:6.3f} {t['bounded'][0]:7.2f}"
    )
print(json.dumps(reg, indent=1))

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.geomspace(1.2e-5, 9e-5, 50)
ax.plot(
    xs,
    g["bd10"][0] - np.log(xs / g["eps"][0]) / (LAM * dt),
    color="0.6",
    lw=1,
    label=r"slope $-1/(\lambda\Delta t)$ through GELU",
)
for i, a in enumerate(acts):
    rs = [r for r in rows if r["activation"] == a]
    ax.scatter(
        [get["eps"](r) for r in rs],
        [get["bd10"](r) for r in rs],
        color=f"C{i}",
        s=40,
        label=a,
        zorder=3,
    )
    ax.scatter(
        [get["eps"](r) for r in rs], [get["twin10"](r) for r in rs], color=f"C{i}", s=40, marker="x"
    )
    ax2.errorbar(
        [get["lam"](r) for r in rs],
        [get["bd10"](r) - table[a]["bd10_predicted_from_gelu_eps_only"] for r in rs],
        xerr=[r["lambda_learned"]["se"] for r in rs],
        fmt="o",
        color=f"C{i}",
        label=a,
    )
ax.set(
    xscale="log",
    xlabel=r"one-step error $\varepsilon$ / attractor size",
    ylabel="breakdown step (10% error)",
    title="H6: breakdown vs one-step error (x = exact twin)",
)
ax.legend(frameon=False, fontsize=8)
ax2.axvline(lam_ref, color="k", lw=0.8, ls="--")
ax2.axhline(0, color="0.6", lw=0.8)
ax2.set(
    xlabel=r"Lyapunov exponent of the learned map (dashed: exact map)",
    ylabel=r"breakdown minus $\varepsilon$-only prediction (steps)",
    title="Residual vs learned exponent",
)
fig.tight_layout()
fig.savefig(ROOT / "figures/h6-activations.png", dpi=110, bbox_inches="tight")
