"""Exploratory: what in the one-step error predicts breakdown, if not its median size?

For every saved model in outputs/h1-h2-width and outputs/h6-activations, take the one-step
error vector e(s) = model(s) - exact(s) on the test trajectories and measure
  - its component along the flow direction F(s) and along the local unstable direction (the
    leading tangent vector, carried along each trajectory by a renormalised twin; the first
    150 steps are discarded as alignment),
  - how coherent it stays along a trajectory (autocorrelation at lags 1, 10, 50).
Caveat found on the first run: on Lorenz the leading tangent vector is nearly parallel to
the flow (median |cos| about 0.9), so the two projections carry the same information.
Writes outputs/h1-h2-width/error_structure.json.

    python experiments/2026-11-24-h1-h2-width/error_structure.py
"""

import glob
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from chaos_dyn.evaluate import LORENZ_SCALE
from chaos_dyn.integrate import rk4_solve
from chaos_dyn.models import StepMLP
from chaos_dyn.sweep import ROOT, load_data
from chaos_dyn.systems import lorenz

EXP = Path(__file__).resolve().parent
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
base, params, train, test = load_data(cfg)
dt = test.dt
X = test.trajs  # (400, 501, 3)
M, T1, _ = X.shape

# leading tangent direction along each test trajectory (renormalised twin, exact dynamics)
d0 = 1e-7
u = np.random.default_rng(0).standard_normal((M, 3))
p = X[:, 0] + d0 * u / np.linalg.norm(u, axis=-1, keepdims=True)
V = np.zeros_like(X)
for k in range(T1):
    delta = p - X[:, k]
    delta /= np.linalg.norm(delta, axis=-1, keepdims=True)
    V[:, k] = delta
    p = rk4_solve(lorenz, X[:, k] + d0 * delta, 1, dt, params, substeps=10)[:, -1]
F = lorenz(0.0, X, *params)
Fh = F / np.linalg.norm(F, axis=-1, keepdims=True)
BURN = 150
sl = slice(BURN, T1 - 1)

rows = []
for f in sorted(
    glob.glob(str(ROOT / "outputs/h1-h2-width/*-s*.json"))
    + glob.glob(str(ROOT / "outputs/h6-activations/*-s*.json"))
):
    r = json.loads(Path(f).read_text())
    if "h6-activations" in f and r["activation"] == "gelu":
        continue  # same three models as the width-256 runs
    dtype = getattr(torch, r["dtype"])
    m = StepMLP(train.trajs, train.cond, r["hidden"], r["depth"], r["activation"]).to(dtype)
    m.load_state_dict(torch.load(f.replace(".json", ".pt")))
    m.eval()
    S = X[:, :-1].reshape(-1, 3)
    with torch.no_grad():
        out = np.concatenate(
            [
                m(
                    torch.as_tensor(S[j : j + 20000], dtype=dtype),
                    torch.zeros((len(S[j : j + 20000]), 0), dtype=dtype),
                )
                .double()
                .numpy()
                for j in range(0, len(S), 20000)
            ]
        )
    e = (out.reshape(M, T1 - 1, 3) - X[:, 1:]) / LORENZ_SCALE
    e = e[:, BURN:]
    n2 = (e**2).sum(-1)
    eu = (e * V[:, sl]).sum(-1)  # along the unstable direction
    ef = (e * Fh[:, sl]).sum(-1)  # along the flow
    ac = lambda a, lag: float((a[:, lag:] * a[:, :-lag]).mean() / (a * a).mean())
    acv = lambda lag, e=e, n2=n2: float((e[:, lag:] * e[:, :-lag]).sum(-1).mean() / n2.mean())
    # coherent sum of the unstable component over one Lyapunov time (110 steps), per trajectory
    L = 110
    cs = np.abs(
        np.add.reduceat(
            eu[:, : (eu.shape[1] // L) * L], np.arange(0, (eu.shape[1] // L) * L, L), axis=1
        )
    )
    rows.append(
        {
            "run": r["run"],
            "activation": r["activation"],
            "hidden": r["hidden"],
            "dtype": r["dtype"],
            "seed": r["seed"],
            "n_params": r["n_params"],
            "bd10": r["breakdown"]["0.1"]["median"],
            "eps": r["eps_ic_median"],
            "eps_p99": r["eps_all"]["p99"],
            "lambda": r["lambda_learned"]["mean"],
            "rms": float(np.sqrt(n2.mean())),
            "frac_unstable": float((eu**2).mean() / n2.mean()),
            "frac_flow": float((ef**2).mean() / n2.mean()),
            "rms_unstable": float(np.sqrt((eu**2).mean())),
            "median_abs_unstable": float(np.median(np.abs(eu))),
            "p99_abs_unstable": float(np.percentile(np.abs(eu), 99)),
            "autocorr_unstable": {str(lag): ac(eu, lag) for lag in (1, 10, 50)},
            "autocorr_vector": {str(lag): acv(lag) for lag in (1, 10, 50)},
            "coherent_unstable_sum_1lyap": float(np.median(cs)),
            "mean_unstable": float(eu.mean()),
        }
    )

y = np.array([r["bd10"] for r in rows])


def r2(*cols):
    A = np.column_stack([np.ones(len(y)), *cols])
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - res.var() / y.var())


g = lambda k: np.log([r[k] for r in rows])
fits = {
    "n_models": len(rows),
    "r2_ln_eps_median": r2(g("eps")),
    "r2_ln_eps_p99": r2(g("eps_p99")),
    "r2_ln_rms": r2(g("rms")),
    "r2_ln_rms_unstable": r2(g("rms_unstable")),
    "r2_ln_median_abs_unstable": r2(g("median_abs_unstable")),
    "r2_ln_p99_abs_unstable": r2(g("p99_abs_unstable")),
    "r2_ln_coherent_unstable_sum": r2(g("coherent_unstable_sum_1lyap")),
    "r2_ln_eps_plus_ln_coherent": r2(g("eps"), g("coherent_unstable_sum_1lyap")),
    "slope_ln_coherent": float(np.polyfit(g("coherent_unstable_sum_1lyap"), y, 1)[0]),
    "slope_ln_eps": float(np.polyfit(g("eps"), y, 1)[0]),
    "slope_theory": float(-1 / (0.906 * dt)),
}
(ROOT / "outputs/h1-h2-width/error_structure.json").write_text(
    json.dumps({"fits": fits, "models": rows}, indent=1)
)
print(
    f"{'run':28} {'bd10':>5} {'eps':>8} {'fr_u':>5} {'fr_fl':>5} {'ac1':>5} {'ac10':>5} {'ac50':>5} {'coh_sum':>8}"
)
for r in rows:
    a = r["autocorr_unstable"]
    print(
        f"{r['run']:28} {r['bd10']:5.0f} {r['eps']:8.2e} {r['frac_unstable']:5.2f} {r['frac_flow']:5.2f} {a['1']:5.2f} {a['10']:5.2f} {a['50']:5.2f} {r['coherent_unstable_sum_1lyap']:8.2e}"
    )
print(json.dumps(fits, indent=1))
