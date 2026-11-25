"""Resumable step-model sweeps on Lorenz for the hypotheses in notes/hypotheses.md.

    python -m chaos_dyn.sweep experiments/<folder> --budget 160 --jobs 4 --scratch ~/ckpt

Each grid point in the experiment's config.yaml is one run. Training is checkpointed to the
scratch folder and resumes on the next call, so a sweep can be driven in short time slices
(``--budget`` seconds each). A finished run leaves ``outputs/<name>/<run>.pt`` and ``.json``.

The training loop is the same as ``train.train_step_model`` with unroll = 1, except that
``torch.manual_seed(seed)`` is called before the model is built, so the seed also sets the
initial weights.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

from chaos_dyn.data import Dataset, LorenzSpec, lorenz_dataset
from chaos_dyn.evaluate import LORENZ_SCALE, breakdown_step, lorenz_error, twin_baseline_lorenz
from chaos_dyn.integrate import rk4_solve
from chaos_dyn.lyapunov import predictability_horizon, predictability_horizon_accumulated
from chaos_dyn.models import StepMLP, n_params
from chaos_dyn.systems import lorenz
from mathlab_shared import rng_from_seed

ROOT = Path(__file__).resolve().parents[2]
LAM = 0.906


def run_id(r: dict) -> str:
    return f"{r['activation']}-w{r['hidden']}-d{r['depth']}-{r['dtype']}-s{r['seed']}"


def grid_runs(cfg: dict) -> list[dict]:
    keys = list(cfg["grid"])
    runs = [dict(zip(keys, v)) for v in itertools.product(*cfg["grid"].values())]
    return runs + list(cfg.get("extra_runs", []))


def load_data(cfg: dict):
    base = yaml.safe_load((ROOT / cfg["base_config"]).read_text())
    params = tuple(base["params"].values())
    ds = lorenz_dataset(LorenzSpec(**base["data"], seed=base["seed"], params=params))
    train, _, test = ds.split(base["split"], seed=base["seed"])
    return base, params, train, test


def benettin_map(step, s0: np.ndarray, n_steps: int, dt: float, renorm_every: int, d0: float):
    """Largest Lyapunov exponent of a discrete map s -> step(s), one estimate per row of s0.

    Twin method in float64: displace by d0, iterate both, renormalise every few steps."""
    s = np.array(s0, dtype=float)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(s.shape)
    p = s + d0 * u / np.linalg.norm(u, axis=-1, keepdims=True)
    total = np.zeros(len(s))
    for k in range(1, n_steps + 1):
        s, p = step(s), step(p)
        if k % renorm_every == 0:
            delta = p - s
            dist = np.linalg.norm(delta, axis=-1, keepdims=True)
            total += np.log(dist[:, 0] / d0)
            p = s + delta * (d0 / dist)
    return total / (n_steps * dt)


class OutOfTime(Exception):
    pass


@torch.no_grad()
def evaluate(
    model: StepMLP,
    cfg: dict,
    base: dict,
    params,
    train: Dataset,
    test: Dataset,
    state: dict,
    deadline: float,
    save_state,
) -> dict:
    """All metrics for one trained model. The three long loops (test rollout, learned-map
    Lyapunov exponent, long run) keep their progress in ``state`` and raise OutOfTime at the
    deadline, so a slow evaluation can be continued by calling this again."""
    model.eval()
    dtype = next(model.parameters()).dtype
    dt = test.dt

    def f(x: np.ndarray, m=model, dt_=dtype) -> np.ndarray:
        out = []
        for j in range(0, len(x), 20_000):
            xt = torch.as_tensor(x[j : j + 20_000], dtype=dt_)
            out.append(m(xt, torch.zeros((len(xt), 0), dtype=dt_)).double().numpy())
        return np.concatenate(out)

    def check(k: int) -> None:
        if k % 20 == 0 and time.time() > deadline:
            save_state()
            raise OutOfTime

    # one-step error over every test state
    if "e1" not in state:
        S, S1 = test.trajs[:, :-1].reshape(-1, 3), test.trajs[:, 1:].reshape(-1, 3)
        state["e1"] = np.linalg.norm(f(S) - S1, axis=-1) / LORENZ_SCALE
    e1 = state["e1"]
    # rollouts 4x longer than the training window
    NL = cfg.get("eval_steps", base["eval_steps"])
    pred = state.setdefault("pred", [test.trajs[:, 0]])
    while len(pred) <= NL:
        check(len(pred))
        pred.append(f(pred[-1]))
    ref = rk4_solve(lorenz, test.trajs[:, 0], NL, dt, params, substeps=10)
    err = lorenz_error(np.stack(pred, 1), ref)
    err = np.where(np.isfinite(err), err, np.inf)
    eps = float(np.median(err[:, 1]))
    twin = twin_baseline_lorenz(Dataset(ref, test.cond, dt, {}), eps, seed=0, params=params)
    out = {
        "eps_ic_median": eps,
        "eps_all": {
            "median": float(np.median(e1)),
            "mean": float(e1.mean()),
            "p90": float(np.percentile(e1, 90)),
            "p99": float(np.percentile(e1, 99)),
            "max": float(e1.max()),
        },
        "breakdown": {},
    }
    for tol in base["tolerances"]:
        bd = breakdown_step(err, tol)
        out["breakdown"][str(tol)] = {
            "median": float(np.median(bd)),
            "p25": float(np.percentile(bd, 25)),
            "p75": float(np.percentile(bd, 75)),
            "twin_median": float(np.median(breakdown_step(twin, tol))),
            # eps is exactly 0 when the model reproduces the reference bit for bit at step 1
            "formula_one_off": float(predictability_horizon(LAM, eps, tol) / dt)
            if eps > 0
            else None,
            "formula_every_step": float(predictability_horizon_accumulated(LAM, eps, tol, dt) / dt)
            if eps > 0
            else None,
        }
    # the learned map's own Lyapunov exponent: twin method, weights evaluated in float64
    m64 = copy.deepcopy(model).double()
    ll = cfg["learned_lyapunov"]
    if "ly" not in state:
        s = np.array(test.trajs[: ll["n_ics"], 0], dtype=float)
        u = np.random.default_rng(0).standard_normal(s.shape)
        p = s + ll["d0"] * u / np.linalg.norm(u, axis=-1, keepdims=True)
        state["ly"] = {"s": s, "p": p, "total": np.zeros(len(s)), "k": 0}
    ly = state["ly"]
    while ly["k"] < ll["n_steps"]:
        check(ly["k"])
        both = f(np.concatenate([ly["s"], ly["p"]]), m64, torch.float64)
        ly["s"], ly["p"] = both[: len(both) // 2], both[len(both) // 2 :]
        ly["k"] += 1
        if ly["k"] % ll["renorm_every"] == 0:
            delta = ly["p"] - ly["s"]
            dist = np.linalg.norm(delta, axis=-1, keepdims=True)
            ly["total"] += np.log(dist[:, 0] / ll["d0"])
            ly["p"] = ly["s"] + delta * (ll["d0"] / dist)
    lam = ly["total"] / (ll["n_steps"] * dt)
    ok = np.isfinite(lam)
    out["lambda_learned"] = {
        "mean": float(lam[ok].mean()) if ok.any() else None,
        "se": float(lam[ok].std(ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else None,
        "frac_finite": float(ok.mean()),
    }
    # long-run statistics (H10)
    lr = cfg["long_run"]
    if "lg" not in state:
        s = test.trajs[: lr["n_ics"], 0]
        state["lg"] = {"s": s, "bounded": np.ones(len(s), bool), "zs": [], "k": 0}
    lg = state["lg"]
    while lg["k"] < lr["n_steps"]:
        check(lg["k"])
        lg["s"] = f(lg["s"])
        lg["bounded"] &= np.isfinite(lg["s"]).all(-1) & (np.abs(lg["s"]).max(-1) < 200)
        if lg["k"] >= lr["discard"]:
            lg["zs"].append(lg["s"][:, 2])
        lg["k"] += 1
    zs, bounded = np.stack(lg["zs"], 0), lg["bounded"]
    tail_std = zs[-2000:].std(0)
    z_true = train.trajs[::4, ::5, 2].ravel()
    zb = zs[:, bounded]
    out["long_run"] = {
        "frac_bounded": float(bounded.mean()),
        "z_wasserstein": float(wasserstein_distance(zb[::5].ravel(), z_true))
        if bounded.any()
        else None,
        "frac_collapsed": float(np.mean(tail_std[bounded] < 0.5)) if bounded.any() else None,
        "min_tail_z_std": float(np.nanmin(np.where(bounded, tail_std, np.nan)))
        if bounded.any()
        else None,
    }
    model.train()
    return out


def child(
    exp: Path, rid: str, budget: float, scratch: Path, out: Path | None = None, threads: int = 1
) -> None:
    t_start = time.time()
    deadline = t_start + budget
    torch.set_num_threads(threads)
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    r = next(x for x in grid_runs(cfg) if run_id(x) == rid)
    out_dir = out or ROOT / "outputs" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    base, params, train, test = load_data(cfg)
    tc = cfg["train"]
    dtype = getattr(torch, r["dtype"])
    torch.manual_seed(r["seed"])
    model = StepMLP(train.trajs, train.cond, r["hidden"], r["depth"], r["activation"]).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["iters"])
    rng = rng_from_seed(r["seed"])
    it, train_s, loss_tail, cpu_s = 0, 0.0, [], 0.0
    ckpt = scratch / cfg["name"] / f"{rid}.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        c = torch.load(ckpt)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        it, train_s, loss_tail = c["it"], c["train_s"], c["loss_tail"]
        cpu_s = c.get("cpu_s", c["train_s"])

    def save():
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "rng": rng.bit_generator.state,
                "it": it,
                "train_s": train_s + time.time() - t_loop,
                "cpu_s": cpu_s + time.process_time() - c_loop,
                "loss_tail": loss_tail[-500:],
            },
            ckpt,
        )

    if it < tc["iters"]:
        trajs = torch.as_tensor(train.trajs, dtype=dtype)
        cond = torch.as_tensor(train.cond, dtype=dtype)
        M, T = trajs.shape[:2]
        t_loop, c_loop = time.time(), time.process_time()
        while it < tc["iters"]:
            if it % 5 == 0 and time.time() > deadline:
                save()
                print(f"{rid}: paused at it {it}/{tc['iters']}", flush=True)
                return
            i = torch.as_tensor(rng.integers(0, M, tc["batch"]))
            k = torch.as_tensor(rng.integers(0, T - 1, tc["batch"]))
            loss = (((model(trajs[i, k], cond[i]) - trajs[i, k + 1]) / model.d_sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_tail.append(float(loss))
            it += 1
        save()
        train_s += time.time() - t_loop
        cpu_s += time.process_time() - c_loop
    epath = scratch / cfg["name"] / f"{rid}.eval"
    state = torch.load(epath) if epath.exists() else {}
    try:
        ev = evaluate(
            model,
            cfg,
            base,
            params,
            train,
            test,
            state,
            deadline - 25,
            lambda: torch.save(state, epath),
        )
    except OutOfTime:
        print(f"{rid}: trained, evaluation paused", flush=True)
        return
    res = {
        "run": rid,
        **r,
        "n_params": n_params(model),
        "train_seconds": train_s,
        "train_cpu_seconds": cpu_s,
        "final_loss": float(np.mean(loss_tail[-500:])),
        **ev,
    }
    torch.save(model.state_dict(), out_dir / f"{rid}.pt")
    (out_dir / f"{rid}.json").write_text(json.dumps(res, indent=1))
    b = res["breakdown"]["0.1"]["median"]
    print(
        f"{rid}: done eps {res['eps_ic_median']:.2e} breakdown(10%) {b:.0f} lam {res['lambda_learned']['mean']}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scratch", type=Path, default=Path.home() / "ckpt")
    ap.add_argument("--out", type=Path, help="override outputs/<name> (for smoke tests)")
    ap.add_argument("--child")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    exp = a.exp.resolve()
    if a.child:
        return child(exp, a.child, a.budget, a.scratch, a.out, a.threads)
    t0 = time.time()
    cfg = yaml.safe_load((exp / "config.yaml").read_text())
    out_dir = a.out or ROOT / "outputs" / cfg["name"]
    todo = [run_id(r) for r in grid_runs(cfg) if not (out_dir / f"{run_id(r)}.json").exists()]
    print(f"{len(todo)} runs left of {len(grid_runs(cfg))}", flush=True)
    threads = max(1, a.jobs // max(1, min(a.jobs, len(todo))))
    running: list[subprocess.Popen] = []
    while True:
        running = [p for p in running if p.poll() is None]
        left = a.budget - (time.time() - t0)
        while todo and len(running) < a.jobs and left > 45:
            cmd = [
                sys.executable,
                "-m",
                "chaos_dyn.sweep",
                str(exp),
                "--child",
                todo.pop(0),
                "--budget",
                str(left - 3),
                "--scratch",
                str(a.scratch),
                "--threads",
                str(threads),
            ]
            running.append(subprocess.Popen(cmd + (["--out", str(a.out)] if a.out else [])))
        if not running:
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
