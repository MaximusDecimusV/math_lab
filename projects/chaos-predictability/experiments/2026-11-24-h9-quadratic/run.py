"""Fit and evaluate the H9 models. Resumable in short slices, like chaos_dyn.sweep.

python experiments/2026-11-24-h9-quadratic/run.py --budget 160 --jobs 4 --scratch ~/ckpt
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from chaos_dyn.models import n_params
from chaos_dyn.structured import PolyStep, QuadStep, VectorFieldStep
from chaos_dyn.sweep import ROOT, OutOfTime, evaluate, load_data
from mathlab_shared import rng_from_seed

EXP = Path(__file__).resolve().parent
cfg = yaml.safe_load((EXP / "config.yaml").read_text())
OUT = ROOT / "outputs" / cfg["name"]


def build(r, train):
    if r["kind"] == "poly":
        return PolyStep(train.trajs, train.cond, r["degree"]).double()
    if r["kind"] == "quad":
        torch.manual_seed(r["seed"])
        return QuadStep(train.trajs, train.cond, r["hidden"], r["depth"])
    return VectorFieldStep(train.trajs, train.dt, r["substeps"]).double()


def fit(model, r, train) -> dict:
    t0, c0 = time.time(), time.process_time()
    info = {}
    if r["kind"] == "poly":
        rng = np.random.default_rng(0)
        i, k = (
            rng.integers(0, len(train.trajs), 400_000),
            rng.integers(0, train.trajs.shape[1] - 1, 400_000),
        )
        model.fit_lstsq(train.trajs[i, k], train.trajs[i, k + 1])
    elif r["kind"] == "vf":
        info["fit_loss"] = model.fit(train.trajs)
    else:  # same loop as sweep.child / train.train_step_model
        tc = cfg["train"]
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["iters"])
        rng = rng_from_seed(r["seed"])
        trajs = torch.as_tensor(train.trajs, dtype=torch.float32)
        cond = torch.as_tensor(train.cond, dtype=torch.float32)
        M, T = trajs.shape[:2]
        tail = []
        for _ in range(tc["iters"]):
            i = torch.as_tensor(rng.integers(0, M, tc["batch"]))
            k = torch.as_tensor(rng.integers(0, T - 1, tc["batch"]))
            loss = (((model(trajs[i, k], cond[i]) - trajs[i, k + 1]) / model.d_sd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tail.append(float(loss))
        info["final_loss"] = float(np.mean(tail[-500:]))
    info.update(train_seconds=time.time() - t0, train_cpu_seconds=time.process_time() - c0)
    return info


def child(rid: str, budget: float, scratch: Path) -> None:
    deadline = time.time() + budget
    torch.set_num_threads(1)
    r = next(x for x in cfg["runs"] if x["id"] == rid)
    base, params, train, test = load_data(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    # the two vf-rk4xN runs share one float64 fit
    fit_id = rid.replace("-f32", "-f64") if r["kind"] == "vf" else rid
    wpath, ipath = OUT / f"{fit_id}.pt", OUT / f"{fit_id}.fit.json"
    model = build(r, train)
    if wpath.exists() and ipath.exists():
        model.load_state_dict(torch.load(wpath))
    elif fit_id != rid:
        print(f"{rid}: waiting for {fit_id}", flush=True)
        return
    else:
        info = fit(model, r, train)
        torch.save(model.state_dict(), wpath)
        ipath.write_text(json.dumps(info))
    info = json.loads(ipath.read_text())
    model = model.to(getattr(torch, r["dtype"]))
    sdir = scratch / cfg["name"]
    sdir.mkdir(parents=True, exist_ok=True)
    epath = sdir / f"{rid}.eval"
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
        print(f"{rid}: fitted, evaluation paused", flush=True)
        return
    res = {"run": rid, **r, "n_params": n_params(model), **info, **ev}
    (OUT / f"{rid}.json").write_text(json.dumps(res, indent=1))
    print(
        f"{rid}: done eps {res['eps_ic_median']:.2e} breakdown(10%) {res['breakdown']['0.1']['median']:.0f} lam {res['lambda_learned']['mean']}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scratch", type=Path, default=Path.home() / "ckpt")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.budget, a.scratch)
    t0 = time.time()
    todo = [r["id"] for r in cfg["runs"] if not (OUT / f"{r['id']}.json").exists()]
    print(f"{len(todo)} runs left of {len(cfg['runs'])}", flush=True)
    running = []
    while True:
        running = [p for p in running if p.poll() is None]
        left = a.budget - (time.time() - t0)
        while todo and len(running) < a.jobs and left > 45:
            running.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--child",
                        todo.pop(0),
                        "--budget",
                        str(left - 3),
                        "--scratch",
                        str(a.scratch),
                    ]
                )
            )
        if not running:
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
