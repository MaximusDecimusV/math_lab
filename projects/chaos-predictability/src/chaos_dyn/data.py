"""Generate and cache trajectory datasets (data/*.npz, gitignored; regenerate from config)."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chaos_dyn import ics
from chaos_dyn.integrate import reference_solve, rk4_solve
from chaos_dyn.systems import lorenz, min_pair_distance, three_body

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Dataset:
    trajs: np.ndarray  # (M, n_steps + 1, d)
    cond: np.ndarray  # (M, c) per-trajectory constants fed to the models (masses), maybe c=0
    dt: float
    meta: dict

    def split(self, fracs=(0.8, 0.1, 0.1), seed: int = 0):
        idx = np.random.default_rng(seed).permutation(len(self.trajs))
        a = int(fracs[0] * len(idx))
        b = a + int(fracs[1] * len(idx))
        parts = idx[:a], idx[a:b], idx[b:]
        return [Dataset(self.trajs[p], self.cond[p], self.dt, self.meta) for p in parts]


def _cache_path(name: str, spec: dict) -> Path:
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    return DATA_DIR / f"{name}-{key}.npz"


def _load_or_build(name: str, spec: dict, build) -> Dataset:
    path = _cache_path(name, spec)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return Dataset(z["trajs"], z["cond"], float(z["dt"]), json.loads(str(z["meta"])))
    trajs, cond, meta = build()
    meta = {**spec, **meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, trajs=trajs, cond=cond, dt=spec["dt"], meta=json.dumps(meta))
    return Dataset(trajs, cond, spec["dt"], meta)


# ------------------------------------------------------------------------ Lorenz


@dataclass(frozen=True)
class LorenzSpec:
    n_traj: int = 4000
    n_steps: int = 500
    dt: float = 0.01
    substeps: int = 10  # RK4 at dt/10 = 1e-3: global error ~1e-10, far below model error
    seed: int = 0
    params: tuple = (10.0, 28.0, 8.0 / 3.0)


def lorenz_dataset(spec: LorenzSpec = LorenzSpec()) -> Dataset:
    """ICs drawn from the attractor: spin up a cloud of random points for 50 time units."""

    def build():
        rng = np.random.default_rng(spec.seed)
        s = rng.normal(0, 10, (spec.n_traj, 3)) + np.array([0, 0, 25])
        s = rk4_solve(lorenz, s, 5000, 0.01, spec.params)[:, -1]  # land on the attractor
        trajs = rk4_solve(lorenz, s, spec.n_steps, spec.dt, spec.params, spec.substeps)
        return trajs, np.zeros((spec.n_traj, 0)), {}

    return _load_or_build("lorenz", asdict(spec), build)


# -------------------------------------------------------------------- three-body


@dataclass(frozen=True)
class ThreeBodySpec:
    family: str = "lagrange"  # lagrange | figure_eight | pythagorean
    n_traj: int = 3000
    scale: float = 0.03  # std of the Gaussian perturbation of the base IC
    n_steps: int = 500
    dt: float = 0.02
    r_min: float = 0.05  # drop trajectories with a closer encounter than this
    rtol: float = 1e-12
    seed: int = 0


BASES = {
    "lagrange": (ics.lagrange_triangle, ics.EQUAL_MASSES),
    "figure_eight": (ics.figure_eight, ics.EQUAL_MASSES),
    "pythagorean": (ics.pythagorean, ics.PYTHAGOREAN_MASSES),
}


def _solve_one(args):
    s0, m, n_steps, dt, rtol = args
    _, S, _ = reference_solve(three_body, s0, n_steps, dt, args=(m,), rtol=rtol, atol=rtol)
    return S


def solve_many(s0s: np.ndarray, m: np.ndarray, n_steps: int, dt: float, rtol=1e-12, workers=4):
    jobs = [(s, m, n_steps, dt, rtol) for s in s0s]
    if workers == 1:
        return np.stack([_solve_one(j) for j in jobs])
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(_solve_one, jobs, chunksize=16)))


def three_body_dataset(spec: ThreeBodySpec = ThreeBodySpec(), workers: int = 4) -> Dataset:
    def build():
        base_fn, m = BASES[spec.family]
        rng = np.random.default_rng(spec.seed)
        s0s = ics.perturb(base_fn(), rng, spec.n_traj, spec.scale, m)
        trajs = solve_many(s0s, m, spec.n_steps, spec.dt, spec.rtol, workers)
        ok = min_pair_distance(trajs).min(axis=1) >= spec.r_min
        cond = np.tile(m, (int(ok.sum()), 1))
        return trajs[ok], cond, {"n_dropped_close_encounter": int((~ok).sum())}

    return _load_or_build(f"threebody-{spec.family}", asdict(spec), build)
