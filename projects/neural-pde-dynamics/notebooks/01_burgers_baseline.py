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
# # Burgers baseline
#
# Reference trajectories that the surrogate will be trained on and judged against.
# Config: `experiments/2026-11-20-burgers-baseline/config.yaml`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from neural_pde import BurgersSpectral, energy

cfg_path = Path("../experiments/2026-11-20-burgers-baseline/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
print(cfg)

# %%
s = BurgersSpectral(n=cfg["n"], nu=cfg["nu"])
u0 = np.sin(s.x) + 0.5 * np.sin(2 * s.x)
times, snaps = s.solve(u0, t_final=cfg["t_final"], dt=cfg["dt"], save_every=cfg["save_every"])

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
im = ax0.pcolormesh(s.x, times, snaps, shading="auto", cmap="RdBu_r")
ax0.set(xlabel="x", ylabel="t", title="u(x, t)")
fig.colorbar(im, ax=ax0)
ax1.plot(times, [energy(u) for u in snaps])
ax1.set(xlabel="t", ylabel="energy", title="Energy decay")
fig.tight_layout()
