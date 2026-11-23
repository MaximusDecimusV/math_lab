# chaos-predictability

**Question.** Chaotic systems are deterministic. Given exact initial conditions and
known masses, the future is fixed. So how far ahead can a *learned* predictor see before
its error explodes, and how much of that limit is the model's fault versus chaos itself?

**Current belief.**
- For a smooth chaotic flow (Lorenz), a learned one-step model gets within about 3 Lyapunov times of the
  limit an exact solver would reach with the same initial error, and breaks down at about 4.9 Lyapunov times.
  The horizon formula with error injected every step predicts that within 15–30%.
- For the three-body problem, the same recipe lasts only about 1.1 Lyapunov times, against about 11 for the exact solver.
  Model error during close encounters dominates (30× larger per step), and energy is not conserved.
- Out of distribution (a different IC family, different masses, or Lorenz started off the attractor), every model fails almost immediately.
- Details: `experiments/2026-11-20-{lorenz,threebody}-horizon/README.md`.

**Next step.** Close the gap between learned rollouts and the exact-twin floor.
Candidates:
- Symplectic or Hamiltonian networks that conserve energy by construction.
- Longer unroll training.
- Regularised coordinates near close encounters.

Open hypotheses about network size and activation functions are in `notes/hypotheses.md`.

## What's here

- `notes/nonlinear-dynamics.typ` is the explainer, from fixed points and bifurcations to
  Lorenz, the double pendulum and the three-body problem. It reuses the AMATH 502 examples.
- `notes/explainer.html` is an interactive version: twin-trajectory simulations of every
  system, published as a private Artifact.
- `notes/dynamics-101.html` is a beginner course, from 1D fixed points to the Lorenz attractor,
  Lyapunov exponents, Lyapunov time and the conditions for chaos, with live figures. Also a private Artifact.
- `src/chaos_dyn/`:
  - `systems.py`: Lorenz, double pendulum and planar three-body right-hand sides, plus invariants.
  - `integrate.py`: the DOP853 reference integrator (rtol 1e-12) and batched RK4.
  - `lyapunov.py`: Benettin max Lyapunov exponent and the horizon formula.
  - `ics.py`: figure-eight, Lagrange triangle, Pythagorean and random bound ICs.
  - `data.py`: cached datasets in `data/` (gitignored, rebuilt from config).
  - `models.py` / `train.py`: `StepMLP` (autoregressive) and `DirectMLP` (IC → N steps).
  - `evaluate.py`: error curves, breakdown steps, exact-twin baseline, invariant drift.
- `tests/` covers:
  - Figure-eight periodicity, and energy/momentum conservation to 1e-9.
  - Tolerance convergence of the reference solver.
  - Lorenz λ ≈ 0.906.
  - Model shapes and training.
- `notebooks/`:
  - `01_chaos_tour.py`: figures for the notes.
  - `02_lorenz_horizon.py`: warm-up.
  - `03_threebody_horizon.py`: main experiment.

## Run

```bash
uv run pytest projects/chaos-predictability
cd projects/chaos-predictability/notebooks
uv run jupytext --to notebook --execute 03_threebody_horizon.py   # ~35 min on a 4-core Intel CPU
```

Trained weights are cached in `outputs/` and datasets in `data/`. Delete those folders to retrain or regenerate.

## Environment note

This machine is an Intel Mac. PyTorch 2.2.2 is the last release with macOS x86_64 wheels,
hence the pin (and `numpy<2`). On Apple silicon or Linux, the pin can be relaxed.
