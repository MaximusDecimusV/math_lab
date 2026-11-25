# chaos-predictability

**Question.** Chaotic systems are deterministic. Given exact initial conditions and
known masses, the future is fixed. So how far ahead can a *learned* predictor see before
its error explodes, and how much of that limit is the model's fault versus chaos itself?

**Current belief.**
- For a smooth chaotic flow (Lorenz), a learned one-step model gets within about 3 Lyapunov times of the
  limit an exact solver would reach with the same initial error, and breaks down at about 4.9 Lyapunov times.
  The horizon formula with error injected every step predicts that within 15–30%.
- (2026-11-24) On Lorenz the horizon is the one-step error ε through a logarithm, and almost nothing else: 46
  models spanning ε from 5e-12 to 1e-3 fall on one line (R² 0.99). Width, activation and the learned exponent do
  not move a model off it. MLPs trained with Adam stall at ε near 1e-5 (5 Lyapunov times) whatever their size.
  Models built on the equations' structure reach 9 to 31 Lyapunov times, and float32 caps everything near 10.
- For the three-body problem, the same recipe lasts only about 1.1 Lyapunov times, against about 11 for the exact solver.
  Model error during close encounters dominates (30× larger per step), and energy is not conserved.
- Out of distribution (a different IC family, different masses, or Lorenz started off the attractor), every model fails almost immediately.
- Details: `experiments/2026-11-20-{lorenz,threebody}-horizon/README.md`.

**Next step.** Close the gap between learned rollouts and the exact-twin floor.
Candidates:
- Symplectic or Hamiltonian networks that conserve energy by construction.
- Longer unroll training.
- Regularised coordinates near close encounters.

## Hypotheses to test

The horizon depends on a model's one-step error ε only through a logarithm: a 10× smaller ε buys about
2.3 Lyapunov times. Most of these claims ask how network size and activation functions change ε, and
whether ε is the right thing to measure. Full predictions, tests and falsifiers are in `notes/hypotheses.md`.

| | Hypothesis | Status |
|---|---|---|
| H0 | Your theory, still to be written down | — |
| **Size** | | |
| H1 | Breakdown grows with log(parameters): each doubling adds a fixed number of steps (about 40–80 on Lorenz) | log growth with the predicted slope α/λ, but α ≈ 0.2: about 17 steps per doubling, 10 past 34k params |
| H2 | Float32 caps the Lorenz horizon near 1000 steps (about 9 Lyapunov times) at any network size | supported: an exact model lasts 1148 steps in float32 and 3441 in float64. MLPs never get there (ε floors at 1.3e-5) |
| H3 | For three-body, more close-encounter data beats more neurons | untested |
| H4 | The direct model needs parameters that grow exponentially with the horizon, so t_p ≈ ln N / (λD) | untested |
| H5 | At a fixed parameter count, width beats depth beyond about 3 layers | untested |
| **Activations** | | |
| H6 | Smooth activations (GELU, SiLU, tanh) beat ReLU at the same one-step error | falsified on Lorenz: ε sets breakdown to within ±45 steps; ReLU's exponent is 1.7% low |
| H7 | The learned map's own Lyapunov exponent predicts breakdown better than ε alone | falsified on 32 models: R² 0.79 → 0.80; the p99 of ε does better (0.84) |
| H8 | Sine activations help during close encounters but hurt on unfamiliar starting conditions | untested |
| H9 | A tiny quadratic network matches the 133k-parameter MLP on Lorenz, because its equations are quadratic | falsified as stated (a quadratic step map is 60× worse; two quad layers tie). A quadratic vector field in RK4 is exact: 31 Lyapunov times with 30 params |
| H10 | Long-run attractor statistics survive breakdown only with smooth activations | ReLU half falsified on Lorenz: every activation keeps the climate |

Suggested order: ~~H6~~, ~~H1 + H2~~, ~~H7~~, ~~H10~~, ~~H9 (Lorenz)~~, then the three-body set (H3, H8, H4, H5, and H9's second half).

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
  - `sweep.py`: resumable step-model sweeps over a config grid, with learned-map λ and long-run statistics.
  - `structured.py`: polynomial-feature, quad-unit and RK4-wrapped vector-field step models (H9).
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
