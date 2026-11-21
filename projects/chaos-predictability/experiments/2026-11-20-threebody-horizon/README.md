# threebody-horizon

**Hypothesis:** As in the Lorenz warm-up, a learned one-step predictor for the planar
three-body problem should break down near the chaos-set horizon (eq. `<horizon-acc>`). It
should break down earlier for ICs further from the training distribution. The direct
IC → trajectory network should do worse than the step model.

**Setup:** see config.yaml and `notebooks/03_threebody_horizon.py`. Commit: `d523dd6` (plus uncommitted work).
- Equal masses, G = 1. ICs are the Lagrange equilateral triangle (an exact, unstable solution) plus
  Gaussian noise of std 0.03 on all 12 coordinates, moved to the centre-of-mass frame.
- Reference is DOP853 at rtol = atol = 1e-12: energy drift 2e-11, and the 1e-12 and 1e-13 tolerances agree to 1e-7 (tests).
- 500 steps at dt = 0.02 for training. 65 of 3000 trajectories were dropped for close encounters (r < 0.05).
  Split 2348 / 293 / 294.
- Test-time rollouts run 1500 steps (30 time units). The extended references are not filtered.
- Median finite-time Lyapunov exponent (FTLE) over 10 units: **0.52** for Lagrange and 0.45 for the figure-eight family.
  So the Lyapunov time is about 1.9 time units, or 95 steps.
- Models use the same architectures and budgets as Lorenz. The step model is an MLP 256×3 with 20k iterations. Masses are an extra input.
- About 34 min in total on a 4-core CPU.

**Result:** error is RMS position error / system scale. Breakdown is the median step at which the error first exceeds the tolerance.

| | one-step ε | tol 1% | tol 10% | tol 50% |
|---|---|---|---|---|
| step | 7.7e-5 | 36 | **105** | 203 |
| step+unroll | 1.1e-4 | 35 | 112 | 215 |
| direct | 6.3e-5 | 108 | **255** | 447 |
| exact twin, δ₀ = ε_step | — | 662 | 1112 | never (≥1501) |
| formula, ε every step | — | 30 | 249 | 403 |
| step model trained on figure-eight, tested on figure-eight | 2.0e-4† | 48 | 123 | 275 |

One-step ε in the table is the error of the first step. † For the figure-eight model, the median is taken over every step of every
test trajectory. For the Lagrange step model the same all-steps median is 1.5e-4.

How breakdown at 10% changes as the ICs move away from training (Lagrange-trained models):

| test ICs | step | direct |
|---|---|---|
| Lagrange, noise 0.03 (training distribution) | 105 | 255 |
| Lagrange, noise 0.06 | 75 | 177 |
| Lagrange, noise 0.10 | 57 | 124 |
| figure-eight, noise 0.03 | 10 | 10 |
| Pythagorean 3-4-5 | 21 | 29 |

Figures: `figures/threebody-horizon-{error,ftle,ood,invariants,example}.png`.

- **The learned models are nowhere near the chaos floor.** An exact solver started 7.7e-5 off (the step model's
  one-step error) stays within 10% for about 1100 steps, or about 11 Lyapunov times. The step model lasts 105 steps,
  about 1.1 Lyapunov times. Unlike Lorenz, the error curve does not run parallel to the twin. It jumps from 1e-4 to 1e-2 in the
  first 40 steps, far faster than e^{λt} allows. This is model error, not chaos.
  - Per-IC FTLE has no correlation with breakdown (corr ≈ 0.00).
  - A step model trained on the near-regular figure-eight family breaks down at about the same step (123).
    How chaotic the family is barely matters when model error dominates.
- **Cause, measured: close approaches.** Gravity's 1/r² force makes the one-step map very stiff
  when two bodies come close. The step model's one-step error rises steadily as they close in:

  | closest pair distance | median one-step error | samples |
  |---|---|---|
  | > 0.8 | 1.2e-4 | 118,753 |
  | 0.4–0.8 | 3.0e-4 | 22,526 |
  | 0.2–0.4 | 6.4e-4 | 4,608 |
  | 0.1–0.2 | 1.5e-3 | 931 |
  | < 0.1 | 3.3e-3 | 182 |

  Close approaches are rare in the training data and about 30× harder per step. Every encounter
  injects a large kick that chaos then amplifies. Lorenz has a smooth quadratic vector field and
  no such hot spots, which is why its learned model tracked the chaos floor.
- **Direct beats step here** (255 against 105 steps), the opposite of Lorenz. The direct model never compounds its own
  errors through encounters, and within 5 Lyapunov times the IC → trajectory map is still learnable. It
  also only exists inside the 500-step window it was trained on.
- **Unroll training does nothing measurable** (112 against 105).
- **Changing the ICs degrades the models steadily, then sharply.**
  - Doubling the perturbation costs about 30% of the horizon, and tripling it about 45%.
  - A different family (figure-eight) or different masses (Pythagorean) breaks both models within 10–30 steps,
    about 0.1–0.3 Lyapunov times.
- **The physics is not conserved.** Energy drifts 3% by step 100 and is O(1) by step 600, against 1e-11 for the
  reference. Angular momentum behaves the same way. Nothing in an unconstrained MLP knows about the invariants.

**Verdict:** failed because model error, not chaos, sets the horizon. The learned step model
loses about 10 of the 11 Lyapunov times that an exact solver with the same initial error would
achieve. Next experiments:
- A Hamiltonian or symplectic network, so energy and angular momentum are conserved by construction.
- Training data weighted toward close encounters, or a regularised (Levi-Civita / KS) coordinate system.
- A predictor of accelerations instead of state updates, integrated with a symplectic step.
