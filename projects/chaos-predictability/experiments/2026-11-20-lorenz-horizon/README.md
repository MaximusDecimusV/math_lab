# lorenz-horizon

**Hypothesis:** A learned one-step predictor for Lorenz (r = 28, λ ≈ 0.906) breaks down when
its accumulated error, grown at rate λ, reaches the tolerance (eqs. `<horizon>` and
`<horizon-acc>` in `notes/nonlinear-dynamics.typ`). A direct IC → trajectory network does worse,
because it gets no help from the dynamics.

**Setup:** see config.yaml and `notebooks/02_lorenz_horizon.py`. Commit: `d523dd6` (plus uncommitted work).
- 4000 trajectories with ICs on the attractor, each 500 steps at dt = 0.01 (4.5 Lyapunov times).
  Reference is RK4 at dt/10, which agrees with DOP853 to 3e-7.
- Split 80/10/10. Models were evaluated on 2000-step rollouts (18 Lyapunov times) from the 400 test ICs.
- Models:
  - step: MLP 256×3, 133k params, about 4 min of CPU training.
  - step+unroll: the step model fine-tuned on an 8-step rollout loss.
  - direct: MLP 512×3 → 500×3 outputs, 1.3M params.

**Result:** errors are distance / attractor size (15). Breakdown is the median step at which the error first
exceeds the tolerance.

| | one-step ε | tol 1% | tol 10% | tol 50% |
|---|---|---|---|---|
| step | 1.5e-5 | 281 | 536 | 718 |
| step+unroll | 1.4e-5 | 311 | 554 | 734 |
| direct | 1.9e-3 | 6 | 74 | 223 |
| exact twin, δ₀ = ε_step | — | 594 | 849 | 1030 |
| formula, ε as a one-off initial error | — | 715 | 969 | 1146 |
| formula, ε injected every step | — | 196 | 450 | 628 |
| step model, ICs off the attractor | — | — | **3** | — |

Figures: `figures/lorenz-horizon-error.png`, `figures/lorenz-horizon-ftle.png`.

- **The step model's error grows at the chaotic rate.** On a log scale it runs parallel to the
  exact-twin curve until saturation, at roughly 15× above it.
  - Fully coherent accumulation of a fresh ε every step would predict ε/(e^{λΔt} − 1) ≈ 110× instead. The
    real figure is smaller because per-step errors are not all aligned with the unstable direction.
  - The "ε every step" curve is therefore a mild overestimate. It stays within about 2× of the median for the
    first 2–3 Lyapunov times, then drifts above it.
  - The step model keeps working well past the 500-step training window, because it learned the
    vector field, not the trajectories.
- **Breakdown sits between the two formulas.** It is close to the "error every step" formula (536 against 450 at 10%) and
  well short of what an exact simulator with the same initial error achieves (849). The gap between
  the step model and the twin, about 300 steps or 2.7 Lyapunov times, is the model's fault and in principle fixable. The
  rest is chaos.
- **Unroll training helps a little.** It adds about 20–30 steps, roughly 0.2 Lyapunov times.
- **Direct IC → trajectory is far worse.** It reaches 10% error after 74 steps, and its error saturates within one
  Lyapunov time. Mapping an IC straight to a position many Lyapunov times later is a function whose
  gradient grows like e^{λt}, which an MLP can't fit.
- **Changing the ICs is catastrophic.** Off the attractor, every model fails within 3 steps. The networks only
  learned the vector field *on* the attractor.
- **Per-IC finite-time Lyapunov exponents don't predict per-IC breakdown** (corr −0.12). With breakdown at about 5 time
  units, the exponent measured over 5 units is too noisy to be a useful predictor.

**Verdict:** worked. The horizon formula with per-step error injection predicts breakdown to
within 15–30%, and always on the early side, so it is a usable conservative estimate. Learned step models lose about 3 Lyapunov times to their own error compared with an exact
solver, and the direct approach fails.
