# Journal

Newest first. One entry per session. Record what you tried, what happened, and
what you now believe. Dead ends are the most valuable entries.

## 2026-11-24

- **chaos-predictability: H6 (activations) and H10 (long-run statistics) on Lorenz**
  (`experiments/2026-11-24-h6-activations`). Run by Claude in a Linux VM on this Mac, driven remotely.
  - 5 activations × 3 seeds, 256×3 step MLP, same budget as 11-20. New: `activation` argument in `models.py`,
    `sweep.py` (resumable sweeps, learned-map Lyapunov exponent, long-run z statistics).
  - I expected ReLU to break down early for its one-step error. It didn't: 448 steps against 543 for GELU, but its
    ε is 3.3× larger, and it lasts 39 steps longer than ε alone predicts.
  - The ReLU map's exponent is 0.888 against 0.904 (exact map, same estimator), consistent in all 3 seeds. Smooth
    activations are within 0.003. The Jacobian argument is right about the mechanism and wrong about the harm.
  - Sine is the worst (398 steps) with the heaviest one-step error tail. tanh beats its ε prediction by 47 steps.
  - Nothing loses the attractor: 750 of 750 rollouts bounded at 20,000 steps, z-distributions at the noise floor.
  - Dead end avoided: the VM kills anything running between commands and caps each at 3 minutes, so training is
    checkpointed in slices. The macOS `.venv` is not usable from the VM.
- **Belief now.** On Lorenz at ε around 1e-5 to 1e-4, breakdown is set by ε to within about 0.4 Lyapunov times,
  whatever the activation. Activations matter through the ε they reach. The leftover residuals point at the tail of
  the ε distribution, not at smoothness.
- **H1 + H2 + H7** (`experiments/2026-11-24-h1-h2-width`), same day, same VM, about 4 hours of wall time.
  - Widths 32 to 1024, 3 seeds, plus 512 and 1024 in float64. `sweep.py` now pauses and resumes evaluation too.
  - The horizon formula's slope is right: 113 steps per e-fold of ε against 110. Breakdown is linear in ln N at
    α/λ. But α is 0.21 and falls to 0.12 past width 128. I had guessed 0.5 to 1.
  - Width 256 → 1024 is 12× the compute for 18 steps. Width 512 looked like a jump (607) until 1024 came in at 561.
    An interim "breakdown keeps climbing while ε is flat" reading was wrong; it was seed noise at 512.
  - Float64 changes nothing. ε floors at 1.3e-5, 100× above float32 resolution. H2's ceiling is not the one we hit.
  - The learned exponent does not help predict breakdown (H7). The p99 of ε does a bit better than the median.
  - Dead end: projecting one-step error onto the unstable direction. On Lorenz that direction is nearly parallel
    to the flow (|cos| about 0.9), so the split says nothing. Needs the adjoint vector.
  - Kept: ReLU's one-step error decorrelates in one step (lag-1 autocorrelation 0.32 against 0.95), which is a
    mechanism for it outlasting its ε.
- **Belief now, revised.** Breakdown on Lorenz is ε through a logarithm, as the formula says, and neither width,
  activation, precision nor the learned exponent moves it off that line by more than seed noise. The lever is the
  ε floor, and that is set by training, not capacity.
- **H9 (Lorenz)** (`experiments/2026-11-24-h9-quadratic`), same evening. New: `structured.py`.
  - I expected a small quadratic network to match the MLP. A quadratic one-step map is 60× worse (109 steps). The
    vector field is quadratic; the flow map over dt is not, and dt·|J| is 0.25, so the cubic terms matter.
  - Two quad layers tie the MLP (514 steps, 3651 params). Adam stalls them at 1.4e-5 although degree 4 is worth 1e-7.
  - Least squares on monomials: degree 3 ties the MLP in 2 seconds, degree 6 reaches 2026 steps.
  - A quadratic vector field inside RK4 × 10, 30 parameters, Gauss-Newton: exact to rounding, 3441 steps = 31
    Lyapunov times. In float32 the same weights last 1148. That is H2's ceiling, finally visible.
  - L-BFGS on the vector-field fit stalled at loss 6e-11; Gauss-Newton took it to 4e-31 in a few steps.
  - 46 models, eight decades of ε, one line: R² 0.99, slope −101 steps per e-fold against −110.
- **Belief now.** On Lorenz the question "how far can a learned predictor see" reduces to "how small is ε", and ε is
  set by how the model is fitted and what it is allowed to represent, not by how many neurons it has. The honest
  caveat is that Lorenz is the easy case: polynomial, known, smooth. Three-body is where this should break.
- **Next.** The three-body set (H3, H8, H4, H5, H9's second half). It needs `sweep.py` extended to three-body data,
  conditioning on masses and the energy-drift metric. Also still open: an ε-matched ReLU against GELU pair, and a
  better-optimised MLP (Gauss-Newton or L-BFGS polish) to see how much of the 1e-5 floor is Adam.

## 2026-11-20 (evening)

- **Environment.** This machine is an Intel Mac. Homebrew no longer supports Intel, so uv came from the official installer.
  PyTorch is pinned to 2.2.2, the last x86_64 macOS wheel, with `numpy<2`. Decided: **PyTorch**, not JAX.
- **neural-pde-dynamics: PINNs** (`experiments/2026-11-20-pinn-laplace-burgers`).
  - Ported the original AMATH 581 Lagaris trial-function Laplace solver. The original code applied the learning rate *twice*
    (`W -= lr * (lr * grad)`) for 10 iterations, so it never trained. Fixed, it reaches a max error of 1.2e-5,
    40× better than finite differences on the same grid.
  - Burgers PINN with periodicity and the IC built into the trial function: 2e-4 vs Cole-Hopf at ν = 0.1.
  - I expected ν = 0.02 to break it. It didn't: max error is 0.16% of the amplitude.
  - It is about 1000× slower than the spectral solver and conserves mass only to 1e-4.
- **chaos-predictability (new project).** The question: chaos is deterministic, so how far can a learned predictor see?
  - Wrote `notes/nonlinear-dynamics.typ`, which builds from AMATH 502 examples up to Lorenz, the double pendulum, three bodies and the
    predictability horizon. Also an interactive twin-trajectory explainer page.
  - Mistake: I first evaluated on the 500-step training window, which is too short. The step model never reached
    10% error inside it. Evaluation now uses rollouts 3–4× longer than training.
  - Added the "error injected every step" form of the horizon formula. It is more realistic than treating
    the one-step error as a one-off initial error.
  - **Lorenz:** the step model breaks down at 10% error after about 4.9 Lyapunov times. Its error grows at exactly λ, about 15× above an
    exact twin with the same initial error. The direct IC → trajectory model is far worse (74 against 536 steps).
    Starting off the attractor, every model fails within 3 steps.
  - **Three-body (Lagrange triangle + noise):** the learned models fail after about 1.1 Lyapunov times, while an exact twin
    lasts about 11. Model error dominates, not chaos. Measured cause: one-step error is 30× worse during close
    approaches (r < 0.1). Energy drifts 3% within 100 steps. The direct model beats the step model here (255 against 105 steps).
    Out of distribution, a different family or different masses breaks both models within 10–30 steps.
- **Belief now.** For smooth chaotic flows a learned step model can get close to the chaos limit. For gravity,
  close encounters plus unconstrained networks waste about 90% of the achievable horizon.
- **Next.** A Hamiltonian/symplectic network or an acceleration predictor with a symplectic integrator; regularised coordinates
  near encounters; push the PINN to lower ν.

## 2026-11-20

- Set up math-lab. Started `neural-pde-dynamics` with a pseudospectral reference
  solver for viscous Burgers, validated against the Cole-Hopf analytic solution.
- Next: pick the surrogate architecture (FNO vs. neural ODE on spectral
  coefficients) and decide JAX or PyTorch.
