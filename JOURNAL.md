# Journal

Newest first. One entry per session. Record what you tried, what happened, and
what you now believe. Dead ends are the most valuable entries.

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
