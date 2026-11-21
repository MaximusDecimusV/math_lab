# neural-pde-dynamics

**Question.** Can a learned surrogate for a nonlinear PDE reproduce long-horizon
dynamics while respecting the invariants the true system has (mass conservation,
energy dissipation, known coherent structures)?

**Current belief.** Hard-constrained PINNs (Lagaris trial functions) are accurate on
smooth problems. Burgers matches Cole-Hopf to 2e-4. At ν = 0.02, with a steep front, the
error is still only 0.16% of the amplitude. They do not conserve mass exactly, and they
are about 1000× slower than the spectral solver (see
`experiments/2026-11-20-pinn-laplace-burgers`). **Framework: PyTorch** (pinned to 2.2.2,
the last release with Intel-Mac wheels).

**Next step.** Push ν down until the PINN breaks, and build mass conservation into the
trial function. Then the learned *time-stepping* surrogate: FNO on the solution field vs. a
neural ODE on the leading spectral coefficients. The chaotic-ODE version of that question
lives in `projects/chaos-predictability`.

## What's here

- `notes/derivation.typ` covers viscous Burgers, the Cole-Hopf solution, and the integrating-factor RK4 scheme.
- `src/neural_pde/burgers.py` is a pseudospectral reference solver.
- `tests/test_burgers.py` checks mass conservation, energy decay, agreement with Cole-Hopf, and fourth-order time convergence.
- `notebooks/01_burgers_baseline.py` is a jupytext notebook that plots a reference solution.
- `src/neural_pde/pinn.py` contains the neural solvers:
  - A Laplace trial-function network, ported from the original AMATH 581 project with its optimizer bug fixed.
  - A periodic, IC-exact Burgers PINN.
  - A direct finite-difference Laplace solve for comparison.
- `tests/test_pinn.py` checks that the boundary and initial conditions hold exactly, the residual operator,
  and finite-difference convergence. The full-accuracy runs are marked `slow` (`uv run pytest -m slow`).
- `notebooks/02_pinn_laplace_burgers.py` trains both solvers and compares them with exact, FD and spectral solutions.

## Candidate systems after Burgers

- KdV (solitons: exact solutions for testing, plus integrable structure)
- Kuramoto-Sivashinsky (chaotic, a hard long-horizon test)
- Allen-Cahn (gradient flow with a Lyapunov functional to check)
