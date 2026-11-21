# neural-pde-dynamics

**Question.** Can a learned surrogate for a nonlinear PDE reproduce long-horizon
dynamics while respecting the invariants the true system has (mass conservation,
energy dissipation, known coherent structures)?

**Current belief.** Unknown. Step one is a trustworthy reference solver, so any
error in the surrogate can be attributed to the surrogate.

**Next step.** Pick a surrogate: FNO on the solution field vs. a neural ODE on
the leading spectral coefficients. Decide JAX or PyTorch.

## What's here

- `notes/derivation.typ` covers viscous Burgers, the Cole-Hopf solution, and the integrating-factor RK4 scheme.
- `src/neural_pde/burgers.py` is a pseudospectral reference solver.
- `tests/test_burgers.py` checks mass conservation, energy decay, agreement with Cole-Hopf, and fourth-order time convergence.
- `notebooks/01_burgers_baseline.py` is a jupytext notebook that plots a reference solution.

## Candidate systems after Burgers

- KdV (solitons: exact solutions for testing, plus integrable structure)
- Kuramoto-Sivashinsky (chaotic, a hard long-horizon test)
- Allen-Cahn (gradient flow with a Lyapunov functional to check)
