# pinn-laplace-burgers

**Hypothesis:**
1. The grad-school version of the Lagaris trial-function network for Laplace failed because of the optimizer, not
   the method. Fixed, it should beat a same-grid finite-difference solve.
2. A hard-constrained PINN for Burgers matches Cole-Hopf at ν = 0.1 to better than 1e-3.
3. At ν = 0.02, with a steep front, it degrades noticeably.

**Setup:** see config.yaml and `notebooks/02_pinn_laplace_burgers.py`. Commit: `d523dd6` (plus uncommitted work).
Everything ran in float64 on CPU (Intel i5, 4 cores), taking about 22 min in total.

**Result:**

| case | metric | value |
|---|---|---|
| Laplace, trained (Adam 2000 + L-BFGS 300, 24² collocation) | max error vs exact | **1.2e-5** |
| Laplace, untrained (what the original update rule amounted to) | max error vs exact | 0.32 |
| Laplace, 5-point FD on the same 24² grid | max error vs exact | 4.5e-4 |
| Burgers ν = 0.1, t ∈ [0, 1] | max abs error vs Cole-Hopf (max \|u\| = 0.115) | **2.1e-4** |
|  | mass drift / energy-increase steps | 1.0e-5 / 0 |
| Burgers ν = 0.02, t ∈ [0, 1.5] | max abs error vs spectral (max \|u\| ≈ 1.3) | **2.1e-3** |
|  | relative L2 error at t = 1.5 | 5.6e-4 |
|  | mass drift | 1.6e-4 |

Figures: `figures/pinn-laplace.png`, `figures/pinn-burgers.png`.

- (1) holds. The original code applied the learning rate twice (`W -= lr * (lr * grad)`) for 10
  iterations, which is effectively no training. With a proper optimizer, the same trial function
  is 40× more accurate than finite differences on the same grid.
- (2) holds.
- (3) is wrong at this viscosity. The error is about 10× larger in absolute terms but still only about
  0.16% of the solution's amplitude. It concentrates along the two steepening fronts (x ≈ 3 and 3.6), as
  expected. ν = 0.02 still resolves the front width (about ν/|Δu| ≈ 0.01) with 4096 random collocation points.
  The PINN conserves mass only approximately (1e-4), whereas the spectral solver conserves it to 1e-12.
  Nothing in the trial function enforces it.

Cost: each Burgers PINN takes about 10 min on CPU, against well under a second for the spectral solver. As a *solver*,
the PINN is not competitive here. Its value is as a differentiable, mesh-free surrogate.

**Verdict:** worked. Next, push ν down (0.005, 0.002) to find where the PINN actually
breaks, and add a mass-conservation constraint to the trial function.
