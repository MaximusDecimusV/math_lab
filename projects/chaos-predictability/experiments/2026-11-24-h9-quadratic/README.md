# h9-quadratic

**Hypothesis (H9):** Matching the structure of the equations beats adding neurons. The Lorenz right-hand side is
quadratic, so a small network of multiplicative units (under 5k parameters) should match or beat the
133k-parameter GELU MLP on ε and breakdown. This experiment also gives H2 (float32 ceiling) the test it could
not get from the width sweep. Lorenz only: the three-body half of H9 is not run yet.

**Setup:** see config.yaml, `src/chaos_dyn/structured.py`, `run.py` and `summarise.py`. Commit: `493c63b` (plus
uncommitted work). Same data, split and evaluation as H6 and H1, with rollouts extended to 4000 steps.
- **poly2 to poly6:** all monomials of the normalised state up to that degree, then a linear map to the one-step
  delta. Least squares on 400k training pairs in float64. No iterative training.
- **quad:** layers h = (A x)·(B x) + C x, width 32, one layer (483 params, a general quadratic map) or two
  (3651 params, up to degree 4). Adam, 15,000 iterations, float32, seeds 0 to 2: the MLP recipe.
- **vf:** a quadratic *vector field* (30 coefficients) advanced one dt by RK4, with 1 substep or 10 (the data
  generator uses 10). Gauss-Newton fit in float64 on 4000 pairs; the same weights are then rolled out in float64
  and in float32.
- Reference from H1: GELU MLP 256×3, 133k params, ε 1.78e-5, breakdown 543 ± 5 steps (4.9 Lyapunov times).

**Result:** breakdown is the median step at which error first exceeds 10%.

| model | params | one-step ε | breakdown 10% | Lyapunov times | learned λ | fit time |
|---|---|---|---|---|---|---|
| poly2 | 30 | 1.1e-3 | 109 | 1.0 | 0.847 | 1 s |
| quad, 1 layer (3 seeds) | 483 | 1.1e-3 | 109 | 1.0 | 0.852 | 59 s |
| poly3 | 60 | 1.1e-5 | 512 | 4.6 | 0.902 | 2 s |
| quad, 2 layers (3 seeds) | 3651 | 1.4e-5 | 514 ± 13 | 4.7 | 0.903 | 94 s |
| *GELU MLP 256×3 (H1)* | *133k* | *1.8e-5* | *543 ± 5* | *4.9* | *0.901* | *9 CPU min* |
| poly4 | 105 | 1.1e-7 | 1028 | 9.3 | 0.901 | 4 s |
| vf, RK4 × 1, float64 | 30 | 1.3e-7 | 1025 | 9.3 | 0.906 | 3 s |
| vf, RK4 × 1, float32 | 30 | 1.4e-7 | 1026 | 9.3 | 0.902 | (same fit) |
| vf, RK4 × 10, float32 | 30 | 9.8e-8 | 1148 | 10.4 | 0.900 | (same fit) |
| poly5 | 168 | 6.7e-10 | 1605 | 14.5 | 0.906 | 4 s |
| poly6 | 252 | 4.6e-12 | 2026 | 18.4 | 0.902 | 4 s |
| vf, RK4 × 10, float64 | 30 | 0 (max 1e-15) | 3441 | 31.2 | 0.907 | 7 s |

Figure: `figures/h9-quadratic.png`. Per-run numbers: `outputs/h9-quadratic/*.json` and `summary.json`.

- **The literal prediction fails, and the reason is worth keeping.** A purely quadratic one-step map is 60× worse
  than the MLP (poly2 and the one-layer quad network agree to three digits: ε 1.1e-3, 109 steps). The *vector
  field* is quadratic; the *flow over dt = 0.01* is not. Its degree-3 terms enter at order dt², and dt·|J| is about
  0.25 here, so they are not small.
- **Two quad layers match the MLP with 36× fewer parameters** (514 against 543 steps), but do not beat it. They can
  represent degree 4, which least squares shows is worth ε 1e-7, yet Adam leaves them at 1.4e-5. That is the same
  optimisation floor every MLP hit in H1.
- **Put the structure in the right place and the floor goes away.**
  - Polynomial features with a closed-form fit: each degree buys about two decades of ε and 500 steps. Degree 3
    (60 params, 2 seconds) already matches the MLP. Degree 6 reaches 18 Lyapunov times.
  - The quadratic vector field inside the data's own integrator reproduces the reference to rounding error with
    30 parameters and lasts 3441 steps, 31 Lyapunov times. That is the float64 limit, not a model limit.
  - With one RK4 substep instead of ten, the integrator mismatch alone sets ε at 1.3e-7 and caps the horizon at
    1025 steps.
- **H2 gets its real test here.** The exact model in float32 breaks down at 1148 steps (10.4 Lyapunov times),
  against 3441 in float64. H2 predicted a float32 ceiling near 1000 steps and named "above about 1100" as a
  falsifier. 1148 is over that line by 4%. I read it as the ceiling existing where predicted, with the constant
  slightly conservative, because rounding errors do not accumulate coherently.
- **The horizon formula holds across eight decades.** Pooling these runs with H1 and H6 (46 models with ε from
  4.6e-12 to 1.1e-3): R² of breakdown on ln ε is 0.989, slope −101 steps per e-fold against −110 in theory. Measured
  breakdown sits 103 ± 47 steps above the "ε every step" formula, so the formula stays a conservative estimate.
- **Bad models do lose the climate (H10 footnote).** poly2 and the one-layer quad network stay bounded, but their
  exponent is 6% low (0.85) and their z-distribution distance is 0.6, ten times the noise floor.

**Verdict:**
- **H9: falsified as stated, supported in spirit.** A quadratic network on the one-step map does not beat the MLP
  (one layer is far worse, two layers tie). Structure that matches the equations where they are actually
  quadratic, the vector field, beats 133k neurons by 2900 steps with 30 parameters.
- **H2: supported, revising this morning's status.** Float32 caps the Lorenz horizon near 1150 steps. The width sweep
  could not see it because training floors ε at 1e-5.

**Caveats:**
- The vector-field fit is sparse-regression system identification in the SINDy family, with the right function
  library handed to it. It shows what the ceiling is. It is not a new method, and it says nothing about systems
  whose equations are unknown or not polynomial, which is the point of the three-body half of H9.
- "Exact" means bit-level agreement with this repo's RK4 reference, not with the true Lorenz flow.
- poly and vf fits are deterministic, so they have one run each.
