# h6-activations

**Hypothesis (H6):** Smooth activations beat ReLU at the same one-step error. A ReLU network's Jacobian is
piecewise constant, so its rollouts should break down earlier than a GELU, SiLU or tanh model with the same ε,
and the Lyapunov exponent of the learned map should sit further from the true one. H10 (long-run statistics
survive breakdown only for smooth activations) is measured from the same models.

**Setup:** see config.yaml, `src/chaos_dyn/sweep.py` and `summarise.py`. Commit: `493c63b` (plus uncommitted work).
- Lorenz data, split and evaluation exactly as in `2026-11-20-lorenz-horizon`: 400 test ICs, 2000-step rollouts.
- Step MLP 256×3 (133k params), 15,000 iterations, batch 2048, lr 2e-3. Activations: ReLU, GELU, SiLU, tanh,
  sine (plain sin with default init, not SIREN). Seeds 0, 1, 2. The seed now also sets the initial weights.
- Learned-map Lyapunov exponent: twin method in float64 on the network, 200 test ICs × 4000 steps. The same
  estimator on the exact map gives 0.9036 ± 0.0034, which is the reference below (not 0.906).
- Long run: 50 test ICs × 20,000 steps; Wasserstein distance between the rollout's z-distribution and the
  training data's. Exact rollouts score 0.067 on the same measure, so that is the noise floor.
- Run in a Linux VM on the Mac, one thread per run. About 550 s of training per run for every activation.

**Result:** mean over 3 seeds (± sd across seeds). Breakdown is the median step at which error first exceeds 10%.

| | one-step ε | ε p99 | breakdown 10% | exact twin | gap to twin | predicted from ε alone | residual | learned λ | z-Wasserstein | bounded |
|---|---|---|---|---|---|---|---|---|---|---|
| GELU | 1.78e-5 | 8.9e-5 | 543 ± 5 | 833 | 291 | 543 (anchor) | 0 | 0.901 ± 0.003 | 0.027 | 100% |
| SiLU | 2.05e-5 | 8.7e-5 | 513 ± 4 | 813 | 300 | 527 | −14 | 0.901 ± 0.003 | 0.025 | 100% |
| tanh | 3.99e-5 | 1.7e-4 | 501 ± 20 | 741 | 241 | 453 | +47 | 0.907 ± 0.001 | 0.031 | 100% |
| ReLU | 5.94e-5 | 2.0e-4 | 448 ± 11 | 700 | 252 | 409 | +39 | 0.888 ± 0.003 | 0.026 | 100% |
| sine | 5.92e-5 | 2.5e-4 | 398 ± 29 | 699 | 301 | 410 | −12 | 0.901 ± 0.003 | 0.067 | 100% |

"Predicted from ε alone" moves the GELU breakdown by ln(ε/ε_GELU)/(λΔt), the horizon formula's slope.
Figure: `figures/h6-activations.png`. Per-run numbers: `outputs/h6-activations/*.json` and `summary.json`.

- **ReLU does not break down early for its ε.** It reaches a 3.3× larger one-step error than GELU on the same
  budget, and that costs it 95 steps (0.9 Lyapunov times). The formula says a 3.3× larger ε should cost 133. ReLU
  lasts 39 steps *longer* than ε alone predicts, and its gap to the exact twin (252) is smaller than GELU's (291).
- **The ReLU map is measurably less chaotic.** Its exponent is 0.888 in all three seeds (0.890, 0.885, 0.891)
  against 0.904 for the exact map with the same estimator. Every smooth activation is within 0.003 of the
  reference. So the mechanism in H6 is real (the piecewise-linear Jacobian distorts the stretching rate by
  1.7%), but its sign is the opposite of the harm predicted: slower error growth.
- **The learned exponent does not explain the residuals.** tanh has the largest positive residual (+47) with an
  exponent on target. Adding |λ_learned − λ_ref| to ln ε moves the pooled R² from 0.76 to 0.79 (15 runs in 5
  clusters, descriptive only).
- **Sine is the worst, and the tail is the likely reason.** Same median ε as ReLU, breakdown 50 steps earlier,
  and the heaviest one-step error tail (p99 2.5e-4 against 2.0e-4, max 5e-3 against 2e-3). Seed 1 is much worse
  than seeds 0 and 2 (365 against 412 and 417).
- **What separates activations is mostly the ε they reach, not what they do with it.** GELU and SiLU reach
  about 2e-5, tanh 4e-5, ReLU and sine 6e-5 on the same budget. The fitted slope of breakdown against ln ε
  across all 15 runs is −90 steps per e-fold against −110 from the formula.
- **H10: no activation loses the climate.** All 750 long rollouts stay bounded for 20,000 steps, none collapses
  to a fixed point, and every z-distribution distance is at or below the exact-rollout noise floor (0.067).
  ReLU is as good as GELU here.
- GELU seed 0 retrained from a seeded init gives 538 steps and ε 1.69e-5, against 536 and 1.54e-5 on 11-20.

**Verdict:** H6 falsified as stated on Lorenz: at this accuracy breakdown is set by ε to within about ±45 steps
(0.4 Lyapunov times) whatever the activation, and ReLU sits on the favourable side. Its secondary prediction
holds: the ReLU map's exponent is off by 1.7%. H10's prediction of ReLU failures is falsified on Lorenz. The
residuals that remain (tanh +47, ReLU +39, sine −12 with a heavy tail) are 2 to 3 seed standard deviations and
are better candidates for the tail-of-ε question than for an activation story.

**Caveats:** ε is not matched by construction, only corrected for with the formula's slope. A cleaner test
trains each activation to a common ε (early stopping or a longer ReLU budget). Three seeds. One width.
