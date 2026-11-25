# h1-h2-width

**Hypotheses:**
- **H1.** The horizon grows with the logarithm of the parameter count: if ε ∝ N^(−α), breakdown rises by α·ln2/λ
  steps per doubling of N.
- **H2.** Float32 limits ε to about 1e-7, which caps the Lorenz horizon near 1000 steps; float64 moves the cap.
- **H7** (no new training). The learned map's Lyapunov exponent predicts breakdown better than ε alone. Pooled
  over these runs and `2026-11-24-h6-activations`.

**Setup:** see config.yaml, `src/chaos_dyn/sweep.py`, `summarise.py` and `error_structure.py`. Commit: `493c63b`
(plus uncommitted work).
- Lorenz step MLP, GELU, depth 3, widths 32 to 1024 (2.3k to 2.1M parameters), seeds 0, 1, 2, float32. Same data,
  15,000 iterations, batch 2048 and lr 2e-3 at every width. The width-256 runs are the three GELU runs from H6.
- H2: widths 512 and 1024 again in float64 (training and rollout), seed 0 only.
- Evaluation as in H6: 400 test ICs, 2000-step rollouts, learned-map exponent, 20,000-step long runs.
- Run in a Linux VM on the Mac. CPU minutes are process time, so they are comparable across widths.

**Result:** float32, mean over 3 seeds (± sd across seeds).

| width | params | one-step ε | ε p99 | breakdown 1% | breakdown 10% | breakdown 50% | exact twin (10%) | gap | learned λ | CPU min |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 2.3k | 5.96e-5 | 2.8e-4 | 161 | 411 ± 45 | 580 | 699 | 288 | 0.900 | 2 |
| 64 | 8.8k | 3.01e-5 | 1.7e-4 | 220 | 479 ± 12 | 646 | 774 | 294 | 0.902 | 3 |
| 128 | 34k | 2.15e-5 | 1.0e-4 | 269 | 517 ± 8 | 680 | 808 | 292 | 0.900 | 5 |
| 256 | 133k | 1.78e-5 | 8.9e-5 | 305 | 543 ± 5 | 728 | 833 | 291 | 0.901 | 9 |
| 512 | 529k | 1.55e-5 | 6.7e-5 | 358 | 607 ± 25 | 784 | 850 | 243 | 0.900 | 34 |
| 1024 | 2.1M | 1.28e-5 | 6.3e-5 | 327 | 561 ± 28 | 743 | 866 | 305 | 0.903 | 109 |

| float64, seed 0 | one-step ε | breakdown 10% | float32 at the same width |
|---|---|---|---|
| 512 | 1.57e-5 | 606 | 1.55e-5, 607 ± 25 |
| 1024 | 1.55e-5 | 574 | 1.28e-5, 561 ± 28 |

Figure: `figures/h1-h2-width.png`. Per-run numbers: `outputs/h1-h2-width/*.json`, `summary.json`, `error_structure.json`.

- **H1's mechanism holds, with a much smaller exponent than assumed.**
  - Across the 18 runs breakdown falls by 113 steps per e-fold of ε. The horizon formula says 110.
  - ε ∝ N^(−α) with α = 0.21 overall. That predicts 15.7 steps per doubling of N; measured 16.8.
  - H1 guessed α between 0.5 and 1 (38 to 77 steps per doubling). The real figure is under half of the low end.
  - α is not constant: 0.38 from width 32 to 128, then 0.12 from 128 to 1024. The curve bends *down*, not up.
- **What 900× the parameters bought:** ε down 4.7×, breakdown up 150 steps (1.4 Lyapunov times), 54× the CPU time.
  Going from 256 to 1024 costs 12× the compute for 18 steps, which is inside the seed noise.
- **The gap to the exact twin does not close with size.** It is 288 to 305 steps at every width except 512. Bigger
  networks get a smaller ε and the twin moves out by the same amount.
- **Width 512 sits about 50 steps above the trend** (gap 243). That is 2 seed standard deviations with 3 seeds, and
  width 1024 does not continue it, so I read it as noise until more seeds say otherwise.
- **H2: precision is not what limits these models.** ε flattens near 1.3e-5, two decades above the float32
  resolution of about 1e-7. Float64 changes nothing at 512 (606 against 607) or at 1024 (574 against 561 ± 28).
  The predicted plateau near 1000 steps was never approached: the best run breaks down at 627. The ceiling at
  this budget is optimisation (15,000 Adam iterations), not arithmetic.
- **H7: the learned exponent adds nothing.** Over 32 models (these 20 plus the 12 non-GELU H6 runs), breakdown
  against ln ε has R² 0.79. Adding |λ_learned − λ_ref| gives 0.80. Every GELU width has an exponent within 0.004
  of the reference, so there is nothing for it to explain. The residual scatter after ln ε is 31 steps, about the
  seed noise.
- **The tail of the one-step error predicts slightly better than its median.** ln(p99 of ε) alone gives R² 0.84,
  ln(RMS) 0.83, against 0.79 for the median. Median plus p99 gives 0.84, so the median adds nothing once the
  tail is known. A modest effect on 32 models in 10 clusters.
- **Error structure (exploratory, `error_structure.py`).**
  - ReLU's one-step error decorrelates within one step along a trajectory (lag-1 autocorrelation 0.32, against
    0.91 to 0.95 for every smooth activation at width 256). Rough error accumulates less coherently, which fits
    ReLU outlasting its ε in H6.
  - Width 1024 is also less coherent (0.68) than 512 (0.89) or 256 (0.95), without lasting longer.
  - Projecting the error on the unstable direction did not help: on Lorenz the leading tangent vector is nearly
    parallel to the flow (median |cos| about 0.9), so "along the flow" and "along the unstable direction" are
    almost the same measurement. A proper version needs the adjoint (covariant) vector.
- **H10 again:** all 20 models stay bounded for 20,000 steps on all 50 ICs; z-Wasserstein at most 0.08.

**Verdict:**
- **H1: supported in form, wrong in size.** Breakdown is linear in ln N with the slope α/λ the formula predicts,
  but α is about 0.2 and falling, so size is a poor lever: past 34k parameters each doubling buys about 10 steps.
- **H2: not supported, and its premise was not reached.** Float32 and float64 plateau in the same place, which is
  H2's own falsifier, but the plateau is set by training at ε about 1e-5, not by precision at 1e-7. H2 can only be
  tested properly once ε is pushed down 100×.
- **H7: falsified on this pool.** ε (better, its tail) predicts breakdown; the learned exponent does not.

**Caveats:** one training budget, so "more neurons" is confounded with "under-trained for its size" at 512 and
1024. Float64 has one seed. Three seeds per width, and the seed sd at widths 32, 512 and 1024 is 25 to 45 steps.

**Next:** the binding constraint is the ε floor from optimisation. A longer or better-conditioned training run at
width 256 (more iterations, a lower final learning rate, or an L-BFGS polish in float64) is a more direct test of
both H1's ceiling and H2 than more width.
