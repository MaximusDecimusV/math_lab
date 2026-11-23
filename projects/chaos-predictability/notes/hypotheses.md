# Hypotheses: network size, activation functions and the chaos horizon

A running list of testable claims about how well neural networks can predict chaotic systems.
Each claim has a concrete prediction, a test, and a result that would falsify it. Update the
status line as experiments land, and link the experiment folder.

## What we already know (2026-11-20 experiments)

| | Lorenz | three-body |
|---|---|---|
| step model | MLP 256×3, GELU, 133k params | same |
| one-step error ε | 1.5e-5 | 7.7e-5 |
| breakdown at 10% error | 536 steps = 4.9 Lyapunov times | 105 steps = 1.1 Lyapunov times |
| exact twin (same ε, no further error) | 849 steps = 7.7 Lyapunov times | 1112 steps = 11.7 Lyapunov times |
| λ, Lyapunov time | 0.906, 110 steps | 0.52, about 95 steps |

The horizon formula with error injected every step is

    t_p ≈ (1/λ) · ln( tol · (e^{λΔt} − 1) / ε )

So **the horizon depends on the one-step error only through a logarithm**. Anything that changes
ε by a factor k moves the horizon by ln(k)/λ: for Lorenz, a 10× reduction in ε is worth about 2.3 Lyapunov times (254 steps).
Most of the hypotheses below follow from asking how neurons and activations change ε, and
whether ε is even the right summary.

## H0: your theory

_To be written down. Once stated, give it a prediction, a test and a falsifier like the rest._

## Network size

**H1. The horizon grows with the logarithm of the parameter count.**
If the one-step error follows a power law, ε ∝ N^(−α), the formula gives a straight line:
t_p ≈ const + (α/λ) ln N.
- Prediction: breakdown step against log N is a straight line. Each doubling of the parameters adds the same
  number of steps, α·ln2/λ: about 38 steps for α = 0.5 and 77 for α = 1 on Lorenz. Going from 133k to 1M
  parameters should add at most about 1–2 Lyapunov times, not double the horizon.
- Test: Lorenz step model with widths 32, 64, 128, 256, 512 and 1024 at depth 3, fixed data, 3 seeds each. Fit α
  from ε against N, then check that the slope of t_p against ln N matches α/λ.
- Falsified if: the curve bends upward (capacity buys more than the log predicts), or it is flat while ε keeps falling
  (ε is then not what limits the horizon).
- Status: untested.

**H2. Precision sets a ceiling no network size can pass.**
Float32 limits ε to roughly 1e-7 relative error. Plugging that into the formula caps the Lorenz horizon at about 1000 steps
(about 9 Lyapunov times). Today's model is at 536 steps, so size and training can buy at most about 4 more Lyapunov times in float32.
- Prediction: in the H1 sweep, ε stops falling near 1e-7 and t_p plateaus at about 1000 steps. Training and
  rolling out in float64 moves the plateau out.
- Test: repeat the two largest H1 widths in float64.
- Falsified if: float32 models exceed about 1100 steps, or the float64 runs plateau at the same place.
- Status: untested.

**H3. For three-body, data near close encounters matters more than neurons.**
The one-step error is 30× larger when two bodies are within 0.1 of each other, and those states are under 1%
of the training data.
- Prediction: 4× the width changes three-body breakdown by less than 20%. Oversampling or loss-weighting close
  encounters with the same network helps more.
- Test: widths 256 and 1024, each trained on the current data and on close-encounter-weighted data.
- Falsified if: width alone gives a gain comparable to the weighting.
- Status: untested.

**H4. The direct model's capacity must grow exponentially with the horizon.**
The map from initial condition to the state at time t stretches by e^(λt) along a fractal attractor of dimension D ≈ 2.06.
Covering it to a fixed tolerance needs N ∝ e^(λDt) pieces, so t_p ≈ ln N / (λD).
- Prediction: the direct model's breakdown also grows with log N, by about 1.2 time units (120 steps) per 10× parameters on Lorenz.
  Comparing that slope with H1's tells us which design gains more from neurons.
- Test: direct model at hidden widths 128 to 2048 on the Lorenz data.
- Falsified if: the slope is far from 1/(λD), for example if it grows like a power of N.
- Status: untested.

**H5. At a fixed parameter count, width beats depth beyond about 3 layers.**
One step of Δt = 0.01 is close to the identity map plus a smooth correction. That needs accuracy more than
compositional depth.
- Prediction: at matched parameter counts, depths 2–3 reach a lower ε than depths 6–8.
- Test: about 130k parameters arranged as 2, 3, 4, 6 and 8 layers.
- Status: untested.

## Activation functions

**H6. Smooth activations beat ReLU at the same one-step error.**
A ReLU network is piecewise linear. Its Jacobian is piecewise constant and jumps across region
boundaries, so the learned map stretches and folds in the wrong places even when its one-step error is small. Rollout errors
grow according to the learned map's Jacobian, not the true one.
- Prediction: ReLU models matched to GELU, SiLU or tanh on one-step ε break down earlier. The Lyapunov exponent of
  the learned map itself (Benettin on the network) is further from 0.906 for ReLU.
- Test: ReLU, GELU, SiLU, tanh and sine at width 256, 3 seeds each. Report ε, breakdown, and the learned map's λ.
- Falsified if: breakdown is set by ε alone, whatever the activation.
- Status: untested.

**H7. The learned Lyapunov exponent predicts the horizon better than ε does.**
- Prediction: across every model in H1 and H6, breakdown correlates more strongly with the combination of
  ε and |λ_learned − λ_true| than with ε alone.
- Test: regression on the pooled H1 and H6 results. Needs no new training.
- Status: untested.

**H8. Sine activations help in close encounters and hurt off-distribution.**
Periodic activations reduce spectral bias, so they fit sharp features like the 1/r² spike better, but they
extrapolate in oscillatory ways.
- Prediction: sine networks lower the three-body close-encounter error and the in-distribution breakdown, but do
  worse than GELU on figure-eight and Pythagorean starting conditions.
- Test: H6's activation sweep on three-body, including the out-of-distribution sets.
- Status: untested.

**H9. Matching the structure of the equations beats adding neurons.**
The Lorenz right-hand side is a quadratic polynomial. Networks with multiplicative units, such as x² activations
or products of two linear layers, can represent it exactly.
- Prediction: a small quadratic network (under 5k parameters) matches or beats the 133k-parameter GELU MLP on ε and
  breakdown on Lorenz, with no comparable gain on three-body, whose force law is not polynomial.
- Test: a quadratic-unit MLP and a polynomial-feature linear model, on Lorenz and three-body.
- Status: untested.

**H10. Long-run statistics survive after the trajectory fails, but only for smooth activations.**
After breakdown a good model should still trace out the attractor, getting the "climate" right even though the "weather" is wrong.
- Prediction: after 20,000 steps, GELU, SiLU and tanh rollouts keep a z-histogram close to the true one
  (Wasserstein distance under 1) and stay bounded. Some ReLU rollouts collapse onto a spurious fixed point or
  cycle, or blow up.
- Test: long rollouts from 50 test starting conditions per model in H6. Measure the z-distribution distance, the
  fraction of rollouts that stay bounded, and the learned λ.
- Status: untested.

## Protocol shared by every test

- Start with Lorenz, where a model trains in about 4 minutes, and move a hypothesis to three-body only once it holds there.
- Use the existing cached datasets, the same training budget and seeds 0, 1 and 2. Record wall-clock time, so that "more
  neurons" is not confused with "more compute".
- Report, for every run:
  - one-step ε (median over test states);
  - breakdown at 1%, 10% and 50%, in steps and in Lyapunov times;
  - the gap to the exact twin;
  - the learned map's λ;
  - for three-body, energy drift at step 100.
- Code change needed first: add an `activation` argument to `mlp()` in `src/chaos_dyn/models.py`
  (GELU is hard-coded today), plus a quadratic-unit option for H9.
- One experiment folder per sweep (`scripts/new-experiment.sh chaos-predictability <name>`), with the hypothesis
  id in its README.

Suggested order, cheapest and most informative first: H6 (activations, Lorenz), then H1 + H2 (width
sweep), H7 (free from those runs), H10, H9, then the three-body set (H3, H8, H4, H5).
