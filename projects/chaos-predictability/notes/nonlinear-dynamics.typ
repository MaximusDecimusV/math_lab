#set page(margin: 1in)
#set math.equation(numbering: "(1)")
#set heading(numbering: "1.")
#set par(justify: true)

#align(center, text(17pt, weight: "bold")[Nonlinear dynamics, and how a deterministic system becomes unpredictable])

These notes build from fixed points up to the three-body problem, reusing the worked
examples from AMATH 502. The last section states the
hypothesis tested in `notebooks/02_lorenz_horizon.py` and `notebooks/03_threebody_horizon.py`.

A *dynamical system* here is an autonomous ODE $dot(bold(x)) = bold(f)(bold(x))$ with
$bold(x) in RR^n$. If $bold(f)$ is smooth, solutions exist and are unique: the
present state fixes the entire future. Chaos is *not* randomness. It is
determinism combined with exponential sensitivity to the initial state.

= One-dimensional flows: fixed points and stability

In 1D the picture is a *phase line*. Fixed points $x^*$ solve $f(x^*) = 0$.
Linearise with $eta = x - x^*$: $dot(eta) approx f'(x^*) eta$, so
$ f'(x^*) < 0 => "stable", quad f'(x^*) > 0 => "unstable", quad f'(x^*) = 0 => "look at higher order". $ <stab1d>

*Example*: $dot(x) = x - x^3$ has $x^* = 0$ (unstable, $f' = 1$) and $x^* = plus.minus 1$
(stable, $f' = -2$). For $dot(x) = 1 - e^(-x^2)$, $x^* = 0$ is *semi-stable*: $f'(0) = 0$ and
$f >= 0$ on both sides.

A trajectory in 1D moves monotonically toward a fixed point or off to infinity.
It can never turn around, so *1D flows cannot oscillate*, let alone be chaotic.
Numerically, RK4's global error is $O(h^4)$ against Euler's $O(h)$. At
$h = 0.05$ on the logistic flow, RK4 was accurate to about $5 times 10^(-10)$.

= Bifurcations: qualitative change as a parameter moves

When a parameter $r$ passes through a value where $f'(x^*) = 0$, fixed points can be
created, destroyed or exchange stability. There are three normal forms:

#table(
  columns: 3,
  [*type*], [*normal form*], [*what happens*],
  [saddle-node], [$dot(x) = r + x^2$], [two fixed points collide and vanish],
  [transcritical], [$dot(x) = r x - x^2$], [two fixed points exchange stability],
  [pitchfork (super)], [$dot(x) = r x - x^3$], [one stable point splits into two (symmetry)],
)

*Example*: harvested fish $dot(x) = x(1 - x) - h$. At $h_c = 1/4$ the stable and
unstable populations collide in a saddle-node. Past it the population collapses for *any*
initial size. A small, smooth change in the parameter causes a catastrophic change
in the behaviour.

= Two dimensions: Jacobians and the trace-determinant plane

For $dot(bold(x)) = bold(f)(bold(x))$ in $RR^2$, linearise at $bold(x)^*$ with the Jacobian
$J = partial bold(f) \/ partial bold(x)$. With $tau = op("tr") J$, $Delta = det J$:
$ lambda_(1,2) = (tau plus.minus sqrt(tau^2 - 4 Delta)) / 2 $ <eig2d>

- $Delta < 0$: saddle.
- $Delta > 0$, $tau^2 > 4 Delta$: node.
- $Delta > 0$, $tau^2 < 4 Delta$: spiral.
- $tau = 0$, $Delta > 0$: center.

Stability is set by the sign of $tau$.

*Examples*:
- $dot(x) = y, dot(y) = -4x$ is a center.
- $dot(x) = 2y, dot(y) = x$ is a saddle.
- $dot(x) = -x - 2y, dot(y) = 2x - y$ is a stable spiral ($tau = -2$, $Delta = 5$).

The RLC circuit $L I'' + R I' + I\/C = 0$ is a node or a spiral depending on the
sign of $R^2 C - 4L$.

The Hartman-Grobman theorem says the linearisation gives the correct picture near
*hyperbolic* points ($op("Re") lambda != 0$). Centers are not hyperbolic, and the
nonlinear terms decide what happens there.

= Conservative vs dissipative systems

If some $E(bold(x))$ satisfies $dot(E) = 0$ along trajectories, the system is
*conservative* and trajectories lie on level sets of $E$.

*Example*: Lotka-Volterra rabbits and foxes conserve $E = ln y - y + mu ln x - mu x$,
so the interior fixed point is a true nonlinear center, surrounded by closed orbits.

The divergence $nabla dot bold(f)$ is the rate at which phase-space volume grows.
Hamiltonian systems (pendulums, gravity) have $nabla dot bold(f) = 0$, so they
preserve volume (Liouville). They therefore have no attractors. *Dissipative*
systems contract volume. Lorenz has $nabla dot bold(f) = -(sigma + 1 + b) < 0$
everywhere, so its trajectories collapse onto an attracting set of zero volume.

= Limit cycles, Hopf, and why chaos needs three dimensions

A *limit cycle* is an isolated closed orbit.

*Example*: van der Pol, $dot(x) = y$, $dot(y) = -x + mu y (1 - x^2)$. The origin is an
unstable spiral, and every orbit is drawn onto one stable cycle.

A *Hopf bifurcation* happens when a complex pair $lambda = alpha plus.minus i omega$
crosses the imaginary axis.
- In the *supercritical* case, a small stable cycle of radius $prop sqrt(mu)$ is born.
- In the *subcritical* case, an unstable cycle shrinks onto the point and the state
  jumps to some distant attractor.

*Example*: $x'' + (x^2 - mu) x' + x = 0$ has a supercritical Hopf at $mu = 0$.
Averaging $x'' + x + epsilon h = 0$ predicts the radius of the cycle
(for $h = (x^4 - 1) x'$ it is $r = 8^(1/4)$).

*Poincaré-Bendixson.* A bounded trajectory of a planar autonomous flow that
avoids fixed points approaches a closed orbit. In 2D the only long-term
behaviours are fixed points, cycles, and connections between them. A trajectory
cannot cross itself, and in the plane that rules out the stretch-and-fold motion
chaos needs. *Chaos in an autonomous ODE requires at least three dimensions.*
Maps are different: even 1D maps can be chaotic, because they jump instead of
flowing continuously.

= Maps and Lyapunov exponents

An iterated map $x_(n+1) = g(x_n)$ has a fixed point $x^*$ that is stable when
$|g'(x^*)| < 1$, and *superstable* when $g'(x^*) = 0$.

*Example*: the Newton map for $g(x) = x^2 - 4$ has superstable fixed points $plus.minus 2$,
which is why Newton's method converges quadratically. The cubic map $x |-> 3x - x^3$ stays
bounded from $x_0 = 1.9$ but escapes from $x_0 = 2.1$.

The *Lyapunov exponent* measures how fast nearby orbits separate:
$ lambda = lim_(n -> oo) 1/n sum_(k=0)^(n-1) ln |g'(x_k)|, quad
  lambda_"flow" = lim_(t -> oo) lim_(delta_0 -> 0) 1/t ln (|delta(t)|) / (|delta_0|) $ <lyap-def>

*Examples*:
- $x |-> r x$ has $lambda = ln r$.
- The decimal shift $x |-> 10 x mod 1$ has $lambda = ln 10$: it reveals one new decimal
  digit of $x_0$ per step, so an initial uncertainty of $10^(-16)$ swamps the state
  after 16 steps.
- The tent map with $r > 2$ has $lambda = ln r$.

*Logistic map* $x_(n+1) = r x_n (1 - x_n)$ (the canonical route to
chaos):
- The fixed point $1 - 1\/r$ loses stability at $r = 3$ and gives way to a
  period-2 orbit.
- Period 4 appears at $r approx 3.449$, then 8, 16, and so on.
- The gaps between these period doublings shrink by Feigenbaum's universal ratio
  $delta = 4.6692...$.
- They accumulate at $r_oo approx 3.5699$. Beyond it, $lambda > 0$ for most $r$,
  with periodic windows (period 3 at $r = 1 + sqrt(8)$).

= Strange attractors: Lorenz

A *strange attractor* has a fractal structure. Recall the Cantor set: remove
middle thirds forever, and the result has zero length but uncountably many points, with
dimension $ln 2 \/ ln 3 approx 0.63$.

The Lorenz system, a three-mode truncation of convection, is
$ dot(x) = sigma (y - x), quad dot(y) = r x - y - x z, quad dot(z) = x y - b z $ <lorenz>
with $sigma = 10$, $b = 8\/3$. Its fixed points are the origin and, for $r > 1$,
$ C^plus.minus = (plus.minus sqrt(b(r - 1)), plus.minus sqrt(b(r - 1)), r - 1) $ <lorenz-fp>

As $r$ increases:

+ For $r < 1$ the origin is globally stable.
+ At $r = 1$ there is a supercritical pitchfork: the origin becomes a saddle and $C^plus.minus$
  are born stable (steady convection rolls turning one way or the other).
+ $C^plus.minus$ lose stability in a *subcritical* Hopf at
  $ r_H = sigma (sigma + b + 3) / (sigma - b - 1) approx 24.74 $ <lorenz-hopf>
+ For $r > r_H$ every fixed point is unstable and there is no stable cycle. Trajectories
  are bounded, because a large ellipsoid traps them. Volume contracts at rate $sigma + 1 + b$.

At $r = 28$ the result is the butterfly. The orbit loops around $C^+$, then $C^-$, in an
order that never repeats. The attractor has dimension about $2.06$ and largest Lyapunov
exponent $lambda_max approx 0.906$. The tests reproduce this ($0.90$, `tests/test_lyapunov.py`).
The mechanism is *stretching* along the unstable direction combined with *folding*,
which happens because trajectories stay bounded. A sheet stretched and folded forever
gives the Cantor-like cross section.

= The double pendulum

Two point masses $m_1, m_2$ on rigid rods $l_1, l_2$, with angles $theta_1, theta_2$ from the
downward vertical. Write $Delta = theta_1 - theta_2$ and $D = 2 m_1 + m_2 - m_2 cos 2 Delta$.
The Euler-Lagrange equations are
$
  dot(theta)_i = omega_i, quad
  dot(omega)_1 = (-g(2 m_1 + m_2) sin theta_1 - m_2 g sin(theta_1 - 2 theta_2)
    - 2 sin Delta thin m_2 (omega_2^2 l_2 + omega_1^2 l_1 cos Delta)) / (l_1 D), \
  dot(omega)_2 = (2 sin Delta (omega_1^2 l_1 (m_1 + m_2) + g (m_1 + m_2) cos theta_1
    + omega_2^2 l_2 m_2 cos Delta)) / (l_2 D)
$ <dp-eom>
and the energy, which is conserved,
$ E = 1/2 (m_1 + m_2) l_1^2 omega_1^2 + 1/2 m_2 l_2^2 omega_2^2
  + m_2 l_1 l_2 omega_1 omega_2 cos Delta - (m_1 + m_2) g l_1 cos theta_1 - m_2 g l_2 cos theta_2 $ <dp-energy>

The phase space is 4D, and conservation of energy confines motion to a 3D shell.
That is enough room for chaos. At *low energy* the motion is close to two coupled
linear oscillators, and the KAM theorem says most such quasi-periodic motion
survives small nonlinearity. At *high energy*, for example both arms released
near horizontal, the shell is mostly chaotic, and two pendulums started
$10^(-8)$ rad apart disagree completely within seconds. Since it is
Hamiltonian, there is no attractor, only a *chaotic sea* with regular islands.

= The three-body problem

Point masses $m_i$ at $bold(q)_i in RR^2$, with $G = 1$:
$ dot.double(bold(q))_i = sum_(j != i) m_j (bold(q)_j - bold(q)_i) / (|bold(q)_j - bold(q)_i|^3) $ <nbody>
Conserved quantities:
$ E = sum_i 1/2 m_i |dot(bold(q))_i|^2 - sum_(i<j) (m_i m_j) / r_(i j), quad
  bold(P) = sum_i m_i dot(bold(q))_i, quad L = sum_i m_i bold(q)_i times dot(bold(q))_i $ <nbody-invariants>

For two bodies these invariants are enough to reduce the problem to quadrature,
which gives Kepler's ellipses. For three bodies they are not. Bruns (1887) and
Poincaré (1890) showed there are no further algebraic or analytic integrals, so
no general closed-form solution exists. Sundman's convergent series (1912)
technically solves the problem, but it converges far too slowly to be useful.
Poincaré, studying exactly this problem, found the tangled homoclinic orbits that
we now call chaos.

There are special exact solutions:
- *Euler collinear* and *Lagrange equilateral* configurations rotating rigidly.
  The equal-mass Lagrange triangle is linearly *unstable* (Routh's criterion needs
  one dominant mass). Perturb it slightly and it breaks up into chaotic motion.
  This is the training family in `03_threebody_horizon.py`.
- The *figure-eight* choreography (Moore 1993; Chenciner-Montgomery 2000), in which
  three equal masses chase each other around one curve:
  $ bold(q)_1 = -bold(q)_2 = (0.97000436, -0.24308753), quad bold(q)_3 = 0, \
    dot(bold(q))_3 = -2 dot(bold(q))_1 = -2 dot(bold(q))_2 = (-0.93240737, -0.86473146) $ <figure-eight>
  with period $T approx 6.3259$. It is *linearly stable*, so small perturbations stay
  nearly periodic. It serves as the "regular" contrast family.
- *Pythagorean problem* (Burrau 1913): masses 3, 4, 5 released from rest at the
  vertices of a 3-4-5 triangle. A sequence of close encounters follows, and
  eventually a binary ejects the third body.

Close encounters ($r_(i j) -> 0$) make the force unbounded and force tiny time
steps. Our reference solver is adaptive 8th-order DOP853 at $10^(-12)$ tolerance.
It is validated by energy drift below $10^(-9)$, by the figure-eight returning
to its start after one period, and by agreement between tolerances $10^(-12)$ and
$10^(-13)$. Trajectories with encounters closer than $r_"min" = 0.05$ are left out
of training.

= Deterministic but unpredictable: the predictability horizon

Suppose a predictor starts with error $delta_0$, which might be measurement
error, round-off, or a neural network's one-step error. In a chaotic region the
error grows like $delta(t) approx delta_0 e^(lambda t)$, so it reaches a
tolerance $Delta$ at
$ t_p approx 1/lambda ln Delta / delta_0 $ <horizon>

This is the central tradeoff. Making the model 10 times more accurate buys only
$ln 10 \/ lambda$ more time, about $2.5$ time units for Lorenz. The quantity
$1\/lambda$ is the *Lyapunov time*.

For a learned surrogate rolled out step by step, there are two error sources.
- The *initial-error* term is amplified by chaos, as in @horizon.
- A *model-error* term is injected at every step and then amplified the same way.
  With per-step error $epsilon$, the total is roughly
  $ delta(t) approx epsilon (e^(lambda t) - 1) / (e^(lambda Delta t) - 1) approx epsilon / (lambda Delta t) e^(lambda t) $ <horizon-acc>
  which grows at the same rate $lambda$ but starts from an effective error $epsilon \/ (lambda Delta t)$.
  With $Delta t = 0.01$ and $lambda approx 1$, that is 100 times larger than $epsilon$. Small time
  steps therefore cost horizon unless the per-step error shrinks with them.

*Hypotheses tested in the notebooks:*
+ The rollout error of the step model grows at the rate set by $lambda$, and it breaks down within a
  few Lyapunov times of where @horizon predicts.
+ A "twin" exact trajectory started $delta_0 = epsilon$ away gives a lower bound on achievable error.
  The gap between the model and its twin is the part of the error that is the model's fault.
+ Breakdown comes later for the regular figure-eight family than for the chaotic Lagrange family,
  and much earlier out of distribution (larger perturbations, different masses).
