/* Shared glossary for the chaos-predictability pages.
   Mark a term with <a class="term" data-term="key" href="...">; tapping it opens a short definition.
   A page that contains <dl id="gloss-list"> gets the full glossary rendered into it.
   Set <main data-course="101"> on the Dynamical Systems 101 page so chapter links stay on-page. */
(() => {
  "use strict";
  const COURSE = "https://claude.ai/artifact/Edxo4qZeDAwA4jtkeKAzGm";
  // [key, term, definition, chapter of Dynamical Systems 101 or null]
  const ENTRIES = [
    ["dynamical-system", "Dynamical system", "A state plus a rule for how it changes in time. Given the starting state, the rule fixes the whole future.", 0],
    ["deterministic", "Deterministic", "No randomness: the same starting state always produces exactly the same future.", 0],
    ["ode", "Differential equation", "An equation for a rate of change, such as ẋ = f(x). It says how fast each variable is changing, given the current state.", 0],
    ["linear", "Linear vs nonlinear", "Linear rules only add and scale the state variables, so solutions can be added together (superposition). Nonlinear rules contain terms like x², xy or sin x, and solutions no longer add.", 0],
    ["ic", "Initial condition", "The starting state of a simulation.", 0],
    ["phase-space", "Phase space", "The space of all possible states. One point is a complete snapshot of the system, and a trajectory is a curve through this space.", 0],
    ["fixed-point", "Fixed point (equilibrium, steady state)", "A state where the rate of change is zero, so the system stays there forever.", 1],
    ["stable", "Stable and unstable", "A fixed point is stable if states that start near it move back toward it, and unstable if they move away.", 1],
    ["taylor", "Taylor expansion", "Approximating a function near a point by its value plus its slope times the offset, plus smaller terms. It amounts to zooming in until a curve looks straight.", 1],
    ["phase-line", "Phase line", "A picture of a one-variable system: the x axis, with arrows showing which way states move and dots at the fixed points.", 1],
    ["bifurcation", "Bifurcation", "A parameter value at which the qualitative picture changes: fixed points appear, vanish or swap stability, or an oscillation is born.", 2],
    ["normal-form", "Normal form", "The simplest equation showing a given type of bifurcation. Close to the bifurcation, any system of that type behaves like its normal form.", 2],
    ["saddle-node", "Saddle-node bifurcation", "Two fixed points, one stable and one unstable, collide and disappear.", 2],
    ["bifurcation-diagram", "Bifurcation diagram", "A plot of a system's long-term states against a parameter. It shows where they split, merge or turn chaotic.", 6],
    ["eigenvalue", "Eigenvalue", "For a matrix A, a number λ with Av = λv for some direction v. Along v the linear system simply grows or shrinks like e<sup>λt</sup>. Complex eigenvalues mean rotation, and their real part is the growth rate.", 3],
    ["trace-det", "Trace and determinant", "For a 2×2 matrix [a b; c d], the trace is τ = a + d and the determinant is Δ = ad − bc. Together they fix both eigenvalues.", 3],
    ["saddle", "Saddle", "A fixed point that attracts along one direction and repels along another. It is always unstable.", 3],
    ["node", "Node", "A fixed point that trajectories approach or leave without circling.", 3],
    ["spiral", "Spiral", "A fixed point that trajectories approach or leave while circling around it.", 3],
    ["jacobian", "Jacobian", "The matrix of partial derivatives of the rule. At a fixed point it is the linear system that the nonlinear one looks like close up.", 4],
    ["hartman", "Hartman–Grobman theorem", "Near a fixed point where no eigenvalue has zero real part, a nonlinear system behaves like its linear approximation.", 4],
    ["divergence", "Divergence", "The sum ∂f/∂x + ∂g/∂y + …. It is the rate at which a small blob of initial conditions grows or shrinks in volume.", 4],
    ["liouville", "Liouville's theorem", "In a frictionless (Hamiltonian) system, every blob of initial conditions keeps its volume as it moves.", 4],
    ["conservative", "Conservative (Hamiltonian) system", "A system without friction that conserves energy. Phase-space volume is preserved, so there are no attractors.", 4],
    ["conserved", "Conserved quantity (integral of motion)", "A combination of the state variables, such as energy or angular momentum, that stays constant along every trajectory.", 4],
    ["attractor", "Attractor", "The set that trajectories settle onto over time: a fixed point, a cycle, or a strange attractor. Only dissipative systems have them.", 4],
    ["limit-cycle", "Limit cycle", "An isolated closed orbit that nearby trajectories spiral toward or away from. It is a self-sustained oscillation.", 5],
    ["hopf", "Hopf bifurcation", "A fixed point loses stability as a pair of complex eigenvalues crosses to a positive real part, and an oscillation is born. In the supercritical case a small stable cycle grows smoothly. In the subcritical case the system jumps somewhere far away.", 5],
    ["averaging", "Averaging", "An approximation for weakly nonlinear oscillators: average the slow change in amplitude over one fast cycle.", 5],
    ["uniqueness", "Uniqueness", "For a smooth rule, each state has exactly one future and one past, so trajectories can never cross.", 5],
    ["autonomous", "Autonomous", "The rule does not depend on the clock time itself: ẋ = f(x), not f(x, t). A pendulum pushed periodically is non-autonomous.", 5],
    ["poincare-bendixson", "Poincaré–Bendixson theorem", "In a plane, a bounded trajectory must end at a fixed point, a closed orbit, or a loop of fixed points. So two-dimensional flows can't be chaotic.", 5],
    ["map", "Map", "A rule that jumps from one state to the next, x<sub>n+1</sub> = g(x<sub>n</sub>), instead of flowing continuously.", 6],
    ["shift-map", "Shift map", "x → 10x mod 1. Multiplying by 10 shifts the decimal digits one place left, and mod 1 keeps only the fractional part.", 6],
    ["logistic", "Logistic map", "x<sub>n+1</sub> = r x<sub>n</sub>(1 − x<sub>n</sub>), a one-line population model that goes from steady to periodic to chaotic as r increases.", 6],
    ["feigenbaum", "Feigenbaum constant", "δ ≈ 4.669, the limiting ratio of successive gaps between period doublings. It is the same for every smooth one-hump map.", 6],
    ["invertible", "Invertible map", "A map that can be run backwards, because each state has exactly one predecessor. The logistic map is not invertible: two different values of x lead to the same next value.", 6],
    ["prandtl", "Prandtl number", "A fluid's viscosity divided by its thermal diffusivity: how quickly momentum spreads compared with heat.", 7],
    ["strange-attractor", "Strange attractor", "An attractor with a fractal structure on which the motion is chaotic.", 7],
    ["fractal-dimension", "Fractal dimension", "A dimension that need not be a whole number. The Lorenz attractor's 2.06 means it is slightly more than a surface: infinitely many layered sheets.", 7],
    ["lyapunov-exponent", "Lyapunov exponent λ", "The long-run average rate at which nearby trajectories separate: a small gap grows like e<sup>λt</sup>. A positive λ on a bounded orbit means chaos.", 8],
    ["quasi-periodic", "Quasi-periodic", "Motion that combines two or more frequencies with an irrational ratio. It never repeats exactly, yet it is not chaotic (λ = 0).", 8],
    ["ftle", "Finite-time Lyapunov exponent (FTLE)", "The same stretching rate measured over a finite window instead of forever. It varies along a trajectory.", 8],
    ["benettin", "Benettin method", "The standard way to compute λ: follow a trajectory and a close neighbour, record how much the gap grew, shrink it back to its starting size, and average the logarithms.", 8],
    ["lyapunov-time", "Lyapunov time", "1/λ, the time for a small error to grow by a factor of e ≈ 2.72. It is the natural clock of a chaotic system.", 9],
    ["tolerance", "Tolerance", "The largest error accepted before a prediction counts as useless. In these experiments it is 10% of the system's size.", 9],
    ["one-step-error", "One-step error ε", "How far a model's single-step prediction lands from the true next state.", 9],
    ["step-model", "Step model", "A neural network that predicts the next state from the current one. It is used by rolling it out.", 9],
    ["rollout", "Rollout", "Running a one-step model over and over, feeding each prediction back in as the next input.", 9],
    ["unroll", "Unroll training", "Training on the error after several model steps chained together, not just one, so the model learns to keep compounding errors small.", 9],
    ["direct-model", "Direct model", "A neural network that maps the initial condition straight to all future states at once, with no rollout.", 9],
    ["exact-twin", "Exact twin", "An exact solver started with an initial error equal to a model's one-step error. No model with that accuracy can do better.", 9],
    ["normal-modes", "Normal modes", "The independent simple oscillations of a linear system. Any small motion is a mix of them.", 10],
    ["torus", "Torus (plural tori)", "The doughnut-shaped surface in phase space that quasi-periodic motion winds around.", 10],
    ["kam", "KAM theory", "Kolmogorov–Arnold–Moser theory. In a frictionless system, most regular tori survive small perturbations, then break up as the perturbation grows. What remains is islands of order in a chaotic sea.", 10],
    ["lagrange-triangle", "Lagrange triangle", "Three masses at the corners of a rigidly rotating equilateral triangle. It is an exact solution, but unstable for equal masses.", 10],
    ["figure-eight", "Figure-eight orbit", "Three equal masses chasing each other around one figure-8 curve. Moore found it numerically in 1993, and Chenciner and Montgomery proved it exists in 2000. It is stable.", 10],
    ["close-encounter", "Close encounter", "Two bodies passing very near each other. Gravity grows like 1/r², so the motion changes violently in a very short time.", 10],
    ["kepler", "Kepler orbits", "The exact ellipse, parabola and hyperbola solutions of the two-body problem.", 10],
    ["pythagorean", "Pythagorean problem", "Burrau's 1913 setup: masses 3, 4 and 5 at the corners of a 3-4-5 right triangle, released from rest. After many close encounters, one body is ejected.", null],
    ["chaos", "Chaos", "Aperiodic long-term behaviour in a deterministic system with sensitive dependence on initial conditions.", 11],
    ["duffing", "Duffing oscillator", "A forced, damped spring with a cubic stiffness term, ẍ + δẋ + αx + βx³ = γ cos ωt. Only two variables, but the periodic forcing adds a dimension, so it can be chaotic.", 11],
    ["henon", "Hénon map", "x<sub>n+1</sub> = 1 − a x<sub>n</sub>² + y<sub>n</sub>, y<sub>n+1</sub> = b x<sub>n</sub>: a two-dimensional invertible map with a strange attractor.", 11],
    ["intermittency", "Intermittency", "Long stretches of nearly regular behaviour broken by irregular bursts.", 11],
    ["ruelle-takens", "Ruelle–Takens scenario", "The finding that two or three competing frequencies are enough to break down into chaos, with no need for an infinite cascade.", 11],
    ["homoclinic", "Homoclinic orbit", "A trajectory that leaves a saddle and comes back to the same saddle. When one forms, chaos can appear abruptly.", 11],
    ["rk4", "RK4 (Runge–Kutta)", "A standard fourth-order method for stepping a differential equation forward. It samples the slope four times per step and combines the results.", null],
    ["dop853", "DOP853", "An eighth-order Runge–Kutta method that adapts its step size to meet a requested error tolerance. It produces the exact reference solutions here.", null],
  ];
  const G = new Map(ENTRIES.map(([k, t, d, ch]) => [k, { t, d, ch }]));
  const local = () => document.querySelector("main")?.dataset.course === "101";

  const style = document.createElement("style");
  style.textContent = `
    a.term { color: inherit; text-decoration: underline dotted; text-decoration-color: var(--gl-accent, currentColor);
      text-underline-offset: 3px; text-decoration-thickness: 1.5px; cursor: help; }
    a.term:hover, a.term[aria-expanded="true"] { color: var(--gl-accent, inherit); }
    .gl-pop { position: absolute; z-index: 50; width: min(330px, calc(100vw - 32px)); background: var(--gl-bg, #fff);
      color: var(--gl-fg, #111); border: 1px solid var(--gl-rule, #ccc); border-radius: 8px; padding: 12px 34px 12px 14px;
      box-shadow: 0 8px 28px rgba(0,0,0,.18); font: 0.9rem/1.5 var(--gl-font, system-ui, sans-serif); }
    .gl-pop b { display: block; margin-bottom: 4px; }
    .gl-pop p { margin: 0 0 6px; color: var(--gl-muted, #555); }
    .gl-pop a { color: var(--gl-accent, inherit); font-size: 0.85rem; }
    .gl-pop button { position: absolute; top: 6px; right: 6px; border: 0; background: none; color: var(--gl-muted, #555);
      font-size: 1.1rem; line-height: 1; padding: 4px 6px; cursor: pointer; }`;
  document.head.appendChild(style);

  function chapterLink(ch) {
    if (ch == null) return "";
    return local()
      ? `<a href="#ch${ch}">Chapter ${ch} covers this →</a>`
      : `<a href="${COURSE}#ch${ch}" target="_blank" rel="noopener">Dynamical Systems 101, chapter ${ch} →</a>`;
  }

  const pop = document.createElement("div");
  pop.className = "gl-pop"; pop.hidden = true; pop.setAttribute("role", "dialog");
  let current = null;
  function hide() { if (current) current.setAttribute("aria-expanded", "false"); pop.hidden = true; current = null; }
  function show(el) {
    const g = G.get(el.dataset.term);
    if (!g) return;
    if (!pop.isConnected) document.body.appendChild(pop);
    pop.innerHTML = `<b>${g.t}</b><p>${g.d}</p>${chapterLink(g.ch)}<button type="button" aria-label="Close definition">×</button>`;
    pop.setAttribute("aria-label", g.t);
    pop.hidden = false;
    const r = el.getBoundingClientRect(), w = pop.offsetWidth;
    const left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left));
    pop.style.left = left + window.scrollX + "px";
    pop.style.top = r.bottom + window.scrollY + 6 + "px";
    if (current) current.setAttribute("aria-expanded", "false");
    current = el; el.setAttribute("aria-expanded", "true");
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest && e.target.closest("a.term");
    if (t) { e.preventDefault(); t === current ? hide() : show(t); return; }
    if (e.target.closest && e.target.closest(".gl-pop button")) { hide(); return; }
    if (!pop.contains(e.target)) hide();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && current) { const c = current; hide(); c.focus(); } });
  window.addEventListener("resize", hide);

  const list = document.getElementById("gloss-list");
  if (list) {
    const rows = [...G.entries()].sort((a, b) => a[1].t.localeCompare(b[1].t));
    list.innerHTML = rows.map(([k, g]) =>
      `<dt id="g-${k}">${g.t}</dt><dd>${g.d}${g.ch != null ? ` <a href="#ch${g.ch}">Chapter ${g.ch}</a>` : ""}</dd>`).join("");
  }
  const h = location.hash.slice(1);
  if (h && /^[A-Za-z0-9-]+$/.test(h)) { const el = document.getElementById(h); if (el) requestAnimationFrame(() => el.scrollIntoView()); }
})();
