#set page(margin: 1in)
#set math.equation(numbering: "(1)")
#set heading(numbering: "1.")

= Viscous Burgers: reference solver

== The equation

On the periodic domain $x in [0, 2 pi)$ with viscosity $nu > 0$:
$ u_t + u u_x = nu u_(x x) $ <burgers>

Two invariants to check against. Mass $integral u dif x$ is conserved exactly.
Multiplying by $u$ and integrating by parts gives energy decay:
$ dif / (dif t) integral u^2 / 2 dif x = -nu integral u_x^2 dif x <= 0 $ <energy>

== Exact solution via Cole-Hopf

Set $u = -2 nu (log phi)_x$. Then @burgers reduces to the heat equation
$phi_t = nu phi_(x x)$. Taking $phi = b + e^(-nu t) cos x$ with $b > 1$ gives
$ u(x, t) = (2 nu e^(-nu t) sin x) / (b + e^(-nu t) cos x) $ <colehopf>

== Discretization

Fourier modes $hat(u)_k$. In conservative form, with 2/3-rule dealiasing $P$:
$ hat(u)_t = -nu k^2 hat(u) + N(hat(u)), quad N(hat(u)) = -(i k) / 2 P cal(F)[u^2] $ <split>

Integrating-factor RK4 (Lawson). With $E = e^(-nu k^2 Delta t)$ and $E_2 = e^(-nu k^2 Delta t / 2)$:
$
  a &= N(hat(u)^n), quad
  b = N(E_2 (hat(u)^n + Delta t / 2 a)), \
  c &= N(E_2 hat(u)^n + Delta t / 2 b), quad
  d = N(E hat(u)^n + Delta t E_2 c), \
  hat(u)^(n+1) &= E hat(u)^n + (Delta t) / 6 (E a + 2 E_2 (b + c) + d)
$ <lawson>

Diffusion is integrated exactly, so stability is set by the nonlinear term
rather than by $nu k_max^2$. The $k = 0$ mode of $N$ is identically zero, so
mass is conserved to round-off.

== Checks (tests/test_burgers.py)

+ Agreement with @colehopf to $10^(-8)$ at $t = 1$.
+ Mass conserved to $10^(-12)$.
+ Energy non-increasing, per @energy.
+ Observed temporal order $approx 4$, measured against a fine-$Delta t$ run so spatial error cancels.
  (Against @colehopf directly, the spatial error floor of $10^(-10)$ swamps the time error.)

= Neural solvers with hard constraints

Both solvers in `src/neural_pde/pinn.py` follow Lagaris, Likas & Fotiadis (1998).
Write the unknown as a *trial function* that satisfies the boundary and initial
conditions for any network $N_theta$. The only loss left is the PDE residual at
collocation points. There are no penalty weights to tune.

== Laplace on the unit square (the original AMATH 581 problem)

$ u_(x x) + u_(y y) = 0 "on" (0,1)^2, quad u(x, 1) = sin pi x, quad u = 0 "on the other three sides" $ <laplace>
$ u(x, y) = (sin pi x sinh pi y) / (sinh pi) $ <laplace-exact>
$ psi(x, y) = y sin pi x + x (1 - x) y (1 - y) N_theta (x, y) $ <laplace-trial>

The loss is the mean of $(psi_(x x) + psi_(y y))^2$ over a $24 times 24$ interior
grid. The second derivatives come from `torch.autograd`. Training is Adam with
cosine decay, then L-BFGS.

*What went wrong in the original.* The update was `W -= lr * (lr * grad)` with
$"lr" = 10^(-3)$, an effective step of $10^(-6)$, run for only 10 iterations.
Training effectively never started.

== Burgers PINN on the periodic domain

The trial function is
$ u(x, t) = u_0(x) + t N_theta (sin x, cos x, t \/ t_f) $ <burgers-trial>
Feeding $sin x, cos x$ instead of $x$ makes $u$ exactly $2 pi$-periodic, and the
factor $t$ makes $u(x, 0) = u_0(x)$ exactly. The loss is the mean of
$(u_t + u u_x - nu u_(x x))^2$ over fresh random collocation points
$(x, t) in [0, 2pi) times [0, t_f]$ at each Adam step, followed by L-BFGS on a
fixed set.

The checks compare against @colehopf for $nu = 0.1$, and against the spectral
solver for $nu = 0.02$, where a steep front forms.
