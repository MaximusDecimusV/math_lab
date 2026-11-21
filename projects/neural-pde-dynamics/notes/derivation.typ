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
