# Stress Gradient Consistency in Gaussian-Grid Transfer

This note explains why the current stress transfer in `mpm_solver.py` is not fully consistent with the discrete Gaussian weights used by the solver, and why that inconsistency can make the material response depend too much on `grid_res`.

## 1. What the solver currently does

For one particle `p`, let

\[
\phi_i = \exp\left(-\frac12 d_i^\top A_p d_i\right),
\qquad
d_i = x_i - x_p,
\qquad
A_p = \Sigma_p^{-1}.
\]

The solver uses the **discretely normalized** weights

\[
w_i = \frac{\phi_i}{Z},
\qquad
Z = \sum_j \phi_j,
\]

over the finite particle stencil.

This is the weight actually used for mass and momentum transfer.

In the current implementation, the stress force term is applied as if

\[
\nabla w_i \approx w_i A_p d_i.
\]

That is the gradient of the unnormalized continuous Gaussian up to sign, but it is **not** the true gradient of the discretely normalized weight \(w_i = \phi_i / Z\).

## 2. Why this is inconsistent

Because \(Z\) depends on particle position, the derivative of \(w_i\) must include the derivative of the denominator:

\[
\nabla w_i
=
\nabla \left(\frac{\phi_i}{Z}\right)
=
\frac{\nabla \phi_i}{Z}
-
\frac{\phi_i}{Z^2}\nabla Z.
\]

Using

\[
\nabla \phi_i = \phi_i A_p d_i,
\qquad
\nabla Z = \sum_j \nabla \phi_j = \sum_j \phi_j A_p d_j,
\]

we get

\[
\nabla w_i
=
w_i \left(A_p d_i - \sum_j w_j A_p d_j\right).
\]

Define

\[
\bar g_p = \sum_j w_j A_p d_j.
\]

Then the consistent discrete gradient is

\[
\nabla w_i = w_i (A_p d_i - \bar g_p).
\]

## 3. What is wrong with the current approximation

The current solver omits the correction term \(-w_i \bar g_p\).

So the stress force uses

\[
\nabla w_i^{\text{current}} \approx w_i A_p d_i
\]

instead of

\[
\nabla w_i^{\text{correct}} = w_i (A_p d_i - \bar g_p).
\]

This matters because the discrete weights satisfy

\[
\sum_i w_i = 1.
\]

Therefore their gradients must satisfy

\[
\sum_i \nabla w_i = 0.
\]

With the corrected formula:

\[
\sum_i \nabla w_i
=
\sum_i w_i (A_p d_i - \bar g_p)
=
\sum_i w_i A_p d_i - \bar g_p \sum_i w_i
=
\bar g_p - \bar g_p
=
0.
\]

But with the current approximation:

\[
\sum_i \nabla w_i^{\text{current}}
=
\sum_i w_i A_p d_i
=
\bar g_p,
\]

which is generally not zero.

So the current internal force is not fully self-consistent with the discrete partition of unity.

## 4. Physical consequence

Internal stress should only generate internal force balance. In discrete form, inconsistent gradients can create:

- spurious net force
- spurious torque
- stronger dependence on stencil truncation
- stronger dependence on grid resolution
- more artifacts near boundaries
- easier instability when `grid_res` increases

This is one reason why simulations can look materially different when changing from `64` to `128`, even if the constitutive parameters are unchanged.

It is not the only reason. Fixed `dt` with smaller `dx` also makes explicit integration less stable. But the stress-gradient inconsistency is a real implementation issue on top of that.

## 5. The stress term that should be used

The current code effectively applies

\[
\Delta (mv)_i^{\text{stress}}
=
-\Delta t \, V_p \, \tau_p \, \left(w_i A_p d_i\right).
\]

The corrected version should use

\[
\Delta (mv)_i^{\text{stress}}
=
-\Delta t \, V_p \, \tau_p \, \nabla w_i
=
-\Delta t \, V_p \, \tau_p \, \left[w_i (A_p d_i - \bar g_p)\right].
\]

Equivalently,

\[
\Delta (mv)_i^{\text{stress}}
=
-\Delta t \, V_p \, \tau_p \, w_i (A_p d_i - \bar g_p).
\]

## 6. Minimal implementation strategy

Inside `P2G`, compute three things per particle:

1. `Z = sum(phi_i)`
2. `g_bar = sum(w_i * (A d_i))`
3. scatter using `grad_w_i = w_i * ((A d_i) - g_bar)`

In pseudocode:

```python
# pass 1
Z += phi

# pass 2
w = phi / Z
g_bar += w * (A @ d)

# pass 3
w = phi / Z
grad_w = w * ((A @ d) - g_bar)
grid_mv[cell] += w * m_p * (v_p + C_p @ d) - dt * V_p * (stress @ grad_w)
grid_m[cell] += w * m_p
```

This is the smallest correction that keeps the Gaussian kernel model and makes the stress force consistent with the discrete normalized weights already used elsewhere.

## 7. Important clarification

This correction does **not** replace the Gaussian body model with a standard MLS node shape function.

It still fully depends on the particle Gaussian:

- center \(x_p\)
- covariance \(\Sigma_p\)
- anisotropy
- support directionality

So this is still a Gaussian-based transfer, just with a gradient consistent with the actual discrete normalized kernel.

## 8. Bottom line

The issue is not that the constitutive law is wrong. The issue is that:

- mass/momentum transfer uses the discrete normalized Gaussian weights \(w_i\)
- stress transfer uses an approximation closer to the gradient of the unnormalized continuous Gaussian

Those two should come from the same discrete shape function.

The minimal fix is:

\[
\nabla w_i \leftarrow w_i \left(A_p d_i - \sum_j w_j A_p d_j\right).
\]

That should reduce spurious grid dependence and make `64` vs `128` behavior more consistent.
