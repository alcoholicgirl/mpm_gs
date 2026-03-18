A GPU-accelerated [Material Point Method](https://en.wikipedia.org/wiki/Material_point_method) (MPM) simulator.

The hyperelastic model governing the constitution model can be chosen as desired. By default it's Neo-Hookean.

Gi(x) stands the gaussian value for i-th ellipsoid at x.

Properties of an ellipsoid is made of:
1. position, rotation, scale, SH (just like most gaussian splatting implementations do).
2. velocity, density (this might be a global variable).

The process of the MPM solver:
1. P2G: For each gaussian ellipsoid, accumulate both velocity and mass onto a Eulerian field according to Gi(x). Now momentum can be derived.
2. Update deformation gradient: F_new = (I + dt * ∇v) * F.
3. G2P: Now that new velocity field is updated, we're able to update momentum for each ellipsoid, using Cauchy Momentum Equation. The integral of G(x) = exp(-1/2 xTx) is (2π)^{3/2} √(det Σ) (V as in M = V * rho). 