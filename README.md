# mpm-gs

GPU-accelerated Material Point Method simulation for 3D Gaussian Splatting assets.

This project treats each 3D Gaussian as both:

- a render primitive for Gaussian splatting
- an MPM material point carrying mass, velocity, and deformation

The result is a compact pipeline:

`PLY Gaussians -> Gaussian-kernel MPM -> Gaussian rasterization -> PNG / MP4`

## Overview

The input is a standard 3DGS `.ply` containing per-Gaussian:

- position
- opacity
- color from SH DC coefficients
- anisotropic scale
- rotation quaternion

At load time, these Gaussians are converted into a `GaussianCloud`. For simulation, each Gaussian is interpreted as an anisotropic material particle with:

- world-space center
- mass `m_p`
- volume `V_p`
- deformation gradient `F_p`
- affine velocity field `v_p + C_p (x - x_p)`
- anisotropic Gaussian kernel `G_p(x)`

The same Gaussian parameters are also reused for rendering.

## Core Idea

This is not a classic B-spline MPM implementation. Instead, particle-grid transfer uses each Gaussian's own anisotropic kernel.

For particle `p`, the kernel is:

`G_p(x) ~ exp(-0.5 * (x - x_p)^T Sigma_p^{-1} (x - x_p))`

where `Sigma_p` comes from the Gaussian's scale and rotation. In code, the inverse covariance is stored per particle and the support is truncated to a finite stencil.

That means each particle is already an extended body, not a point sample with a separate interpolation kernel bolted on top.

The stencil radius is computed per particle as

$$
r_p = \left\lceil \frac{3 \sigma_{\max,p}}{\Delta x} \right\rceil
$$

so finer grids naturally use wider grid-space support. A manual cap can still be applied from the CLI for performance experiments, but the default is fully adaptive support.

## Simulation State

The solver tracks:

- particle position `x`
- particle velocity `v`
- deformation gradient `F`
- affine velocity matrix `C`
- particle mass `m`
- particle volume `V_p`
- inverse covariance `inv_cov`
- grid mass `grid_m`
- grid momentum / velocity accumulator `grid_mv`

The grid lives in normalized simulation space `[0, 1]^3`. Input geometry is centered, scaled into the simulation box, then mapped back to world space for rendering.

## One Substep

Each MPM substep is:

1. Reset grid
2. P2G transfer
3. Grid update
4. G2P transfer

### 1. P2G

For each particle, mass and momentum are splatted to neighboring grid nodes using the particle's anisotropic Gaussian weights.

Current momentum transfer uses a local affine velocity field:

`v(x) ~= v_p + C_p (x - x_p)`

so grid momentum receives:

`w_ip * m_p * (v_p + C_p d_ip)`

with `d_ip = x_i - x_p`.

The same pass also applies elastic forces through the particle stress:

`-dt * V_p * w_ip * stress_p * (inv_cov_p * d_ip)`

The grid weights are normalized over the truncated stencil of each particle.

### 2. Grid Update

After P2G, each active grid node is converted from momentum to velocity:

`v_i = grid_mv_i / grid_m_i`

Then the solver applies:

- gravity
- simple sticky/separating box boundary conditions near the simulation domain border

### 3. G2P

Grid velocity is transferred back to particles in two parts:

- particle center velocity
- local affine velocity matrix `C`

The current implementation computes `C` with an MLS-style local affine fit over the actual discrete stencil, not with the old continuous Gaussian-gradient approximation.

For particle `p`, let:

- `x_p` be the particle center
- `x_i` be a neighboring grid node
- `d_i = x_i - x_p`
- `v_i` be the grid velocity at node `i`
- `w_i` be the normalized Gaussian weight over the truncated stencil

We fit a local affine velocity field

$$
v(x) \approx v_p + C_p (x - x_p)
$$

by minimizing a weighted least-squares objective:

$$
\min_{v_p,\, C_p}
\sum_{i \in \mathcal{N}(p)}
w_i
\left\|
v_i - \left(v_p + C_p d_i\right)
\right\|^2
$$

where `\mathcal{N}(p)` is the particle's discrete support on the grid.

Because the implementation uses normalized weights over the stencil, the fit is solved in two passes.

First compute weighted means:

$$
\bar{v}_p = \frac{\sum_i w_i v_i}{\sum_i w_i},
\qquad
\bar{d}_p = \frac{\sum_i w_i d_i}{\sum_i w_i}
$$

Then build the discrete cross-covariance and second-moment matrices:

$$
\mathrm{cov}_{vd}
=
\sum_i
w_i
\left(v_i - \bar{v}_p\right)
\left(d_i - \bar{d}_p\right)^T
$$

$$
\mathrm{cov}_{dd}
=
\sum_i
w_i
\left(d_i - \bar{d}_p\right)
\left(d_i - \bar{d}_p\right)^T
$$

Finally solve for the best affine map:

$$
C_p = \mathrm{cov}_{vd}\,\mathrm{cov}_{dd}^{-1}
$$

and recover the particle-center velocity:

$$
v_p = \bar{v}_p - C_p \bar{d}_p
$$

In code, a small diagonal regularizer is added to `\mathrm{cov}_{dd}` before inversion to avoid singular fits when the local stencil is poorly conditioned.

This is an MLS approximation in the sense that the local affine field is fitted directly from the discrete sampled neighborhood. That matters because this solver does not operate on an ideal continuous kernel:

- the Gaussian support is truncated
- the neighborhood is sampled on a finite Cartesian grid
- the support may be asymmetric near boundaries
- weights are renormalized over the surviving stencil

So the current `C_p` is not the continuous Gaussian gradient proxy used by the older version. Instead, it is the best affine approximation to the actual discrete grid velocity field seen by the particle.

### 4. Deformation Update

After G2P, the deformation gradient is updated explicitly:

`F_{n+1} = (I + dt * C_p) * F_n`

Then singular values are clamped to keep `F` from becoming numerically extreme.

## Constitutive Model

The current stress model is Mooney-Rivlin / Neo-Hookean-style, parameterized by:

- Young's modulus
- Poisson ratio
- `c2_ratio`

The solver computes a Kirchhoff stress from `F` and injects it during P2G.

## Rendering

Rendering is separate from simulation.

- Particle positions are always updated from the solver.
- If `--deform_render` is disabled, Gaussians keep their original covariance for rendering.
- If `--deform_render` is enabled, rendering uses `F_p` to deform each Gaussian's covariance ellipsoid.

This means `deform_render` changes visualization only. It does not change the simulation itself.

The renderer:

1. projects Gaussians into screen space
2. converts 3D covariance to 2D covariance
3. depth-sorts splats
4. builds tile lists
5. alpha-composites them in a Taichi rasterizer

## Current Numerical Character

This codebase is still an experimental simulator. A few implementation choices matter when interpreting behavior:

- time integration is explicit
- finer grids usually require smaller `dt`
- `deform_render` can visually amplify errors already present in `F`
- particle-grid transfer is Gaussian-kernel-based rather than standard MPM shape functions
- boundary handling is simple and not angular-momentum preserving

So if motion becomes unstable when `grid_res` increases, that is expected behavior for an explicit elastic method unless the timestep is reduced accordingly.

## Main Files

- `main.py`: CLI and simulation/render loop
- `ply_loader.py`: load `.ply` Gaussians into arrays
- `mpm_solver.py`: Gaussian-kernel MPM solver
- `renderer.py`: Gaussian projection, sorting, tiling, rasterization
- `argsort.py`: Taichi radix sort used by the renderer
- `video_export.py`: save PNG frames and compile MP4

## Usage

Example:

```bash
python main.py --ply assets/ficus.ply --gpu --video output.mp4
```

Useful flags:

- `--frames`: number of rendered frames
- `--substeps`: MPM substeps per rendered frame
- `--dt`: timestep per substep
- `--grid_res`: simulation grid resolution
- `--max_radius`: optional cap on Gaussian support radius in grid cells; `0` keeps it adaptive
- `--youngs`: Young's modulus
- `--poisson`: Poisson ratio
- `--c2_ratio`: Mooney-Rivlin parameter split
- `--deform_render`: render deformed Gaussian covariances
- `--opacity_thresh`: drop low-opacity Gaussians before simulation

## Practical Reading Order

If you want to understand the code quickly, read in this order:

1. `main.py`
2. `ply_loader.py`
3. `mpm_solver.py`
4. `renderer.py`
5. `argsort.py`
