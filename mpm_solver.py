"""
Gaussian-kernel MPM solver
"""

import numpy as np
import taichi as ti

from ply_loader import GaussianCloud


@ti.data_oriented
class MPMSolver:
    def __init__(
        self,
        n_particles: int,
        grid_res: int = 128,
        dt: float = 1e-4,
        youngs_modulus: float = 1e4,
        poisson_ratio: float = 0.3,
        rho: float = 1000.0,
        gravity: tuple = (0.0, 0.0, -9.8),
        max_radius: int = 0,
        c2_ratio: float = 0.0,
        floor_friction: float = 0.0,
        boundary_thickness: float = 4.0 / 64.0,
    ):
        self.n_particles = n_particles
        self.grid_res = grid_res
        self.dt = dt
        self.rho = rho
        self.max_r_cap = max_radius      # 0 means adaptive / uncapped

        mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        # Mooney-Rivlin: C1 + C2 = mu/2; c2_ratio=0 → pure Neo-Hookean
        self.C1 = 0.5 * mu * (1.0 - c2_ratio)
        self.C2 = 0.5 * mu * c2_ratio
        self.kappa = (
            youngs_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )

        self.dx = 1.0 / grid_res
        self.inv_dx = float(grid_res)
        self.gravity = ti.Vector(list(gravity), dt=ti.f32)
        self.floor_friction = max(0.0, float(floor_friction))
        self.boundary_thickness = max(0.0, float(boundary_thickness))

        self.x       = ti.Vector.field(3,    ti.f32, n_particles)
        self.v       = ti.Vector.field(3,    ti.f32, n_particles)
        self.F       = ti.Matrix.field(3, 3, ti.f32, n_particles)
        self.C       = ti.Matrix.field(3, 3, ti.f32, n_particles) 
        self.m       = ti.field(ti.f32, n_particles)
        self.V_p     = ti.field(ti.f32, n_particles)
        self.inv_cov = ti.Matrix.field(3, 3, ti.f32, n_particles)
        self.Z_field = ti.field(ti.f32, n_particles)
        self.max_r_field = ti.field(ti.i32, n_particles)   # per-particle stencil radius

        self.grid_mv = ti.Vector.field(3, ti.f32, shape=(grid_res,) * 3)
        self.grid_m  = ti.field(ti.f32,           shape=(grid_res,) * 3)

    # ─────────────────────────────────────────────────────────────────
    def init_from_cloud(
        self,
        cloud: GaussianCloud,
        scene_scale: float = 1.0,
        scene_offset: np.ndarray | None = None,
        kernel_scale: float = 1.0,
    ):
        world_pos = cloud.positions.astype(np.float32)
        if scene_offset is None:
            scene_offset = world_pos.mean(axis=0)
        scene_offset = scene_offset.astype(np.float32)

        centered_pos = world_pos - scene_offset[None, :]
        max_extent = float(np.percentile(np.abs(centered_pos), 99))
        world_to_sim = (scene_scale / (max_extent * 2.0)) if max_extent > 0 else 1.0
        sim_pos = centered_pos * world_to_sim + 0.5

        self.x.from_numpy(sim_pos)
        self.v.from_numpy(np.zeros((self.n_particles, 3), np.float32))
        self.F.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (self.n_particles, 1, 1))
        )
        self.C.from_numpy(
            np.zeros((self.n_particles, 3, 3), dtype=np.float32)
        )

        self._scene_offset     = scene_offset
        self._scene_max_extent = max_extent
        self._scene_scale      = scene_scale

        q = cloud.rotations.astype(np.float32)
        w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = (
            np.stack(
                [
                    1 - 2 * (y_ * y_ + z_ * z_),
                    2 * (x_ * y_ - w_ * z_),
                    2 * (x_ * z_ + w_ * y_),
                    2 * (x_ * y_ + w_ * z_),
                    1 - 2 * (x_ * x_ + z_ * z_),
                    2 * (y_ * z_ - w_ * x_),
                    2 * (x_ * z_ - w_ * y_),
                    2 * (y_ * z_ + w_ * x_),
                    1 - 2 * (x_ * x_ + y_ * y_),
                ],
                axis=-1,
            )
            .reshape(-1, 3, 3)
            .astype(np.float32)
        )

        sigma_sim_raw = np.maximum(cloud.scales.astype(np.float32) * world_to_sim, 1e-8)

        # The transfer kernel still needs a grid-aware lower bound so that the
        # discrete stencil does not collapse when a render Gaussian is much
        # smaller than one cell.
        sigma_transfer = np.maximum(sigma_sim_raw * kernel_scale, 1.0 * self.dx)
        inv_s2  = 1.0 / sigma_transfer ** 2
        inv_cov = np.einsum("nij,nj,nkj->nik", R, inv_s2, R).astype(np.float32)

        # Physical mass/volume should not be tied to dx. We use a
        # resolution-independent lower bound based on the average particle
        # spacing in normalized simulation space to avoid near-zero masses from
        # tiny render-only splats.
        bbox_extent = np.maximum(sim_pos.max(axis=0) - sim_pos.min(axis=0), 1e-6)
        bbox_volume = float(np.prod(bbox_extent))
        mean_spacing = float((bbox_volume / max(self.n_particles, 1)) ** (1.0 / 3.0))
        sigma_mass_floor = max(0.5 * mean_spacing, 1e-6)
        sigma_mass = np.maximum(sigma_sim_raw, sigma_mass_floor)

        TWO_PI_32 = float((2.0 * np.pi) ** 1.5)
        V_arr = (TWO_PI_32 * sigma_mass[:, 0] * sigma_mass[:, 1] * sigma_mass[:, 2]).astype(np.float32)
        m_arr = (self.rho * V_arr * cloud.opacities).astype(np.float32)

        # per-particle stencil radius: ceil(3σ_max / dx). By default this is
        # fully adaptive; an optional global cap can still be applied for
        # performance experiments.
        sigma_max  = sigma_transfer.max(axis=1)                            # grid space
        max_r_arr  = np.ceil(3.0 * sigma_max * self.inv_dx).astype(np.int32)
        max_r_arr  = np.maximum(max_r_arr, 1).astype(np.int32)
        if self.max_r_cap > 0:
            max_r_arr = np.minimum(max_r_arr, self.max_r_cap).astype(np.int32)

        self.inv_cov.from_numpy(inv_cov)
        self.V_p.from_numpy(V_arr)
        self.m.from_numpy(m_arr)
        self.max_r_field.from_numpy(max_r_arr)

        # M₀ in world space for renderer deformation: M₀[n] = R[n] @ diag(s_world[n])
        # (R * s_world[:, None, :]) broadcasts as R[n,i,j] * s_world[n,j]  ✓
        s_world = cloud.scales.astype(np.float32)
        self._M0_world = (R * s_world[:, None, :]).astype(np.float32)  # (N, 3, 3)

    def get_positions_world(self) -> np.ndarray:
        pos = self.x.to_numpy()
        pos -= 0.5
        if self._scene_max_extent > 0:
            pos *= self._scene_max_extent * 2.0 / self._scene_scale
        pos += self._scene_offset
        return pos

    def get_deformed_M_world(self) -> np.ndarray:
        F = self.F.to_numpy()
        return np.einsum("nij,njk->nik", F, self._M0_world)

    @ti.kernel
    def _reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_mv[I] = ti.Vector.zero(ti.f32, 3)
            self.grid_m[I]  = 0.0

    @ti.kernel
    def _p2g(self):
        for p in range(self.n_particles):
            xp  = self.x[p]
            ic  = self.inv_cov[p]
            mr  = self.max_r_field[p]
            base = ti.cast(xp * self.inv_dx, ti.i32)

            # Mooney-Rivlin Kirchhoff stress
            # τ = 2C1(b−I) + 2C2(I1·b − b² − 2I) + κ·ln(J)·I
            Fp  = self.F[p]
            J   = ti.max(Fp.determinant(), 0.01)
            b   = Fp @ Fp.transpose()
            I1  = b.trace()
            eye = ti.Matrix.identity(ti.f32, 3)
            stress = (2.0 * self.C1 * (b - eye)
                      + 2.0 * self.C2 * (I1 * b - b @ b - 2.0 * eye)
                      + self.kappa * ti.log(J) * eye)

            # normalisation Z  (per-particle stencil radius)
            Z = 0.0
            for di in range(2 * mr + 1):
                for dj in range(2 * mr + 1):
                    for dk in range(2 * mr + 1):
                        cell = base + ti.Vector([di - mr, dj - mr, dk - mr])
                        if (0 <= cell[0] < self.grid_res
                                and 0 <= cell[1] < self.grid_res
                                and 0 <= cell[2] < self.grid_res):
                            d    = ti.cast(cell, ti.f32) * self.dx - xp
                            maha = d.dot(ic @ d)
                            if maha <= 9.0:
                                Z += ti.exp(-0.5 * maha)
            self.Z_field[p] = Z

            if Z > 1e-12:
                # Gradient correction for the discretely normalized Gaussian:
                # grad w_i = w_i * (A d_i - sum_j w_j A d_j).
                g_bar = ti.Vector.zero(ti.f32, 3)
                for di in range(2 * mr + 1):
                    for dj in range(2 * mr + 1):
                        for dk in range(2 * mr + 1):
                            cell = base + ti.Vector([di - mr, dj - mr, dk - mr])
                            if (0 <= cell[0] < self.grid_res
                                    and 0 <= cell[1] < self.grid_res
                                    and 0 <= cell[2] < self.grid_res):
                                d    = ti.cast(cell, ti.f32) * self.dx - xp
                                Ad   = ic @ d
                                maha = d.dot(Ad)
                                if maha <= 9.0:
                                    w = ti.exp(-0.5 * maha) / Z
                                    g_bar += w * Ad

                for di in range(2 * mr + 1):
                    for dj in range(2 * mr + 1):
                        for dk in range(2 * mr + 1):
                            cell = base + ti.Vector([di - mr, dj - mr, dk - mr])
                            if (0 <= cell[0] < self.grid_res
                                    and 0 <= cell[1] < self.grid_res
                                    and 0 <= cell[2] < self.grid_res):
                                d    = ti.cast(cell, ti.f32) * self.dx - xp
                                Ad   = ic @ d
                                maha = d.dot(Ad)
                                if maha <= 9.0:
                                    w = ti.exp(-0.5 * maha) / Z
                                    grad_w = w * (Ad - g_bar)
                                    self.grid_mv[cell] += (
                                        w * self.m[p] * (self.v[p] + self.C[p] @ d)
                                        - self.dt * self.V_p[p] * stress @ grad_w
                                    )
                                    self.grid_m[cell] += w * self.m[p]

    @ti.kernel
    def _grid_update(self):
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0.0:
                v  = self.grid_mv[I] / m
                v += self.dt * self.gravity
                node_pos = ti.cast(I, ti.f32) * self.dx
                if node_pos[2] < self.boundary_thickness and v[2] < 0.0:
                    vn = -v[2]
                    vt = ti.Vector([v[0], v[1], 0.0])
                    vt_norm = ti.sqrt(vt.dot(vt) + 1e-12)
                    max_drop = self.floor_friction * vn
                    scale = ti.max(0.0, 1.0 - max_drop / vt_norm)
                    v[0] = vt[0] * scale
                    v[1] = vt[1] * scale
                    v[2] = 0.0
                for d in ti.static(range(3)):
                    if node_pos[d] < self.boundary_thickness and v[d] < 0:
                        v[d] = 0.0
                    if node_pos[d] > 1.0 - self.boundary_thickness and v[d] > 0:
                        v[d] = 0.0
                self.grid_mv[I] = v

    @ti.kernel
    def _g2p(self):
        for p in range(self.n_particles):
            xp  = self.x[p]
            ic  = self.inv_cov[p]
            mr  = self.max_r_field[p]
            base = ti.cast(xp * self.inv_dx, ti.i32)
            Z   = self.Z_field[p]

            mean_v = ti.Vector.zero(ti.f32, 3)
            mean_d = ti.Vector.zero(ti.f32, 3)
            wsum   = 0.0
            C_p    = ti.Matrix.zero(ti.f32, 3, 3)

            if Z > 1e-12:
                # First pass: weighted means for the local affine MLS fit
                for di in range(2 * mr + 1):
                    for dj in range(2 * mr + 1):
                        for dk in range(2 * mr + 1):
                            cell = base + ti.Vector([di - mr, dj - mr, dk - mr])
                            if (0 <= cell[0] < self.grid_res
                                    and 0 <= cell[1] < self.grid_res
                                    and 0 <= cell[2] < self.grid_res):
                                d    = ti.cast(cell, ti.f32) * self.dx - xp
                                maha = d.dot(ic @ d)
                                if maha <= 9.0:
                                    w  = ti.exp(-0.5 * maha) / Z
                                    gv = self.grid_mv[cell]
                                    wsum   += w
                                    mean_v += w * gv
                                    mean_d += w * d

                if wsum > 1e-12:
                    mean_v /= wsum
                    mean_d /= wsum

                    cov_vd = ti.Matrix.zero(ti.f32, 3, 3)
                    cov_dd = ti.Matrix.zero(ti.f32, 3, 3)

                    # Second pass: solve for the best affine map over the
                    # actual discrete stencil instead of using the continuous
                    # Gaussian gradient as a proxy.
                    for di in range(2 * mr + 1):
                        for dj in range(2 * mr + 1):
                            for dk in range(2 * mr + 1):
                                cell = base + ti.Vector([di - mr, dj - mr, dk - mr])
                                if (0 <= cell[0] < self.grid_res
                                        and 0 <= cell[1] < self.grid_res
                                        and 0 <= cell[2] < self.grid_res):
                                    d    = ti.cast(cell, ti.f32) * self.dx - xp
                                    maha = d.dot(ic @ d)
                                    if maha <= 9.0:
                                        w  = ti.exp(-0.5 * maha) / Z
                                        gv = self.grid_mv[cell]
                                        dd = d - mean_d
                                        dv = gv - mean_v
                                        cov_vd += w * dv.outer_product(dd)
                                        cov_dd += w * dd.outer_product(dd)

                    reg = 1e-6 * ti.max(cov_dd.trace(), self.dx * self.dx)
                    cov_dd += reg * ti.Matrix.identity(ti.f32, 3)
                    C_p = cov_vd @ cov_dd.inverse()

            new_v = mean_v - C_p @ mean_d
            self.C[p] = C_p
            self.v[p] = new_v
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * self.C[p]) @ self.F[p]

            # clamp singular values
            svd_u, svd_s, svd_v = ti.svd(self.F[p])
            for svd_i in ti.static(range(3)):
                svd_s[svd_i, svd_i] = ti.min(ti.max(svd_s[svd_i, svd_i], 0.1), 10.0)
            self.F[p] = svd_u @ svd_s @ svd_v.transpose()

            self.x[p] += self.dt * new_v

            lo = self.boundary_thickness
            hi = 1.0 - self.boundary_thickness
            for d in ti.static(range(3)):
                if self.x[p][d] < lo:
                    self.x[p][d] = lo
                    if self.v[p][d] < 0.0:
                        if d == 2 and self.floor_friction > 0.0:
                            vn = -self.v[p][d]
                            vt = ti.Vector([self.v[p][0], self.v[p][1]])
                            vt_norm = ti.sqrt(vt.dot(vt) + 1e-12)
                            max_drop = self.floor_friction * vn
                            scale = ti.max(0.0, 1.0 - max_drop / vt_norm)
                            self.v[p][0] = vt[0] * scale
                            self.v[p][1] = vt[1] * scale
                        self.v[p][d] = 0.0
                if self.x[p][d] > hi:
                    self.x[p][d] = hi
                    if self.v[p][d] > 0.0:
                        self.v[p][d] = 0.0

    def substep(self):
        self._reset_grid()
        self._p2g()
        self._grid_update()
        self._g2p()

    def step(self, n_substeps: int = 1):
        for _ in range(n_substeps):
            self.substep()
