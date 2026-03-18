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
        max_radius: int = 4,
    ):
        self.n_particles = n_particles
        self.grid_res = grid_res
        self.dt = dt
        self.rho = rho
        self.max_r = max_radius

        self.mu_0 = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
        self.lambda_0 = (
            youngs_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )

        self.dx = 1.0 / grid_res
        self.inv_dx = float(grid_res)
        self.gravity = ti.Vector(list(gravity), dt=ti.f32)

        # ── particle fields ───────────────────────────────────────────
        self.x = ti.Vector.field(3, ti.f32, n_particles)
        self.v = ti.Vector.field(3, ti.f32, n_particles)
        self.F = ti.Matrix.field(3, 3, ti.f32, n_particles)
        self.m = ti.field(ti.f32, n_particles)
        self.V_p = ti.field(ti.f32, n_particles)
        self.inv_cov = ti.Matrix.field(3, 3, ti.f32, n_particles)
        self.Z_field = ti.field(ti.f32, n_particles)

        # ── grid fields ───────────────────────────────────────────────
        self.grid_mv = ti.Vector.field(3, ti.f32, shape=(grid_res,) * 3)
        self.grid_m = ti.field(ti.f32, shape=(grid_res,) * 3)

    # ─────────────────────────────────────────────────────────────────
    def init_from_cloud(
        self,
        cloud: GaussianCloud,
        scene_scale: float = 1.0,
        scene_offset: np.ndarray | None = None,
        kernel_scale: float = 1.0,
    ):
        pos = cloud.positions.copy().astype(np.float32)
        if scene_offset is None:
            scene_offset = pos.mean(axis=0)
        pos -= scene_offset
        # max_extent = np.abs(pos).max()
        max_extent = float(np.percentile(np.abs(pos), 99)) 
        if max_extent > 0:
            pos /= max_extent * 2.0 / scene_scale
        pos += 0.5

        self.x.from_numpy(pos)
        self.v.from_numpy(np.zeros((self.n_particles, 3), np.float32))
        self.F.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (self.n_particles, 1, 1))
        )

        self._scene_offset = scene_offset
        self._scene_max_extent = max_extent
        self._scene_scale = scene_scale

        # covariance in grid space
        w2g = (scene_scale / (max_extent * 2.0)) if max_extent > 0 else 1.0

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

        s = np.maximum(cloud.scales.astype(np.float32) * w2g * kernel_scale, 1.0 * self.dx)
        inv_s2 = 1.0 / s**2
        inv_cov = np.einsum("nij,nj,nkj->nik", R, inv_s2, R).astype(np.float32)

        TWO_PI_32 = float((2.0 * np.pi) ** 1.5)
        V_arr = (TWO_PI_32 * s[:, 0] * s[:, 1] * s[:, 2]).astype(np.float32)
        m_arr = (self.rho * V_arr * cloud.opacities).astype(np.float32)

        self.inv_cov.from_numpy(inv_cov)
        self.V_p.from_numpy(V_arr)
        self.m.from_numpy(m_arr)

    def get_positions_world(self) -> np.ndarray:
        pos = self.x.to_numpy()
        pos -= 0.5
        if self._scene_max_extent > 0:
            pos *= self._scene_max_extent * 2.0 / self._scene_scale
        pos += self._scene_offset
        return pos

    # ─────────────────────────────────────────────────────────────────
    @ti.kernel
    def _reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_mv[I] = ti.Vector.zero(ti.f32, 3)
            self.grid_m[I] = 0.0

    @ti.kernel
    def _p2g(self):
        stencil = ti.static(2 * self.max_r + 1)
        for p in range(self.n_particles):
            xp = self.x[p]
            ic = self.inv_cov[p]
            base = ti.cast(xp * self.inv_dx, ti.i32)

            # Neo-Hookean
            Fp = self.F[p]
            J = ti.max(Fp.determinant(), 0.01)
            b = Fp @ Fp.transpose()
            stress = self.mu_0 * (
                b - ti.Matrix.identity(ti.f32, 3)
            ) + self.lambda_0 * ti.log(J) * ti.Matrix.identity(ti.f32, 3)

            # normalisation Z
            Z = 0.0
            for di, dj, dk in ti.ndrange(stencil, stencil, stencil):
                cell = base + ti.Vector(
                    [di - self.max_r, dj - self.max_r, dk - self.max_r]
                )
                if (
                    0 <= cell[0] < self.grid_res
                    and 0 <= cell[1] < self.grid_res
                    and 0 <= cell[2] < self.grid_res
                ):
                    d = ti.cast(cell, ti.f32) * self.dx - xp
                    maha = d.dot(ic @ d)
                    if maha <= 9.0:
                        Z += ti.exp(-0.5 * maha)
            self.Z_field[p] = Z

            if Z > 1e-12:
                for di, dj, dk in ti.ndrange(stencil, stencil, stencil):
                    cell = base + ti.Vector(
                        [di - self.max_r, dj - self.max_r, dk - self.max_r]
                    )
                    if (
                        0 <= cell[0] < self.grid_res
                        and 0 <= cell[1] < self.grid_res
                        and 0 <= cell[2] < self.grid_res
                    ):
                        d = ti.cast(cell, ti.f32) * self.dx - xp
                        maha = d.dot(ic @ d)
                        if maha <= 9.0:
                            w = ti.exp(-0.5 * maha) / Z
                            self.grid_mv[cell] += w * self.m[p] * self.v[
                                p
                            ] - self.dt * self.V_p[p] * w * stress @ (ic @ d)
                            self.grid_m[cell] += w * self.m[p]

    @ti.kernel
    def _grid_update(self):
        border = 3
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0.0:
                v = self.grid_mv[I] / m
                v += self.dt * self.gravity
                for d in ti.static(range(3)):
                    if I[d] < border and v[d] < 0:
                        v[d] = 0.0
                    if I[d] >= self.grid_res - border and v[d] > 0:
                        v[d] = 0.0
                self.grid_mv[I] = v

    @ti.kernel
    def _g2p(self):
        stencil = ti.static(2 * self.max_r + 1)
        border = 3
        for p in range(self.n_particles):
            xp = self.x[p]
            ic = self.inv_cov[p]
            base = ti.cast(xp * self.inv_dx, ti.i32)
            Z = self.Z_field[p]

            new_v = ti.Vector.zero(ti.f32, 3)
            grad_v = ti.Matrix.zero(ti.f32, 3, 3)

            if Z > 1e-12:
                for di, dj, dk in ti.ndrange(stencil, stencil, stencil):
                    cell = base + ti.Vector(
                        [di - self.max_r, dj - self.max_r, dk - self.max_r]
                    )
                    if (
                        0 <= cell[0] < self.grid_res
                        and 0 <= cell[1] < self.grid_res
                        and 0 <= cell[2] < self.grid_res
                    ):
                        d = ti.cast(cell, ti.f32) * self.dx - xp
                        maha = d.dot(ic @ d)
                        if maha <= 9.0:
                            w = ti.exp(-0.5 * maha) / Z
                            gv = self.grid_mv[cell]
                            new_v += w * gv
                            # ∇v_p = Σ_j w·gv ⊗ (∂w/∂xp direction)
                            # ∂w/∂xp = w·(ic·d)  where d = xj - xp
                            grad_v += w * gv.outer_product(ic @ d)

            self.v[p] = new_v

            # deformation gradient update
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * grad_v) @ self.F[p]

            # clamp singular values to keep F in a stable range
            svd_u, svd_s, svd_v = ti.svd(self.F[p])
            for svd_i in ti.static(range(3)):
                svd_s[svd_i, svd_i] = ti.min(ti.max(svd_s[svd_i, svd_i], 0.1), 10.0)
            self.F[p] = svd_u @ svd_s @ svd_v.transpose()

            self.x[p] += self.dt * new_v

            lo = (border + 1) * self.dx
            hi = 1.0 - (border + 1) * self.dx
            for d in ti.static(range(3)):
                if self.x[p][d] < lo:
                    self.x[p][d] = lo
                    if self.v[p][d] < 0.0:
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
