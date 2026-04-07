"""Preprocess hollow Gaussian shells into volumetric Gaussian fills.

This module is intentionally solver-decoupled. It operates on a Gaussian cloud,
builds a voxel-domain interior mask, bakes boundary appearance, solves a
harmonic extension for SH coefficients, and emits isotropic interior Gaussians.

The implementation is designed as a practical baseline:

1. Load a 3DGS PLY with full SH coefficients.
2. Bake a Gaussian density field onto a voxel grid.
3. Threshold and morphologically close the shell support.
4. Flood-fill exterior empty voxels to recover the enclosed cavity.
5. Bake boundary SH coefficients on shell voxels touching that cavity.
6. Solve Laplace equations inside the cavity.
7. Sample isotropic interior Gaussians from the cavity volume.

The harmonic extension is carried out independently per SH coefficient/channel.
This keeps the appearance pipeline simple: the boundary provides Dirichlet
values, and the interior receives a smooth continuation of the same SH basis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from plyfile import PlyData


SH_C0 = 0.28209479177387814  # 0.5 * sqrt(1 / pi)


@dataclass
class GaussianSHCloud:
    """Gaussian cloud carrying full SH coefficients."""

    positions: np.ndarray     # (N, 3) float32
    opacities: np.ndarray     # (N,) float32
    scales: np.ndarray        # (N, 3) float32
    rotations: np.ndarray     # (N, 4) float32, quaternion (w, x, y, z)
    sh_coeffs: np.ndarray     # (N, K, 3) float32, K SH coeffs per RGB channel

    @property
    def colors(self) -> np.ndarray:
        return np.clip(self.sh_coeffs[:, 0, :] * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class VoxelGrid:
    """Axis-aligned voxel grid used for preprocessing."""

    resolution: tuple[int, int, int]
    origin: np.ndarray         # (3,) world-space min corner
    voxel_size: float          # scalar voxel size, cubic grid

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.resolution

    def voxel_centers_1d(self, axis: int, lo: int, hi: int) -> np.ndarray:
        idx = np.arange(lo, hi, dtype=np.float32)
        return self.origin[axis] + (idx + 0.5) * self.voxel_size

    def indices_to_world(self, indices: np.ndarray) -> np.ndarray:
        return self.origin[None, :] + (indices.astype(np.float32) + 0.5) * self.voxel_size

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        return np.floor((points - self.origin[None, :]) / self.voxel_size).astype(np.int32)


@dataclass
class BoundarySH:
    """Sparse Dirichlet boundary values for SH coefficients."""

    indices: np.ndarray        # (B, 3) int32
    sh_coeffs: np.ndarray      # (B, K, 3) float32


@dataclass
class FillConfig:
    """Configuration for the interior-fill preprocessing pipeline."""

    grid_resolution: int = 128
    support_sigmas: float = 3.0
    density_threshold: float = 0.5
    close_iters: int = 1
    min_fill_depth_voxels: int = 2
    sigma_scale: float = 0.45
    max_sigma_scale: float = 1.25
    fill_opacity: float = 1.0
    harmonic_iters: int = 256
    harmonic_tol: float = 1e-4
    high_order_decay: float = 0.35


@dataclass
class FillResult:
    """Output of the volumetric fill preprocessing."""

    interior_cloud: GaussianSHCloud
    grid: VoxelGrid
    density: np.ndarray          # (nx, ny, nz) float32
    occupancy: np.ndarray        # (nx, ny, nz) bool, thresholded shell support
    shell_mask: np.ndarray       # (nx, ny, nz) bool, post-processed shell mask
    interior_mask: np.ndarray    # (nx, ny, nz) bool, enclosed empty cavity
    boundary_mask: np.ndarray    # (nx, ny, nz) bool, shell voxels adjacent to cavity
    depth_voxels: np.ndarray     # (nx, ny, nz) int32, cavity depth from shell


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3).astype(np.float32)


def _sorted_vertex_props(vertex, prefix: str) -> list[str]:
    names = [name for name in vertex.data.dtype.names if name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[-1]))


def load_ply_with_sh(path: str, opacity_threshold: float = 0.0) -> GaussianSHCloud:
    """Load a standard 3DGS PLY and keep all SH coefficients."""

    plydata = PlyData.read(path)
    vertex = plydata["vertex"]

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    scales = np.exp(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float32)
    )
    rotations = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1
    ).astype(np.float32)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=-1, keepdims=True), 1e-8)

    dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float32)
    rest_names = _sorted_vertex_props(vertex, "f_rest_")
    if rest_names:
        rest = np.stack([vertex[name] for name in rest_names], axis=-1).astype(np.float32)
        if rest.shape[1] % 3 != 0:
            raise ValueError(f"Expected f_rest_* count to be divisible by 3, got {rest.shape[1]}")
        rest = rest.reshape(len(rest), -1, 3)
        sh_coeffs = np.concatenate([dc[:, None, :], rest], axis=1)
    else:
        sh_coeffs = dc[:, None, :]

    if opacity_threshold > 0.0:
        keep = opacities >= opacity_threshold
        positions = positions[keep]
        opacities = opacities[keep]
        scales = scales[keep]
        rotations = rotations[keep]
        sh_coeffs = sh_coeffs[keep]

    return GaussianSHCloud(
        positions=positions,
        opacities=opacities.astype(np.float32),
        scales=scales.astype(np.float32),
        rotations=rotations.astype(np.float32),
        sh_coeffs=sh_coeffs.astype(np.float32),
    )


def infer_cubic_grid(cloud: GaussianSHCloud, resolution: int, support_sigmas: float = 3.0) -> VoxelGrid:
    """Infer a cubic voxel grid covering the cloud and its Gaussian support."""

    sigma_max = cloud.scales.max(axis=1)
    pad = float(np.max(support_sigmas * sigma_max))
    lo = cloud.positions.min(axis=0) - pad
    hi = cloud.positions.max(axis=0) + pad
    extent = float(np.max(hi - lo))
    voxel_size = extent / float(resolution)
    center = 0.5 * (lo + hi)
    origin = center - 0.5 * extent
    return VoxelGrid(
        resolution=(resolution, resolution, resolution),
        origin=origin.astype(np.float32),
        voxel_size=float(voxel_size),
    )


def _clip_bbox(lo: np.ndarray, hi: np.ndarray, shape: tuple[int, int, int]) -> tuple[slice, slice, slice] | None:
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(shape, dtype=np.int32))
    if np.any(lo >= hi):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def bake_density_grid(
    cloud: GaussianSHCloud,
    grid: VoxelGrid,
    support_sigmas: float = 3.0,
) -> np.ndarray:
    """Bake Gaussian occupancy density onto a voxel grid.

    This uses the full anisotropic covariance implied by Gaussian scale and
    rotation. It is intentionally a preprocessing baseline, not a highly tuned
    rasterizer.
    """

    density = np.zeros(grid.shape, dtype=np.float32)
    rotmats = _quat_to_rotmat(cloud.rotations)

    for idx in range(len(cloud)):
        center = cloud.positions[idx]
        sigma = cloud.scales[idx]
        sigma_max = float(np.max(sigma))
        radius = support_sigmas * sigma_max
        bbox_lo = np.floor((center - radius - grid.origin) / grid.voxel_size).astype(np.int32)
        bbox_hi = np.ceil((center + radius - grid.origin) / grid.voxel_size).astype(np.int32) + 1
        bbox = _clip_bbox(bbox_lo, bbox_hi, grid.shape)
        if bbox is None:
            continue

        xs = grid.voxel_centers_1d(0, bbox[0].start, bbox[0].stop)
        ys = grid.voxel_centers_1d(1, bbox[1].start, bbox[1].stop)
        zs = grid.voxel_centers_1d(2, bbox[2].start, bbox[2].stop)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        diff = np.stack([xx - center[0], yy - center[1], zz - center[2]], axis=-1)

        inv_s2 = 1.0 / np.maximum(sigma.astype(np.float64), 1e-8) ** 2
        R = rotmats[idx].astype(np.float64)
        inv_cov = R @ np.diag(inv_s2) @ R.T
        maha = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
        w = np.exp(-0.5 * maha).astype(np.float32)
        density[bbox] += cloud.opacities[idx] * w

    return density


def dilate6(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def erode6(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :, :] &= mask[:-1, :, :]
    out[:-1, :, :] &= mask[1:, :, :]
    out[:, 1:, :] &= mask[:, :-1, :]
    out[:, :-1, :] &= mask[:, 1:, :]
    out[:, :, 1:] &= mask[:, :, :-1]
    out[:, :, :-1] &= mask[:, :, 1:]
    return out


def close6(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(iters):
        out = dilate6(out)
    for _ in range(iters):
        out = erode6(out)
    return out


def flood_fill_outside(shell_mask: np.ndarray) -> np.ndarray:
    """Mark empty voxels connected to the grid boundary."""

    shape = shell_mask.shape
    outside = np.zeros(shape, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()

    def enqueue_if_empty(i: int, j: int, k: int) -> None:
        if shell_mask[i, j, k] or outside[i, j, k]:
            return
        outside[i, j, k] = True
        q.append((i, j, k))

    nx, ny, nz = shape
    for i in range(nx):
        for j in range(ny):
            enqueue_if_empty(i, j, 0)
            enqueue_if_empty(i, j, nz - 1)
    for i in range(nx):
        for k in range(nz):
            enqueue_if_empty(i, 0, k)
            enqueue_if_empty(i, ny - 1, k)
    for j in range(ny):
        for k in range(nz):
            enqueue_if_empty(0, j, k)
            enqueue_if_empty(nx - 1, j, k)

    nbrs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in nbrs:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            if shell_mask[ni, nj, nk] or outside[ni, nj, nk]:
                continue
            outside[ni, nj, nk] = True
            q.append((ni, nj, nk))

    return outside


def extract_enclosed_void(shell_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return enclosed empty voxels and the shell voxels touching that cavity."""

    outside = flood_fill_outside(shell_mask)
    interior_void = (~shell_mask) & (~outside)
    cavity_boundary = shell_mask & dilate6(interior_void)
    return interior_void, cavity_boundary


def bake_boundary_sh(
    cloud: GaussianSHCloud,
    grid: VoxelGrid,
    boundary_mask: np.ndarray,
    support_sigmas: float = 3.0,
) -> BoundarySH:
    """Bake boundary SH values by weighted splatting of source Gaussians."""

    boundary_indices = np.argwhere(boundary_mask)
    if len(boundary_indices) == 0:
        return BoundarySH(
            indices=np.zeros((0, 3), dtype=np.int32),
            sh_coeffs=np.zeros((0, cloud.sh_coeffs.shape[1], 3), dtype=np.float32),
        )

    boundary_id = -np.ones(grid.shape, dtype=np.int32)
    boundary_id[boundary_mask] = np.arange(len(boundary_indices), dtype=np.int32)
    coeff_sum = np.zeros((len(boundary_indices), cloud.sh_coeffs.shape[1], 3), dtype=np.float32)
    weight_sum = np.zeros(len(boundary_indices), dtype=np.float32)
    rotmats = _quat_to_rotmat(cloud.rotations)

    for idx in range(len(cloud)):
        center = cloud.positions[idx]
        sigma = cloud.scales[idx]
        sigma_max = float(np.max(sigma))
        radius = support_sigmas * sigma_max
        bbox_lo = np.floor((center - radius - grid.origin) / grid.voxel_size).astype(np.int32)
        bbox_hi = np.ceil((center + radius - grid.origin) / grid.voxel_size).astype(np.int32) + 1
        bbox = _clip_bbox(bbox_lo, bbox_hi, grid.shape)
        if bbox is None:
            continue

        local_boundary = boundary_id[bbox]
        hit = local_boundary >= 0
        if not np.any(hit):
            continue

        xs = grid.voxel_centers_1d(0, bbox[0].start, bbox[0].stop)
        ys = grid.voxel_centers_1d(1, bbox[1].start, bbox[1].stop)
        zs = grid.voxel_centers_1d(2, bbox[2].start, bbox[2].stop)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        diff = np.stack([xx - center[0], yy - center[1], zz - center[2]], axis=-1)

        inv_s2 = 1.0 / np.maximum(sigma.astype(np.float64), 1e-8) ** 2
        R = rotmats[idx].astype(np.float64)
        inv_cov = R @ np.diag(inv_s2) @ R.T
        maha = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
        weights = (cloud.opacities[idx] * np.exp(-0.5 * maha)).astype(np.float32)

        ids = local_boundary[hit]
        w = weights[hit]
        coeff_sum[ids] += w[:, None, None] * cloud.sh_coeffs[idx][None, :, :]
        weight_sum[ids] += w

    valid = weight_sum > 1e-12
    coeff_sum[valid] /= weight_sum[valid, None, None]
    coeff_sum[~valid] = 0.0
    return BoundarySH(indices=boundary_indices.astype(np.int32), sh_coeffs=coeff_sum)


def solve_harmonic_scalar(
    domain_mask: np.ndarray,
    boundary_mask: np.ndarray,
    boundary_indices: np.ndarray,
    boundary_values: np.ndarray,
    max_iters: int = 256,
    tol: float = 1e-4,
) -> np.ndarray:
    """Solve a scalar harmonic extension with Dirichlet boundary conditions."""

    field = np.zeros(domain_mask.shape, dtype=np.float32)
    field[boundary_indices[:, 0], boundary_indices[:, 1], boundary_indices[:, 2]] = boundary_values
    interior_mask = domain_mask & ~boundary_mask
    if not np.any(interior_mask):
        return field

    occ = domain_mask.astype(np.float32)
    occ_pad = np.pad(occ, 1)
    deg = (
        occ_pad[:-2, 1:-1, 1:-1]
        + occ_pad[2:, 1:-1, 1:-1]
        + occ_pad[1:-1, :-2, 1:-1]
        + occ_pad[1:-1, 2:, 1:-1]
        + occ_pad[1:-1, 1:-1, :-2]
        + occ_pad[1:-1, 1:-1, 2:]
    )
    deg = np.maximum(deg, 1.0)

    for _ in range(max_iters):
        prev = field
        pad = np.pad(prev, 1)
        nbr_sum = (
            pad[:-2, 1:-1, 1:-1] * occ_pad[:-2, 1:-1, 1:-1]
            + pad[2:, 1:-1, 1:-1] * occ_pad[2:, 1:-1, 1:-1]
            + pad[1:-1, :-2, 1:-1] * occ_pad[1:-1, :-2, 1:-1]
            + pad[1:-1, 2:, 1:-1] * occ_pad[1:-1, 2:, 1:-1]
            + pad[1:-1, 1:-1, :-2] * occ_pad[1:-1, 1:-1, :-2]
            + pad[1:-1, 1:-1, 2:] * occ_pad[1:-1, 1:-1, 2:]
        )
        updated = prev.copy()
        updated[interior_mask] = nbr_sum[interior_mask] / deg[interior_mask]
        updated[boundary_indices[:, 0], boundary_indices[:, 1], boundary_indices[:, 2]] = boundary_values
        err = float(np.max(np.abs(updated[interior_mask] - prev[interior_mask])))
        field = updated
        if err < tol:
            break

    return field


def depth_from_boundary(mask: np.ndarray) -> np.ndarray:
    """Approximate voxel depth by iterative 6-neighborhood erosion layers."""

    depth = -np.ones(mask.shape, dtype=np.int32)
    active = mask.copy()
    layer = 0
    while np.any(active):
        shell = active & ~erode6(active)
        depth[shell] = layer
        active[shell] = False
        layer += 1
    return np.maximum(depth, 0)


def sample_interior_gaussians(
    grid: VoxelGrid,
    interior_mask: np.ndarray,
    depth_voxels: np.ndarray,
    sampled_sh: np.ndarray,
    config: FillConfig,
) -> GaussianSHCloud:
    """Convert enclosed interior voxels into isotropic interior Gaussians."""

    candidate_mask = interior_mask & (depth_voxels >= config.min_fill_depth_voxels)
    indices = np.argwhere(candidate_mask)
    if len(indices) == 0:
        return GaussianSHCloud(
            positions=np.zeros((0, 3), dtype=np.float32),
            opacities=np.zeros((0,), dtype=np.float32),
            scales=np.zeros((0, 3), dtype=np.float32),
            rotations=np.zeros((0, 4), dtype=np.float32),
            sh_coeffs=np.zeros((0, sampled_sh.shape[1], 3), dtype=np.float32),
        )

    positions = grid.indices_to_world(indices).astype(np.float32)
    depth_scale = np.minimum(
        config.max_sigma_scale,
        np.maximum(config.sigma_scale, config.sigma_scale * depth_voxels[candidate_mask]),
    )
    sigma = (depth_scale * grid.voxel_size).astype(np.float32)
    scales = np.repeat(sigma[:, None], 3, axis=1)
    rotations = np.zeros((len(indices), 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full(len(indices), config.fill_opacity, dtype=np.float32)
    return GaussianSHCloud(
        positions=positions,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        sh_coeffs=sampled_sh.astype(np.float32),
    )


def harmonic_extend_sh_to_points(
    domain_mask: np.ndarray,
    boundary_mask: np.ndarray,
    boundary: BoundarySH,
    sample_indices: np.ndarray,
    sample_depth_voxels: np.ndarray,
    config: FillConfig,
) -> np.ndarray:
    """Solve harmonic SH fields and sample them at interior voxel locations."""

    if len(sample_indices) == 0:
        return np.zeros((0, boundary.sh_coeffs.shape[1], 3), dtype=np.float32)

    n_coeffs = boundary.sh_coeffs.shape[1]
    samples = np.zeros((len(sample_indices), n_coeffs, 3), dtype=np.float32)

    for coeff_idx in range(n_coeffs):
        decay = np.exp(-config.high_order_decay * coeff_idx * sample_depth_voxels).astype(np.float32)
        if coeff_idx == 0:
            decay.fill(1.0)
        for rgb_idx in range(3):
            scalar_field = solve_harmonic_scalar(
                domain_mask=domain_mask,
                boundary_mask=boundary_mask,
                boundary_indices=boundary.indices,
                boundary_values=boundary.sh_coeffs[:, coeff_idx, rgb_idx],
                max_iters=config.harmonic_iters,
                tol=config.harmonic_tol,
            )
            sampled = scalar_field[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]
            samples[:, coeff_idx, rgb_idx] = sampled * decay

    return samples


def build_filled_interior(
    cloud: GaussianSHCloud,
    config: FillConfig,
) -> FillResult:
    """Run the full preprocessing pipeline and return interior Gaussians."""

    grid = infer_cubic_grid(cloud, resolution=config.grid_resolution, support_sigmas=config.support_sigmas)
    density = bake_density_grid(cloud, grid, support_sigmas=config.support_sigmas)
    occupancy = density >= config.density_threshold
    shell_mask = occupancy.copy()
    if config.close_iters > 0:
        shell_mask = close6(shell_mask, iters=config.close_iters)
    interior_mask, boundary_mask = extract_enclosed_void(shell_mask)
    boundary = bake_boundary_sh(cloud, grid, boundary_mask, support_sigmas=config.support_sigmas)
    depth = depth_from_boundary(interior_mask)

    sample_mask = interior_mask & (depth >= config.min_fill_depth_voxels)
    sample_indices = np.argwhere(sample_mask)
    sampled_sh = harmonic_extend_sh_to_points(
        domain_mask=interior_mask | boundary_mask,
        boundary_mask=boundary_mask,
        boundary=boundary,
        sample_indices=sample_indices,
        sample_depth_voxels=depth[sample_mask].astype(np.float32),
        config=config,
    )
    interior_cloud = sample_interior_gaussians(
        grid=grid,
        interior_mask=interior_mask,
        depth_voxels=depth,
        sampled_sh=sampled_sh,
        config=config,
    )
    return FillResult(
        interior_cloud=interior_cloud,
        grid=grid,
        density=density,
        occupancy=occupancy,
        shell_mask=shell_mask,
        interior_mask=interior_mask,
        boundary_mask=boundary_mask,
        depth_voxels=depth,
    )


__all__ = [
    "BoundarySH",
    "FillConfig",
    "FillResult",
    "GaussianSHCloud",
    "VoxelGrid",
    "bake_boundary_sh",
    "bake_density_grid",
    "build_filled_interior",
    "extract_enclosed_void",
    "harmonic_extend_sh_to_points",
    "infer_cubic_grid",
    "load_ply_with_sh",
]
