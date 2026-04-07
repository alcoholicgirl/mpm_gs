"""
Rasterizer
"""

import numpy as np
import taichi as ti

from argsort import argsort
from ply_loader import GaussianCloud

class Camera:
    def __init__(self, width: int, height: int, fovx: float, c2w: np.ndarray):
        self.width = width
        self.height = height
        self.fovx = fovx
        self.c2w = c2w.astype(np.float64)

    @property
    def fx(self):
        return self.width / (2.0 * np.tan(self.fovx / 2.0))

    @property
    def fy(self):
        return self.fx

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0

    @property
    def w2c(self):
        return np.linalg.inv(self.c2w)

    @staticmethod
    def look_at(
        eye, target, up, width=800, height=600, fovx=np.radians(60)
    ) -> "Camera":
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        z = eye - target
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = x
        c2w[:3, 1] = y
        c2w[:3, 2] = z
        c2w[:3, 3] = eye
        return Camera(width=width, height=height, fovx=fovx, c2w=c2w)


# linalg
def _quat_to_rotmat(q):
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
    ).reshape(-1, 3, 3)


def _project(cloud: GaussianCloud, cam: Camera, M_world: np.ndarray | None = None):
    w2c = cam.w2c.astype(np.float32)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    W, H = cam.width, cam.height

    pos_c = cloud.positions @ R.T + t  # (N,3)
    z = pos_c[:, 2]

    # cull
    mask = z < -0.1
    if not mask.any():
        return None
    pos_c = pos_c[mask]
    z = z[mask]
    op = cloud.opacities[mask]
    col = cloud.colors[mask]
    sc = cloud.scales[mask]
    rot = cloud.rotations[mask]

    # depth sort
    order = argsort((-z).astype(np.float32))
    pos_c = pos_c[order]
    z     = z[order]
    op    = op[order]
    col   = col[order]
    sc    = sc[order]
    rot   = rot[order]

    # project centers
    inv_z = 1.0 / z
    px = (fx * pos_c[:, 0] * inv_z + cx).astype(np.float32)
    py = (fy * pos_c[:, 1] * inv_z + cy).astype(np.float32)

    # jacobian
    if M_world is not None:
        M_c   = M_world[mask][order]
        cov3w = M_c @ M_c.transpose(0, 2, 1)
    else:
        Rg    = _quat_to_rotmat(rot)
        M     = Rg * sc[:, None, :]                          # R @ diag(s), all f32
        cov3w = M @ M.transpose(0, 2, 1)
    cov3c = np.einsum("ij,njk,lk->nil", R, cov3w, R)

    xc, yc = pos_c[:, 0], pos_c[:, 1]
    J = np.zeros((len(pos_c), 2, 3), dtype=np.float32)
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * xc * inv_z * inv_z
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * yc * inv_z * inv_z

    cov2 = np.einsum("nij,njk,nlk->nil", J, cov3c, J)
    cov2[:, 0, 0] += 0.3
    cov2[:, 1, 1] += 0.3  # low-pass

    det = np.maximum(cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2, 1e-8)
    inv_d = 1.0 / det
    ic_a = (cov2[:, 1, 1] * inv_d).astype(np.float32)
    ic_c = (cov2[:, 0, 0] * inv_d).astype(np.float32)
    ic_b = (-2.0 * cov2[:, 0, 1] * inv_d).astype(np.float32)

    trace = cov2[:, 0, 0] + cov2[:, 1, 1]
    sq = np.sqrt(np.maximum(0, trace * trace / 4.0 - det))
    radius = np.ceil(3.0 * np.sqrt(np.maximum(trace / 2.0 + sq, 0))).astype(np.int32)

    # cull
    on_screen = (
        (px + radius >= 0) & (px - radius < W) & (py + radius >= 0) & (py - radius < H)
    )
    px = px[on_screen]
    py = py[on_screen]
    ic_a = ic_a[on_screen]
    ic_b = ic_b[on_screen]
    ic_c = ic_c[on_screen]
    radius = radius[on_screen]
    col = col[on_screen].astype(np.float32)
    op = op[on_screen].astype(np.float32)

    return dict(
        px=px,
        py=py,
        ic_a=ic_a,
        ic_b=ic_b,
        ic_c=ic_c,
        radius=radius,
        colors=col,
        opacities=op,
    )


def _build_tile_lists(px, py, radius, W, H, tile_size=16):
    """
    Returns:
      tile_list  : int32 (M,)   — Gaussian indices sorted by (tile_id, depth_rank)
      tile_off   : int32 (T+1,) — exclusive prefix sums of per-tile counts
    where T = n_tiles_y * n_tiles_x, M = total (tile, Gaussian) pairs.
    Gaussians are already in depth order (index = depth rank).
    """
    ntx = (W + tile_size - 1) // tile_size
    nty = (H + tile_size - 1) // tile_size
    T = ntx * nty
    N = len(px)

    if N == 0:
        return np.empty(0, np.int32), np.zeros(T + 1, np.int32)

    tx_lo = np.clip((px - radius) // tile_size, 0, ntx - 1).astype(np.int32)
    tx_hi = np.clip((px + radius) // tile_size, 0, ntx - 1).astype(np.int32)
    ty_lo = np.clip((py - radius) // tile_size, 0, nty - 1).astype(np.int32)
    ty_hi = np.clip((py + radius) // tile_size, 0, nty - 1).astype(np.int32)

    n_tx = (tx_hi - tx_lo + 1).astype(np.int32)
    n_ty = (ty_hi - ty_lo + 1).astype(np.int32)
    n_tiles_each = n_tx * n_ty  # (N,)

    total = int(n_tiles_each.sum())
    if total == 0:
        return np.empty(0, np.int32), np.zeros(T + 1, np.int32)

    # Expand: g_idx[i] appears n_tiles_each[i] times
    g_idx = np.repeat(np.arange(N, dtype=np.int32), n_tiles_each)  # (M,)

    # Local offset within each Gaussian's tile block: 0, 1, ..., n_tiles-1
    cum = np.empty(N + 1, dtype=np.int64)
    cum[0] = 0
    np.cumsum(n_tiles_each, out=cum[1:])
    local_off = np.arange(total, dtype=np.int32) - np.repeat(
        cum[:-1], n_tiles_each
    ).astype(np.int32)

    nx_rep = np.repeat(n_tx, n_tiles_each)
    local_ty = (local_off // nx_rep).astype(np.int32)
    local_tx = (local_off % nx_rep).astype(np.int32)

    act_tx = np.repeat(tx_lo, n_tiles_each) + local_tx
    act_ty = np.repeat(ty_lo, n_tiles_each) + local_ty
    tile_ids = act_ty * ntx + act_tx  # (M,)

    # Sort by (tile_id, depth_rank=g_idx) — g_idx already encodes depth order
    sort_key = tile_ids.astype(np.int64) * N + g_idx
    order = argsort(sort_key, stable=True)
    tile_list = g_idx[order].astype(np.int32)
    sorted_tids = tile_ids[order]

    # Prefix-sum for per-tile offsets
    tile_counts = np.bincount(sorted_tids, minlength=T).astype(np.int32)
    tile_off = np.empty(T + 1, dtype=np.int32)
    tile_off[0] = 0
    np.cumsum(tile_counts, out=tile_off[1:])

    return tile_list, tile_off


@ti.data_oriented
class GaussianRenderer:
    def __init__(
        self, width: int, height: int, max_gaussians: int, tile_size: int = 16
    ):
        self.W = width
        self.H = height
        self.tile_size = tile_size
        self.ntx = (width + tile_size - 1) // tile_size
        self.nty = (height + tile_size - 1) // tile_size
        self.n_tiles = self.ntx * self.nty
        self.max_g = max_gaussians

        # per-Gaussian projected data (uploaded each frame)
        self.t_px = ti.field(ti.f32, max_gaussians)
        self.t_py = ti.field(ti.f32, max_gaussians)
        self.t_ic_a = ti.field(ti.f32, max_gaussians)
        self.t_ic_b = ti.field(ti.f32, max_gaussians)
        self.t_ic_c = ti.field(ti.f32, max_gaussians)
        self.t_color = ti.Vector.field(3, ti.f32, max_gaussians)
        self.t_opacity = ti.field(ti.f32, max_gaussians)

        # tile lists: use ti.ndarray so shape can vary each frame
        # (no recompilation needed when M changes)
        self.t_tile_off = ti.field(ti.i32, self.n_tiles + 1)

        self.canvas = ti.Vector.field(4, ti.f32, (height, width))

    @ti.kernel
    def _rasterise(
        self,
        tile_list: ti.types.ndarray(ti.i32, ndim=1),
        bg_r: ti.f32,
        bg_g: ti.f32,
        bg_b: ti.f32,
        bg_a: ti.f32,
    ):
        ts = ti.static(self.tile_size)
        for v, u in self.canvas:
            tile_id = (v // ts) * self.ntx + (u // ts)
            start = self.t_tile_off[tile_id]
            end = self.t_tile_off[tile_id + 1]
            T = 1.0
            r = g = b = 0.0  # premultiplied foreground accumulation
            uf = ti.cast(u, ti.f32)
            vf = ti.cast(v, ti.f32)
            for k in range(start, end):
                i = tile_list[k]
                du = uf - self.t_px[i]
                dv = vf - self.t_py[i]
                exp = (
                    self.t_ic_a[i] * du * du
                    + self.t_ic_b[i] * du * dv
                    + self.t_ic_c[i] * dv * dv
                ) * 0.5
                if exp > 4.5:
                    continue
                alpha = ti.min(self.t_opacity[i] * ti.exp(-exp), 0.999)
                w = alpha * T
                col = self.t_color[i]
                r += w * col[0]
                g += w * col[1]
                b += w * col[2]
                T *= 1.0 - alpha
                if T < 1e-4:
                    break
            alpha_out = 1.0 - T * (1.0 - bg_a)
            premul_r = r + T * bg_a * bg_r
            premul_g = g + T * bg_a * bg_g
            premul_b = b + T * bg_a * bg_b
            out_r = 0.0
            out_g = 0.0
            out_b = 0.0
            if alpha_out > 1e-6:
                inv_alpha = 1.0 / alpha_out
                out_r = premul_r * inv_alpha
                out_g = premul_g * inv_alpha
                out_b = premul_b * inv_alpha
            self.canvas[v, u] = ti.Vector([out_r, out_g, out_b, alpha_out])

    def render(
        self,
        cloud: GaussianCloud,
        cam: Camera,
        bg: np.ndarray | None = None,
        M_world: np.ndarray | None = None,
    ) -> np.ndarray:
        if bg is None:
            bg = np.ones(4, dtype=np.float32)
        bg = np.asarray(bg, dtype=np.float32)
        if bg.shape[0] == 3:
            bg = np.concatenate([bg, np.ones(1, dtype=np.float32)])

        proj = _project(cloud, cam, M_world=M_world)
        if proj is None:
            blank = np.ones((cam.height, cam.width, 4), dtype=np.float32)
            blank *= bg[None, None, :]
            return (blank.clip(0, 1) * 255).astype(np.uint8)

        px, py = proj["px"], proj["py"]
        ic_a = proj["ic_a"]
        ic_b = proj["ic_b"]
        ic_c = proj["ic_c"]
        radius = proj["radius"]
        colors = proj["colors"]
        opacities = proj["opacities"]

        # build tile lists (numpy, vectorised)
        tile_list, tile_off = _build_tile_lists(
            px, py, radius, cam.width, cam.height, self.tile_size
        )

        n = len(px)
        assert n <= self.max_g, f"n={n} > max_gaussians={self.max_g}"

        # upload per-Gaussian data — pad to field size (culled n ≤ max_gaussians)
        def _pad(a, fill=0.0):
            pad_w = self.max_g - len(a)
            return np.ascontiguousarray(
                np.pad(a, [(0, pad_w)] + [(0, 0)] * (a.ndim - 1), constant_values=fill)
            )

        self.t_px.from_numpy(_pad(px))
        self.t_py.from_numpy(_pad(py))
        self.t_ic_a.from_numpy(_pad(ic_a))
        self.t_ic_b.from_numpy(_pad(ic_b))
        self.t_ic_c.from_numpy(_pad(ic_c))
        self.t_color.from_numpy(_pad(colors))
        self.t_opacity.from_numpy(_pad(opacities))
        self.t_tile_off.from_numpy(np.ascontiguousarray(tile_off.astype(np.int32)))

        # tile_list passed as ndarray → no pre-allocation needed
        tl = np.ascontiguousarray(tile_list.astype(np.int32))
        self._rasterise(tl, float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]))

        return (self.canvas.to_numpy().clip(0, 1) * 255).astype(np.uint8)
