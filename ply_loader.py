"""Load 3DGS PLY files into numpy arrays."""

from dataclasses import dataclass
import numpy as np
from plyfile import PlyData


@dataclass
class GaussianCloud:
    positions: np.ndarray    # (N, 3) float32
    opacities: np.ndarray    # (N,)   float32, after sigmoid
    colors: np.ndarray       # (N, 3) float32, RGB in [0,1] from DC SH
    scales: np.ndarray       # (N, 3) float32, actual scale (after exp)
    rotations: np.ndarray    # (N, 4) float32, quaternion (w, x, y, z), normalised

    def __len__(self):
        return len(self.positions)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sh_dc_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    """Convert DC SH coefficients to RGB. SH DC = color / (0.5 * sqrt(1/pi)) - 0.5"""
    SH_C0 = 0.28209479177387814  # 0.5 * sqrt(1/pi)
    return np.clip(sh_dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)


def load_ply(path: str, opacity_threshold: float = 0.0) -> GaussianCloud:
    """Load a standard 3DGS PLY file."""
    plydata = PlyData.read(path)
    v = plydata["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    opacities = _sigmoid(np.array(v["opacity"], dtype=np.float32))
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    colors = _sh_dc_to_rgb(sh_dc)
    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    )
    rotations = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1
    ).astype(np.float32)
    norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
    rotations = rotations / np.maximum(norms, 1e-8)

    if opacity_threshold > 0.0:
        keep = opacities >= opacity_threshold
        positions = positions[keep]
        opacities = opacities[keep]
        colors    = colors[keep]
        scales    = scales[keep]
        rotations = rotations[keep]

    return GaussianCloud(
        positions=positions,
        opacities=opacities,
        colors=colors,
        scales=scales,
        rotations=rotations,
    )
