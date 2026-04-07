"""Run MPM simulation on a shell Gaussian cloud augmented with volumetric fill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from gaussian_fill import FillConfig, build_filled_interior, load_ply_with_sh
from main import make_scene_camera
from mpm_solver import MPMSolver
from ply_loader import GaussianCloud
from renderer import GaussianRenderer
from video_export import VideoExporter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=str, required=True)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--substeps", type=int, default=20)
    p.add_argument("--dt", type=float, default=2e-4)
    p.add_argument("--grid_res", type=int, default=64)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fovx_deg", type=float, default=60.0)
    p.add_argument("--bg_rgba", type=float, nargs=4, default=(1.0, 1.0, 1.0, 1.0))
    p.add_argument("--object_scale", type=float, default=0.8)
    p.add_argument("--out_dir", type=str, default="frames_fill_sim")
    p.add_argument("--video", type=str, default="fill_sim.mp4")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--youngs", type=float, default=1e5)
    p.add_argument("--poisson", type=float, default=0.3)
    p.add_argument("--c2_ratio", type=float, default=0.0)
    p.add_argument("--floor_friction", type=float, default=0.0)
    p.add_argument("--boundary_thickness", type=float, default=4.0 / 64.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--deform_render", action="store_true")
    p.add_argument("--fill_grid", type=int, default=64)
    p.add_argument("--fill_thresh", type=float, default=0.35)
    p.add_argument("--fill_close", type=int, default=1)
    p.add_argument("--fill_min_depth", type=int, default=2)
    p.add_argument("--fill_iters", type=int, default=64)
    p.add_argument("--fill_sigma_scale", type=float, default=0.45)
    p.add_argument("--fill_max_sigma_scale", type=float, default=1.25)
    p.add_argument("--fill_opacity", type=float, default=1.0)
    p.add_argument("--fill_high_order_decay", type=float, default=0.35)
    p.add_argument("--fill_only", action="store_true",
                   help="Simulate only the generated interior Gaussians")
    return p.parse_args()


def to_render_cloud(cloud_sh) -> GaussianCloud:
    return GaussianCloud(
        positions=cloud_sh.positions.astype(np.float32),
        opacities=cloud_sh.opacities.astype(np.float32),
        colors=cloud_sh.colors.astype(np.float32),
        scales=cloud_sh.scales.astype(np.float32),
        rotations=cloud_sh.rotations.astype(np.float32),
    )


def merge_clouds(shell: GaussianCloud, interior: GaussianCloud, fill_only: bool) -> GaussianCloud:
    if fill_only:
        return interior
    return GaussianCloud(
        positions=np.concatenate([shell.positions, interior.positions], axis=0),
        opacities=np.concatenate([shell.opacities, interior.opacities], axis=0),
        colors=np.concatenate([shell.colors, interior.colors], axis=0),
        scales=np.concatenate([shell.scales, interior.scales], axis=0),
        rotations=np.concatenate([shell.rotations, interior.rotations], axis=0),
    )


def main():
    args = parse_args()

    ti.init(arch=ti.gpu if args.gpu else ti.cpu)

    shell_sh = load_ply_with_sh(args.ply)
    fill_cfg = FillConfig(
        grid_resolution=args.fill_grid,
        density_threshold=args.fill_thresh,
        close_iters=args.fill_close,
        min_fill_depth_voxels=args.fill_min_depth,
        harmonic_iters=args.fill_iters,
        sigma_scale=args.fill_sigma_scale,
        max_sigma_scale=args.fill_max_sigma_scale,
        fill_opacity=args.fill_opacity,
        high_order_decay=args.fill_high_order_decay,
    )
    fill_result = build_filled_interior(shell_sh, fill_cfg)

    shell = to_render_cloud(shell_sh)
    interior = to_render_cloud(fill_result.interior_cloud)
    cloud = merge_clouds(shell, interior, args.fill_only)

    print(f"Loaded shell: {len(shell)} gaussians")
    print(f"Shell voxels (raw/closed): {int(fill_result.occupancy.sum())} / {int(fill_result.shell_mask.sum())}")
    print(f"Cavity boundary voxels: {int(fill_result.boundary_mask.sum())}")
    print(f"Interior cavity voxels: {int(fill_result.interior_mask.sum())}")
    print(f"Generated interior: {len(interior)} gaussians")
    print(f"Simulating total: {len(cloud)} gaussians")

    bg = np.asarray(args.bg_rgba, dtype=np.float32)
    cam = make_scene_camera(shell, args.width, args.height, args.fovx_deg)
    renderer = GaussianRenderer(args.width, args.height, max_gaussians=len(cloud))
    solver = MPMSolver(
        n_particles=len(cloud),
        grid_res=args.grid_res,
        dt=args.dt,
        youngs_modulus=args.youngs,
        poisson_ratio=args.poisson,
        c2_ratio=args.c2_ratio,
        floor_friction=args.floor_friction,
        boundary_thickness=args.boundary_thickness,
    )
    solver.init_from_cloud(cloud, scene_scale=args.object_scale)
    exporter = VideoExporter(out_dir=args.out_dir, fps=args.fps)

    for frame in range(args.frames):
        solver.step(n_substeps=args.substeps)
        cloud.positions[:] = solver.get_positions_world()
        M_world = solver.get_deformed_M_world() if args.deform_render else None
        img = renderer.render(cloud, cam, bg=bg, M_world=M_world)
        exporter.write_frame(img)
        if frame % 10 == 0:
            print(f"  Frame {frame}/{args.frames}")

    if args.video:
        exporter.compile_video(args.video, bg_rgb=tuple(bg[:3]))

    summary_path = Path(args.out_dir) / "fill_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"shell_gaussians={len(shell)}",
                f"shell_voxels_raw={int(fill_result.occupancy.sum())}",
                f"shell_voxels_closed={int(fill_result.shell_mask.sum())}",
                f"boundary_voxels={int(fill_result.boundary_mask.sum())}",
                f"interior_voxels={int(fill_result.interior_mask.sum())}",
                f"interior_gaussians={len(interior)}",
                f"total_gaussians={len(cloud)}",
                f"fill_grid={args.fill_grid}",
                f"fill_threshold={args.fill_thresh}",
                f"fill_min_depth={args.fill_min_depth}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
