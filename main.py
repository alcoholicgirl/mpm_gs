"""
mpm-gs: GPU-accelerated MPM simulator for 3D Gaussian Splatting.
"""

import argparse
import numpy as np
import taichi as ti

from ply_loader import load_ply, GaussianCloud
from mpm_solver import MPMSolver
from renderer import Camera, GaussianRenderer
from video_export import VideoExporter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ply",      type=str, required=True)
    p.add_argument("--frames",   type=int,   default=200)
    p.add_argument("--substeps", type=int,   default=20)
    p.add_argument("--dt",       type=float, default=2e-4)
    p.add_argument("--grid_res", type=int,   default=128)
    p.add_argument("--max_radius", type=int, default=0,
                   help="Gaussian support radius cap in grid cells; 0 means fully adaptive support")
    p.add_argument("--width",    type=int,   default=800)
    p.add_argument("--height",   type=int,   default=600)
    p.add_argument("--fovx_deg", type=float, default=60.0)
    p.add_argument("--bg_rgba",  type=float, nargs=4, default=(1.0, 1.0, 1.0, 1.0),
                   metavar=("R", "G", "B", "A"),
                   help="Background color and alpha in 0..1")
    p.add_argument("--object_scale", type=float, default=0.8,
                   help="Relative size of the object inside the normalized simulation box")
    p.add_argument("--out_dir",  type=str,   default="frames")
    p.add_argument("--video",    type=str,   default=None)
    p.add_argument("--fps",      type=int,   default=60)
    p.add_argument("--youngs",   type=float, default=1e5)
    p.add_argument("--poisson",  type=float, default=0.3)
    p.add_argument("--c2_ratio",   type=float, default=0.0,
                   help="Mooney-Rivlin C2/(C1+C2) ratio (0=Neo-Hookean, 0.5=equal split)")
    p.add_argument("--deform_render", action="store_true",
                   help="Apply deformation gradient F to Gaussian covariances during rendering")
    p.add_argument("--gpu",      action="store_true", help="Use Metal/CUDA backend")
    p.add_argument("--no_sim",        action="store_true", help="Render only, skip MPM")
    p.add_argument("--opacity_thresh", type=float, default=0.0,
                   help="Discard Gaussians with opacity below this value (0–1)")
    p.add_argument("--kernel_scale",  type=float, default=1.0,
                   help="Sim kernel size relative to render kernel (0–1, smaller → less phantom collision)")
    p.add_argument("--floor_friction", type=float, default=0.0,
                   help="Coulomb-style friction coefficient applied on the bottom floor (z-min boundary)")
    p.add_argument("--boundary_thickness", type=float, default=4.0 / 64.0,
                   help="World-space thickness of the box collision boundary in normalized simulation space")
    return p.parse_args()


def make_scene_camera(cloud: GaussianCloud, width: int, height: int,
                      fovx_deg: float) -> Camera:
    """Fixed camera for Z-up scenes: sit back along -Y, Z is up."""
    pos = cloud.positions
    centre = pos.mean(axis=0).astype(np.float64)
    extent = float(np.percentile(np.abs(pos - centre), 95))  # robust extent
    # Target the bottom of the scene — that's where things settle under -Z gravity
    z_min = float(pos[:, 2].min())
    target = np.array([centre[0], centre[1], z_min + extent * 0.3])
    eye = target + np.array([0.0, -extent * 5.0, extent * 1.5])
    return Camera.look_at(
        eye=eye, target=target,
        up=np.array([0.0, 0.0, 1.0]),
        width=width, height=height,
        fovx=np.radians(fovx_deg),
    )


def main():
    args = parse_args()

    if args.gpu:
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)

    print(f"Loading {args.ply} ...")
    cloud = load_ply(args.ply, opacity_threshold=args.opacity_thresh)
    print(f"  {len(cloud)} Gaussians" +
          (f"  (opacity ≥ {args.opacity_thresh})" if args.opacity_thresh > 0 else ""))
    bg = np.asarray(args.bg_rgba, dtype=np.float32)
    cam = make_scene_camera(cloud, args.width, args.height, args.fovx_deg)
    renderer = GaussianRenderer(args.width, args.height, max_gaussians=len(cloud))

    solver = None
    if not args.no_sim:
        solver = MPMSolver(
            n_particles=len(cloud),
            grid_res=args.grid_res,
            dt=args.dt,
            youngs_modulus=args.youngs,
            poisson_ratio=args.poisson,
            max_radius=args.max_radius,
            c2_ratio=args.c2_ratio,
            floor_friction=args.floor_friction,
            boundary_thickness=args.boundary_thickness,
        )
        solver.init_from_cloud(cloud, scene_scale=args.object_scale, kernel_scale=args.kernel_scale)

    exporter = VideoExporter(out_dir=args.out_dir, fps=args.fps)

    print(f"Rendering {args.frames} frames ...")
    for frame in range(args.frames):
        if solver is not None:
            solver.step(n_substeps=args.substeps)
            cloud.positions[:] = solver.get_positions_world()

        M_world = solver.get_deformed_M_world() if (solver is not None and args.deform_render) else None
        img = renderer.render(cloud, cam, bg=bg, M_world=M_world)
        exporter.write_frame(img)

        if frame % 10 == 0:
            print(f"  Frame {frame}/{args.frames}")

    print("Done.")

    if args.video:
        exporter.compile_video(args.video, bg_rgb=tuple(bg[:3]))


if __name__ == "__main__":
    main()
