"""Save rendered frames as PNG files and optionally compile to MP4."""

import os
from pathlib import Path
import numpy as np
from PIL import Image


class VideoExporter:
    def __init__(self, out_dir: str = "frames", fps: int = 30):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._frame_idx = 0

    def write_frame(self, img: np.ndarray):
        """img: (H, W, 3|4) uint8"""
        path = self.out_dir / f"frame_{self._frame_idx:05d}.png"
        Image.fromarray(img).save(str(path))
        self._frame_idx += 1

    def compile_video(self, output_path: str = "output.mp4", bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> str:
        """Compile saved PNG frames into an MP4 using imageio-ffmpeg."""
        import imageio
        # Build sorted frame list
        frames = sorted(self.out_dir.glob("frame_*.png"))
        if not frames:
            raise RuntimeError("No frames found to compile.")
        writer = imageio.get_writer(output_path, fps=self.fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=1)
        bg = (np.clip(np.asarray(bg_rgb, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.float32)
        for f in frames:
            arr = np.array(Image.open(f))
            if arr.ndim == 3 and arr.shape[2] == 4:
                alpha = arr[..., 3:4].astype(np.float32) / 255.0
                rgb = arr[..., :3].astype(np.float32)
                arr = np.round(rgb * alpha + bg[None, None, :] * (1.0 - alpha)).astype(np.uint8)
            writer.append_data(arr)
        writer.close()
        print(f"Video saved to {output_path}  ({len(frames)} frames @ {self.fps} fps)")
        return output_path
