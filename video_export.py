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
        """img: (H, W, 3) uint8"""
        path = self.out_dir / f"frame_{self._frame_idx:05d}.png"
        Image.fromarray(img).save(str(path))
        self._frame_idx += 1

    def compile_video(self, output_path: str = "output.mp4") -> str:
        """Compile saved PNG frames into an MP4 using imageio-ffmpeg."""
        import imageio
        pattern = str(self.out_dir / "frame_%05d.png")
        # Build sorted frame list
        frames = sorted(self.out_dir.glob("frame_*.png"))
        if not frames:
            raise RuntimeError("No frames found to compile.")
        writer = imageio.get_writer(output_path, fps=self.fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=1)
        for f in frames:
            writer.append_data(np.array(Image.open(f)))
        writer.close()
        print(f"Video saved to {output_path}  ({len(frames)} frames @ {self.fps} fps)")
        return output_path
