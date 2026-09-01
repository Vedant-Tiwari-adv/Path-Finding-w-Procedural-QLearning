"""Video capture for evaluation episodes (optional, never required for training).

Frames are captured from the PyBullet renderer and written with ``imageio``.
Both are optional: if ``imageio`` is missing, :func:`write_video` falls back to
writing a numbered PNG sequence and tells the user how to assemble it, so a
missing codec never costs you the recording.

Recording defaults to a headless ``DIRECT`` client, which means clips can be
produced while other work is running without a window stealing focus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from drone_rl.utils.logging import get_logger

__all__ = ["VideoRecorder", "write_video", "imageio_available"]

_LOG = get_logger(__name__)


def imageio_available() -> bool:
    """True when ``imageio`` can be imported."""
    try:
        import imageio.v2  # noqa: F401
    except ImportError:
        return False
    return True


def write_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: int = 30,
    quality: int = 8,
) -> Path:
    """Write RGB frames to MP4 (or GIF, chosen by the file extension).

    Falls back to a PNG sequence when ``imageio`` or an MP4 encoder is
    unavailable; the returned path then points at the containing directory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")

    if not imageio_available():
        return _write_png_sequence(frames, output_path)

    import imageio.v2 as imageio

    try:
        if output_path.suffix.lower() == ".gif":
            imageio.mimsave(output_path, list(frames), duration=1.0 / max(fps, 1))
        else:
            imageio.mimwrite(
                output_path, list(frames), fps=fps, quality=quality, macro_block_size=None
            )
    except Exception as exc:  # pragma: no cover - depends on codec availability
        _LOG.warning("Could not encode %s (%s); writing a PNG sequence instead.",
                     output_path.name, exc)
        return _write_png_sequence(frames, output_path)

    _LOG.info("Wrote %s (%d frames, %d fps)", output_path, len(frames), fps)
    return output_path


def _write_png_sequence(frames: Sequence[np.ndarray], output_path: Path) -> Path:
    """Save frames as ``frame_00000.png`` under a directory next to the target."""
    directory = output_path.with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        for index, frame in enumerate(frames):
            imageio.imwrite(directory / f"frame_{index:05d}.png", frame)
    except ImportError:
        from PIL import Image  # matplotlib pulls in Pillow

        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(directory / f"frame_{index:05d}.png")

    _LOG.warning(
        "Saved %d PNG frames to %s. Assemble with:\n"
        "  ffmpeg -framerate 30 -i %s/frame_%%05d.png -pix_fmt yuv420p out.mp4",
        len(frames),
        directory,
        directory,
    )
    return directory


@dataclass
class VideoRecorder:
    """Accumulates frames from a renderer and writes them on :meth:`save`.

    Parameters
    ----------
    renderer:
        A :class:`~drone_rl.environment.renderer.PyBulletRenderer`.
    every_n_steps:
        Capture one frame every N environment steps. The simulation runs at
        ``1 / dt`` steps per second (about 6.7 Hz by default), so capturing
        every step and playing back at 30 fps produces a natural ~4.5x speedup.
    max_frames:
        Hard cap, so a runaway episode cannot exhaust memory.
    """

    renderer: Any
    every_n_steps: int = 1
    max_frames: int = 3_000
    width: int | None = None
    height: int | None = None
    overlay: Any = None
    frames: list[np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = []
        self._step = 0

    def capture(
        self,
        force: bool = False,
        hud_lines: Sequence[str] = (),
        trajectory: np.ndarray | None = None,
        drone_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        start_position: np.ndarray | None = None,
        status_colour: tuple[int, int, int] | None = None,
    ) -> None:
        """Grab one frame if this step is due, compositing the overlay if set.

        PyBullet debug items (text, trail, rays) are not rasterised into
        ``getCameraImage`` output, so anything that must appear in the video is
        drawn here instead - see :mod:`drone_rl.evaluation.overlay`.
        """
        self._step += 1
        due = force or (self._step % max(1, self.every_n_steps) == 0)
        if not due or len(self.frames) >= self.max_frames:
            return

        frame = self.renderer.capture_frame(self.width, self.height)
        if self.overlay is not None:
            view_matrix, projection_matrix = self.renderer.camera_matrices(
                self.width, self.height
            )
            frame = self.overlay.draw(
                frame,
                lines=hud_lines,
                trajectory=trajectory,
                drone_position=drone_position,
                target_position=target_position,
                start_position=start_position,
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
                status_colour=status_colour,
            )
        self.frames.append(frame)

    def hold(self, count: int) -> None:
        """Repeat the last frame ``count`` times (a pause on a key moment)."""
        if self.frames and len(self.frames) + count <= self.max_frames:
            self.frames.extend([self.frames[-1]] * int(count))

    def save(self, output_path: str | Path, fps: int = 30) -> Path | None:
        """Write the accumulated frames, or return ``None`` if there are none."""
        if not self.frames:
            _LOG.warning("No frames captured; nothing written for %s", output_path)
            return None
        return write_video(self.frames, output_path, fps=fps)

    def reset(self) -> None:
        """Drop captured frames and restart the step counter."""
        self.frames = []
        self._step = 0
