"""Draw HUD text and the flown trajectory directly onto captured frames.

Why this exists
---------------
PyBullet's ``addUserDebugText`` / ``addUserDebugLine`` items are drawn by the
GUI's own render loop into the on-screen window. They are **not** part of the
scene that ``getCameraImage`` rasterises, so a recorded clip made from captured
frames contains the city and the drone but none of the overlay - no episode
info, no trajectory ribbon, no sensor rays. The live window shows them; the
video does not.

Rather than screen-scraping the window, the overlay is composited onto the
frame here: text with Pillow, and world-space geometry projected through the
same view/projection matrices used to render the frame, so the drawn trajectory
lands exactly on the flown path.

Projection
----------
PyBullet returns the view and projection matrices as flat 16-element tuples in
**column-major** (OpenGL) order. For a world point ``p``::

    clip = P @ V @ [p, 1]
    ndc  = clip.xyz / clip.w                 (perspective divide)
    x_px = (ndc.x * 0.5 + 0.5) * width
    y_px = (1 - (ndc.y * 0.5 + 0.5)) * height   (image y grows downward)

Points with ``clip.w <= 0`` are behind the camera and are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["project_points", "FrameOverlay"]

# Palette, matching the renderer's scene colours.
_HUD_TEXT = (255, 255, 255)
_HUD_PANEL = (16, 20, 26)
_TRAIL = (60, 220, 255)
_TARGET = (60, 220, 100)
_START = (240, 150, 40)
_RETICLE = (255, 235, 90)


def _matrix_from_flat(flat: Sequence[float]) -> np.ndarray:
    """Reshape a PyBullet 16-element column-major matrix into 4x4 row-major."""
    return np.asarray(flat, dtype=np.float64).reshape(4, 4, order="F")


def project_points(
    points: np.ndarray,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixel coordinates.

    Parameters
    ----------
    points:
        ``(N, 3)`` world-space points.
    view_matrix, projection_matrix:
        Flat 16-element matrices as returned by PyBullet.
    width, height:
        Frame size in pixels.

    Returns
    -------
    (pixels, visible)
        ``pixels`` is ``(N, 2)`` float pixel coordinates; ``visible`` is an
        ``(N,)`` bool mask that is ``False`` for points behind the camera.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.size == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    view = _matrix_from_flat(view_matrix)
    projection = _matrix_from_flat(projection_matrix)

    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    clip = homogeneous @ view.T @ projection.T

    w = clip[:, 3]
    visible = w > 1e-6
    safe_w = np.where(visible, w, 1.0)
    ndc = clip[:, :3] / safe_w[:, None]

    pixels = np.empty((points.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
    pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    return pixels, visible


@dataclass
class FrameOverlay:
    """Composites HUD text and projected world geometry onto RGB frames.

    Pillow is used for drawing; it arrives with matplotlib, so no extra
    dependency is introduced. If it is somehow unavailable the overlay degrades
    to returning the frame untouched rather than failing a recording.
    """

    width: int
    height: int
    font_size: int = 15
    panel_alpha: float = 0.55

    def __post_init__(self) -> None:
        self._font = self._load_font()

    def _load_font(self):
        """Load a monospaced font, falling back to Pillow's bitmap default."""
        try:
            from PIL import ImageFont
        except ImportError:  # pragma: no cover - Pillow ships with matplotlib
            return None
        for name in ("consola.ttf", "DejaVuSansMono.ttf", "cour.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, self.font_size)
            except OSError:
                continue
        return ImageFont.load_default()

    def draw(
        self,
        frame: np.ndarray,
        lines: Sequence[str] = (),
        trajectory: np.ndarray | None = None,
        drone_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        start_position: np.ndarray | None = None,
        view_matrix: Sequence[float] | None = None,
        projection_matrix: Sequence[float] | None = None,
        status_colour: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with the overlay composited on.

        World-space elements are drawn only when both matrices are supplied.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover
            return frame

        image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        if view_matrix is not None and projection_matrix is not None:
            self._draw_world(
                draw, trajectory, drone_position, target_position, start_position,
                view_matrix, projection_matrix,
            )
        if lines:
            self._draw_hud(draw, lines, status_colour)
        return np.asarray(image, dtype=np.uint8)

    # ---- world-space elements ------------------------------------------------

    def _draw_world(
        self,
        draw,
        trajectory,
        drone_position,
        target_position,
        start_position,
        view_matrix,
        projection_matrix,
    ) -> None:
        """Draw the flown path plus start/target/drone markers."""
        project = lambda pts: project_points(  # noqa: E731
            pts, view_matrix, projection_matrix, self.width, self.height
        )

        if trajectory is not None and len(trajectory) >= 2:
            pixels, visible = project(np.asarray(trajectory))
            # Draw only segments whose endpoints are both in front of the
            # camera; a segment crossing the camera plane would otherwise
            # smear across the frame.
            for index in range(len(pixels) - 1):
                if not (visible[index] and visible[index + 1]):
                    continue
                draw.line(
                    [tuple(pixels[index]), tuple(pixels[index + 1])],
                    fill=(*_TRAIL, 210),
                    width=3,
                )

        for position, colour, label, radius in (
            (start_position, _START, "START", 7),
            (target_position, _TARGET, "TARGET", 11),
        ):
            if position is None:
                continue
            pixels, visible = project(np.asarray(position).reshape(1, 3))
            if not visible[0]:
                continue
            x, y = pixels[0]
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline=(*colour, 255), width=3,
            )
            draw.text((x + radius + 4, y - 8), label, fill=(*colour, 255), font=self._font)

        if drone_position is not None:
            pixels, visible = project(np.asarray(drone_position).reshape(1, 3))
            if visible[0]:
                # A reticle keeps the drone findable even in a wide shot, where
                # it is only a few pixels across.
                x, y = pixels[0]
                r_outer, r_inner = 20, 9
                draw.ellipse(
                    [x - r_inner, y - r_inner, x + r_inner, y + r_inner],
                    outline=(*_RETICLE, 230), width=2,
                )
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    draw.line(
                        [
                            (x + dx * r_inner, y + dy * r_inner),
                            (x + dx * r_outer, y + dy * r_outer),
                        ],
                        fill=(*_RETICLE, 230), width=2,
                    )

    # ---- HUD -----------------------------------------------------------------

    def _draw_hud(self, draw, lines: Sequence[str], status_colour) -> None:
        """Draw the text block over a translucent panel."""
        pad = 10
        line_height = self.font_size + 6
        text_width = max(
            (draw.textlength(line, font=self._font) for line in lines), default=180
        )
        panel = [
            pad,
            pad,
            pad + text_width + 2 * pad,
            pad + len(lines) * line_height + pad,
        ]
        draw.rectangle(panel, fill=(*_HUD_PANEL, int(255 * self.panel_alpha)))

        for index, line in enumerate(lines):
            # Tint the status line so the outcome reads at a glance.
            colour = _HUD_TEXT
            if status_colour is not None and line.upper().startswith("STATUS"):
                colour = status_colour
            draw.text(
                (pad * 2, pad + index * line_height),
                line,
                fill=(*colour, 255),
                font=self._font,
            )
