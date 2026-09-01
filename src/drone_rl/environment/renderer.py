"""PyBullet visualisation layer (Phase 10 / Phase 1 visual output).

The renderer *displays* the world; it does not decide it. All dynamics,
collision and sensing happen in the analytic core (``geometry.py``), and this
module mirrors that state into a PyBullet scene each step. The consequence is
that a demo replays exactly the trajectory the agent would fly headless - the
GUI cannot silently disagree with training physics.

PyBullet is imported lazily so that training, testing and plotting all work on a
machine where PyBullet is not installed. Creating a :class:`PyBulletRenderer`
without PyBullet raises a clear, actionable error.

Connection modes
----------------
``gui``
    Opens an interactive OpenGL window. Used by ``demo.py`` /
    ``compare_learning.py``.
``direct``
    Headless. Used for offscreen frame capture (video recording) and for the
    ray-parity test against the analytic core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from drone_rl.environment.camera import CAMERA_HELP_LINES, CameraConfig, CameraController
from drone_rl.environment.city import CityLayout
from drone_rl.utils.logging import get_logger

__all__ = ["PyBulletRenderer", "RenderConfig", "pybullet_available"]

_LOG = get_logger(__name__)

# Palette (RGBA) - flat colours, deliberately not photorealistic.
_COLOR_BUILDING = (0.62, 0.64, 0.70, 1.0)
_COLOR_BUILDING_TALL = (0.45, 0.48, 0.58, 1.0)
_COLOR_GROUND = (0.22, 0.24, 0.27, 1.0)
_COLOR_DRONE_BODY = (0.10, 0.55, 0.95, 1.0)
_COLOR_DRONE_ARM = (0.15, 0.17, 0.20, 1.0)
_COLOR_ROTOR = (0.95, 0.75, 0.10, 1.0)
_COLOR_TARGET = (0.10, 0.85, 0.35, 0.55)
_COLOR_START = (0.95, 0.55, 0.10, 0.6)
_COLOR_TRAIL = (0.20, 0.90, 1.00)
_COLOR_RAY_CLEAR = (0.25, 0.85, 0.35)
_COLOR_RAY_HIT = (0.95, 0.25, 0.20)


def pybullet_available() -> bool:
    """Return ``True`` when the ``pybullet`` module can be imported."""
    try:
        import pybullet  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class RenderConfig:
    """Visual options for the PyBullet scene."""

    mode: str = "gui"
    show_rays: bool = False
    show_trail: bool = True
    trail_length: int = 400
    show_target_marker: bool = True
    show_start_marker: bool = True
    camera_distance: float = 34.0
    camera_yaw: float = 45.0
    camera_pitch: float = -38.0
    follow_drone: bool = False
    #: Initial camera mode: free | follow | orbit | cinematic. Overrides
    #: ``follow_drone`` when set. See :mod:`drone_rl.environment.camera`.
    camera_mode: str | None = None
    interactive_camera: bool = True
    show_camera_help: bool = True
    realtime_factor: float = 1.0
    width: int = 1280
    height: int = 720
    camera_fov: float = 60.0


class PyBulletRenderer:
    """Mirrors environment state into a PyBullet scene.

    Typical lifecycle::

        renderer = PyBulletRenderer(RenderConfig(mode="gui"))
        renderer.build_city(layout)          # once per episode/city
        renderer.set_markers(start, target)
        renderer.update(position, yaw, ...)  # every step
        renderer.close()
    """

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()
        self._p = self._import_pybullet()
        self.client_id = self._connect()
        self._building_ids: list[int] = []
        self._ground_id: int | None = None
        self._drone_id: int | None = None
        self._target_id: int | None = None
        self._start_id: int | None = None
        self._ray_ids: list[int] = []
        self._trail_ids: list[int] = []
        self._text_ids: list[int] = []
        self._last_trail_point: np.ndarray | None = None
        self._configure_visualiser()

    # ---- setup -------------------------------------------------------------

    @staticmethod
    def _import_pybullet():
        """Import PyBullet with an actionable error when it is missing."""
        try:
            import pybullet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PyBullet is required for visualisation but is not installed.\n"
                "On Windows it must be compiled from source, which needs the MSVC "
                "C++ build tools:\n"
                "  winget install Microsoft.VisualStudio.2022.BuildTools "
                '--override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools"\n'
                "  pip install pybullet\n"
                "On Linux/macOS a wheel is available: pip install pybullet\n"
                "Training and evaluation do not require PyBullet - only the 3D demo does."
            ) from exc
        return pybullet

    def _connect(self) -> int:
        """Open a PyBullet connection in the configured mode."""
        p = self._p
        mode = self.config.mode.lower()
        if mode == "gui":
            options = f"--width={self.config.width} --height={self.config.height}"
            client = p.connect(p.GUI, options=options)
            if client < 0:  # pragma: no cover - depends on display availability
                _LOG.warning("Could not open a PyBullet GUI window; falling back to DIRECT.")
                client = p.connect(p.DIRECT)
                self.config.mode = "direct"
        elif mode == "direct":
            client = p.connect(p.DIRECT)
        else:
            raise ValueError(f"Unknown render mode {self.config.mode!r}; use 'gui' or 'direct'")
        if client < 0:  # pragma: no cover
            raise RuntimeError("Failed to connect to PyBullet")
        return client

    def _configure_visualiser(self) -> None:
        """Set gravity-free simulation, a clean HUD, and the camera rig."""
        p = self._p
        p.setGravity(0, 0, 0, physicsClientId=self.client_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id
        )  # hide side panels
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id
        )

        camera_config = CameraConfig.from_mapping(
            {
                "camera_distance": self.config.camera_distance,
                "camera_yaw": self.config.camera_yaw,
                "camera_pitch": self.config.camera_pitch,
                "camera_mode": self.config.camera_mode,
                "follow_drone": self.config.follow_drone,
            }
        )
        self.camera = CameraController(p, self.client_id, camera_config)

    # ---- scene construction ------------------------------------------------

    def build_city(self, layout: CityLayout) -> None:
        """(Re)build the static scene for ``layout``.

        Bodies are created as static (mass 0) collision+visual shapes. The
        physics engine never steps, so these exist purely to be drawn and, in
        the parity test, to be ray-cast against.
        """
        p = self._p
        self.clear_dynamic()
        for body_id in self._building_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self._building_ids.clear()
        if self._ground_id is not None:
            p.removeBody(self._ground_id, physicsClientId=self.client_id)
            self._ground_id = None

        cfg = layout.config
        # Ground slab, slightly larger than the city for visual framing.
        ground_half = [cfg.width * 0.6, cfg.depth * 0.6, 0.1]
        ground_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=ground_half,
            rgbaColor=_COLOR_GROUND,
            physicsClientId=self.client_id,
        )
        ground_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=ground_half, physicsClientId=self.client_id
        )
        self._ground_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=ground_collision,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0.0, 0.0, -0.1],
            physicsClientId=self.client_id,
        )

        tall_threshold = cfg.building_min_height + 0.6 * (
            cfg.building_max_height - cfg.building_min_height
        )
        for building in layout.buildings:
            color = _COLOR_BUILDING_TALL if building.height >= tall_threshold else _COLOR_BUILDING
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=list(building.half_extents),
                rgbaColor=color,
                physicsClientId=self.client_id,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=list(building.half_extents),
                physicsClientId=self.client_id,
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=list(building.center),
                physicsClientId=self.client_id,
            )
            self._building_ids.append(body_id)

        if self._drone_id is None:
            self._drone_id = self._create_drone_body()

    def _create_drone_body(self) -> int:
        """Build a quadrotor from primitives: body, four arms, four rotors.

        Uses a multi-body with fixed links so the whole craft moves as one rigid
        object when its base pose is reset.
        """
        p = self._p
        body_half = [0.18, 0.18, 0.07]
        arm_half = [0.22, 0.03, 0.02]
        rotor_radius, rotor_height = 0.16, 0.025

        base_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=body_half,
            rgbaColor=_COLOR_DRONE_BODY,
            physicsClientId=self.client_id,
        )
        base_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=body_half, physicsClientId=self.client_id
        )

        link_visuals: list[int] = []
        link_collisions: list[int] = []
        link_positions: list[list[float]] = []
        link_orientations: list[list[float]] = []

        arm_offset = 0.30
        for index, angle_deg in enumerate((45.0, 135.0, 225.0, 315.0)):
            angle = math.radians(angle_deg)
            dx, dy = arm_offset * math.cos(angle), arm_offset * math.sin(angle)
            quat = p.getQuaternionFromEuler([0.0, 0.0, angle])

            link_visuals.append(
                p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=arm_half,
                    rgbaColor=_COLOR_DRONE_ARM,
                    physicsClientId=self.client_id,
                )
            )
            link_collisions.append(-1)
            link_positions.append([dx * 0.6, dy * 0.6, 0.0])
            link_orientations.append(list(quat))

            link_visuals.append(
                p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=rotor_radius,
                    length=rotor_height,
                    rgbaColor=_COLOR_ROTOR,
                    physicsClientId=self.client_id,
                )
            )
            link_collisions.append(-1)
            link_positions.append([dx, dy, 0.05])
            link_orientations.append([0.0, 0.0, 0.0, 1.0])

        num_links = len(link_visuals)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0.0, 0.0, 2.0],
            linkMasses=[0.0] * num_links,
            linkCollisionShapeIndices=link_collisions,
            linkVisualShapeIndices=link_visuals,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0]] * num_links,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * num_links,
            linkParentIndices=[0] * num_links,
            linkJointTypes=[p.JOINT_FIXED] * num_links,
            linkJointAxis=[[0, 0, 1]] * num_links,
            physicsClientId=self.client_id,
        )

    def set_markers(self, start: np.ndarray, target: np.ndarray, target_radius: float) -> None:
        """Draw translucent spheres at the start and target positions."""
        p = self._p
        for attr in ("_target_id", "_start_id"):
            body_id = getattr(self, attr)
            if body_id is not None:
                p.removeBody(body_id, physicsClientId=self.client_id)
                setattr(self, attr, None)

        if self.config.show_target_marker:
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(target_radius),
                rgbaColor=_COLOR_TARGET,
                physicsClientId=self.client_id,
            )
            self._target_id = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[float(v) for v in target],
                physicsClientId=self.client_id,
            )
        if self.config.show_start_marker:
            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.45,
                rgbaColor=_COLOR_START,
                physicsClientId=self.client_id,
            )
            self._start_id = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[float(v) for v in start],
                physicsClientId=self.client_id,
            )
        self._last_trail_point = np.asarray(start, dtype=np.float64).copy()

    # ---- per-step update ---------------------------------------------------

    def update(
        self,
        position: np.ndarray,
        yaw: float,
        ray_origins: np.ndarray | None = None,
        ray_endpoints: np.ndarray | None = None,
        ray_hit_mask: Sequence[bool] | None = None,
    ) -> None:
        """Move the drone and refresh debug overlays."""
        p = self._p
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if self._drone_id is not None:
            half = yaw * 0.5
            quat = [0.0, 0.0, math.sin(half), math.cos(half)]
            p.resetBasePositionAndOrientation(
                self._drone_id,
                [float(v) for v in position],
                quat,
                physicsClientId=self.client_id,
            )

        if self.config.show_trail:
            self._extend_trail(position)
        if self.config.show_rays and ray_endpoints is not None:
            self._draw_rays(position, ray_endpoints, ray_hit_mask)
        if self.config.interactive_camera:
            # Reads the keyboard and applies whichever tracking mode is active.
            self.camera.update(position)

    def _extend_trail(self, position: np.ndarray) -> None:
        """Append one segment to the flown trajectory ribbon."""
        p = self._p
        if self._last_trail_point is None:
            self._last_trail_point = position.copy()
            return
        if float(np.linalg.norm(position - self._last_trail_point)) < 1e-3:
            return
        line_id = p.addUserDebugLine(
            [float(v) for v in self._last_trail_point],
            [float(v) for v in position],
            lineColorRGB=list(_COLOR_TRAIL),
            lineWidth=2.5,
            physicsClientId=self.client_id,
        )
        self._trail_ids.append(line_id)
        self._last_trail_point = position.copy()
        while len(self._trail_ids) > self.config.trail_length:
            p.removeUserDebugItem(self._trail_ids.pop(0), physicsClientId=self.client_id)

    def _draw_rays(
        self,
        origin: np.ndarray,
        endpoints: np.ndarray,
        hit_mask: Sequence[bool] | None,
    ) -> None:
        """Redraw the sensor rays, red where they hit something."""
        p = self._p
        for item_id in self._ray_ids:
            p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
        self._ray_ids.clear()
        for index, endpoint in enumerate(np.asarray(endpoints, dtype=np.float64)):
            hit = bool(hit_mask[index]) if hit_mask is not None else False
            color = _COLOR_RAY_HIT if hit else _COLOR_RAY_CLEAR
            self._ray_ids.append(
                p.addUserDebugLine(
                    [float(v) for v in origin],
                    [float(v) for v in endpoint],
                    lineColorRGB=list(color),
                    lineWidth=1.4,
                    physicsClientId=self.client_id,
                )
            )

    def set_hud(self, lines: Sequence[str], position: np.ndarray | None = None) -> None:
        """Replace the on-screen text overlay with ``lines``.

        The camera help and status are appended automatically when the
        interactive rig is active, so every GUI script documents its own
        controls without repeating them.
        """
        p = self._p
        lines = list(lines)
        if self.config.interactive_camera and self.config.show_camera_help:
            # A blank entry would be an empty debug-text body; use a space.
            lines = lines + [" ", self.camera.status_line, *CAMERA_HELP_LINES]

        anchor = (
            np.asarray(position, dtype=np.float64)
            if position is not None
            else np.array([0.0, 0.0, 18.0])
        )

        # Update in place when the line count is unchanged. Every debug-item call
        # is a synchronous round-trip to the GUI server, so the remove-then-add
        # form costs two per line per frame - with a dozen lines that dominated
        # the frame budget and visibly throttled the viewer. `replaceItemUniqueId`
        # halves the calls and removes the flicker of text blinking out and back.
        reuse = len(lines) == len(self._text_ids)
        if not reuse:
            for item_id in self._text_ids:
                p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
            self._text_ids.clear()

        for index, line in enumerate(lines):
            location = [
                float(anchor[0]),
                float(anchor[1]),
                float(anchor[2]) + 3.0 - index * 0.9,
            ]
            if reuse:
                p.addUserDebugText(
                    line,
                    location,
                    textColorRGB=[1.0, 1.0, 1.0],
                    textSize=1.25,
                    replaceItemUniqueId=self._text_ids[index],
                    physicsClientId=self.client_id,
                )
            else:
                self._text_ids.append(
                    p.addUserDebugText(
                        line,
                        location,
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.25,
                        physicsClientId=self.client_id,
                    )
                )

    def flash_event(self, position: np.ndarray, color: tuple[float, float, float]) -> None:
        """Draw a short-lived cross at ``position`` to mark a collision/success."""
        p = self._p
        position = np.asarray(position, dtype=np.float64).reshape(3)
        size = 0.9
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = size
            p.addUserDebugLine(
                [float(v) for v in position - offset],
                [float(v) for v in position + offset],
                lineColorRGB=list(color),
                lineWidth=4.0,
                lifeTime=2.5,
                physicsClientId=self.client_id,
            )

    # ---- capture / teardown -------------------------------------------------

    def camera_matrices(
        self, width: int | None = None, height: int | None = None
    ) -> tuple[Sequence[float], Sequence[float]]:
        """Return ``(view_matrix, projection_matrix)`` for the current camera.

        Exposed so overlay code can project world points into the exact frame
        that :meth:`capture_frame` produces.
        """
        width = int(width or self.config.width)
        height = int(height or self.config.height)
        view_matrix = self.camera.view_matrix()
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=width / max(height, 1),
            nearVal=0.1,
            farVal=400.0,
            physicsClientId=self.client_id,
        )
        return view_matrix, projection_matrix

    def capture_frame(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Render an RGB frame as a ``(H, W, 3)`` uint8 array.

        The view and projection matrices are computed from the camera rig
        rather than read back from the debug visualiser, because
        ``getDebugVisualizerCamera`` is only meaningful in GUI mode. Building
        them explicitly means video recording works headless, so clips can be
        produced without opening a window that steals focus.
        """
        p = self._p
        width = int(width or self.config.width)
        height = int(height or self.config.height)

        view_matrix, proj_matrix = self.camera_matrices(width, height)
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
            if self.config.mode == "gui"
            else p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        frame = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
        return frame[:, :, :3]

    def raytest_batch(self, origins: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
        """Run PyBullet's own ray test; used to validate the analytic core."""
        p = self._p
        results = p.rayTestBatch(
            [[float(v) for v in o] for o in np.atleast_2d(origins)],
            [[float(v) for v in e] for e in np.atleast_2d(endpoints)],
            physicsClientId=self.client_id,
        )
        return np.array([r[2] for r in results], dtype=np.float64)  # hit fraction

    def clear_dynamic(self) -> None:
        """Remove trails, rays and HUD text (called between episodes)."""
        p = self._p
        for collection in (self._trail_ids, self._ray_ids, self._text_ids):
            for item_id in collection:
                try:
                    p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
                except Exception:  # pragma: no cover - id may already be gone
                    pass
            collection.clear()
        self._last_trail_point = None

    def close(self) -> None:
        """Disconnect from PyBullet."""
        if self.client_id is not None and self.client_id >= 0:
            try:
                self._p.disconnect(physicsClientId=self.client_id)
            except Exception:  # pragma: no cover - already disconnected
                pass
            self.client_id = -1

    def __enter__(self) -> "PyBulletRenderer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
