"""Interactive keyboard camera control for the PyBullet viewer.

PyBullet's built-in viewer only offers mouse control, which is awkward during a
live demo when one hand is on the keyboard. This module adds a small camera rig
driven by keys, plus automatic tracking modes so the drone can be followed
without any input at all.

Key choice
----------
Only arrows, modifiers and function keys are bound. PyBullet's viewer already
claims most letter keys for its own toggles (``w`` wireframe, ``s`` shadows,
``g`` panels, ``a`` AABBs, ``i`` pause, ...), so binding letters here would
fight the viewer and surprise the user mid-demo.

Modes
-----
``FREE``
    Camera stays where you put it. Arrows fly the focus point around.
``FOLLOW``
    Focus point tracks the drone; you still control angle and zoom.
``ORBIT``
    Focus point tracks the drone and the camera circles it automatically -
    the hands-off "show the drone" mode.
``CINEMATIC``
    Slow orbit around the city centre, for establishing shots of the layout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["CameraMode", "CameraConfig", "CameraController", "CAMERA_HELP_LINES"]


class CameraMode(str, Enum):
    """How the camera decides where to look."""

    FREE = "free"
    FOLLOW = "follow"
    ORBIT = "orbit"
    CINEMATIC = "cinematic"

    @property
    def tracks_drone(self) -> bool:
        """True when the focus point should follow the drone."""
        return self in (CameraMode.FOLLOW, CameraMode.ORBIT)


#: Shown in the GUI overlay and printed by the demo scripts.
CAMERA_HELP_LINES: tuple[str, ...] = (
    "Arrows: pan  |  Shift/Ctrl: up/down  |  PgUp/PgDn: zoom",
    "Alt+Arrows: orbit angle  |  F1 free  F2 follow  F3 orbit  F4 cinematic  F5 reset",
)


@dataclass
class CameraConfig:
    """Tuning for the interactive camera rig."""

    distance: float = 34.0
    yaw: float = 45.0
    pitch: float = -38.0
    target: tuple[float, float, float] = (0.0, 0.0, 4.0)
    mode: CameraMode = CameraMode.FREE

    move_speed: float = 22.0  # metres/second of focus-point travel
    zoom_speed: float = 26.0  # metres/second of distance change
    angle_speed: float = 70.0  # degrees/second for Alt+arrow rotation
    orbit_speed: float = 18.0  # degrees/second in ORBIT mode
    cinematic_speed: float = 8.0  # degrees/second in CINEMATIC mode
    follow_smoothing: float = 0.86  # 0 = snap to drone, ->1 = heavy lag

    #: Rate (1/s) at which input velocity eases toward its commanded value.
    #: Key state is binary, so integrating it directly makes the camera start and
    #: stop dead - the jerk you feel is the missing acceleration, not the frame
    #: rate. Easing velocity rather than position gives the rig weight while
    #: staying fully responsive. Higher is snappier; ~10-16 feels right.
    move_damping: float = 13.0
    angle_damping: float = 16.0

    #: Frame rate the smoothing constants above are quoted at. Smoothing is
    #: re-derived per frame from the real elapsed time, so behaviour is identical
    #: whether the viewer runs at 15 or 144 fps.
    reference_fps: float = 60.0

    min_distance: float = 3.0
    max_distance: float = 140.0
    min_pitch: float = -89.0
    max_pitch: float = 12.0
    min_height: float = 0.5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "CameraConfig":
        """Build from a ``render:`` config block.

        Accepts the ``camera_*`` names used in ``environment.yaml`` as well as
        the bare field names.
        """
        if not mapping:
            return cls()
        aliases = {
            "camera_distance": "distance",
            "camera_yaw": "yaw",
            "camera_pitch": "pitch",
            "camera_target": "target",
            "camera_mode": "mode",
        }
        known = {f for f in cls.__dataclass_fields__}
        values: dict[str, Any] = {}
        for key, value in mapping.items():
            field_name = aliases.get(key, key)
            # ``None`` means "not specified" (e.g. camera_mode left unset in
            # YAML), so it must not overwrite the dataclass default.
            if field_name in known and value is not None:
                values[field_name] = value

        # follow_drone is the legacy boolean switch; honour it when no explicit
        # mode was given, so old configs keep working.
        if "mode" not in values and mapping.get("follow_drone"):
            values["mode"] = CameraMode.FOLLOW
        if "mode" in values and not isinstance(values["mode"], CameraMode):
            values["mode"] = CameraMode(str(values["mode"]).lower())
        if "target" in values:
            values["target"] = tuple(float(v) for v in values["target"])
        return cls(**values)


class CameraController:
    """Reads keyboard state each frame and drives the debug visualiser camera.

    In ``DIRECT`` mode PyBullet reports no keyboard events, so this degrades to
    the automatic modes only and never errors.

    Parameters
    ----------
    pybullet_module:
        The imported ``pybullet`` module (injected so this file never imports it
        at module scope).
    client_id:
        PyBullet physics client to drive.
    config:
        Initial placement and speeds.
    """

    def __init__(self, pybullet_module, client_id: int, config: CameraConfig | None = None) -> None:
        self._p = pybullet_module
        self.client_id = client_id
        self.config = config or CameraConfig()

        self._initial = CameraConfig(**{f: getattr(self.config, f) for f in
                                        self.config.__dataclass_fields__})
        self.mode = self.config.mode
        self.distance = float(self.config.distance)
        self.yaw = float(self.config.yaw)
        self.pitch = float(self.config.pitch)
        self.target = np.array(self.config.target, dtype=np.float64)

        # Eased input state: velocities, not positions, so the rig accelerates.
        self._pan_velocity = np.zeros(3, dtype=np.float64)
        self._zoom_velocity = 0.0
        self._yaw_velocity = 0.0
        self._pitch_velocity = 0.0

        self._last_time = time.perf_counter()
        self._keys = self._resolve_key_codes()
        self.apply()

    @staticmethod
    def _blend(rate: float, dt: float) -> float:
        """Fraction of the way to move toward a target this frame.

        ``1 - exp(-rate * dt)`` is the frame-rate-independent form of a
        per-frame lerp: doubling the frame rate halves each step's blend, so the
        motion traces the same curve in wall-clock time either way. A raw
        constant alpha does not, which is why fixed-alpha smoothing drifts in
        feel whenever the frame rate moves.
        """
        return float(1.0 - np.exp(-max(rate, 0.0) * max(dt, 0.0)))

    # ---- key handling -------------------------------------------------------

    def _resolve_key_codes(self) -> dict[str, int]:
        """Look up PyBullet's special key codes, tolerating missing names."""
        p = self._p
        names = {
            "left": "B3G_LEFT_ARROW",
            "right": "B3G_RIGHT_ARROW",
            "up": "B3G_UP_ARROW",
            "down": "B3G_DOWN_ARROW",
            "page_up": "B3G_PAGE_UP",
            "page_down": "B3G_PAGE_DOWN",
            "shift": "B3G_SHIFT",
            "control": "B3G_CONTROL",
            "alt": "B3G_ALT",
            "f1": "B3G_F1",
            "f2": "B3G_F2",
            "f3": "B3G_F3",
            "f4": "B3G_F4",
            "f5": "B3G_F5",
        }
        resolved: dict[str, int] = {}
        for label, attribute in names.items():
            code = getattr(p, attribute, None)
            if code is not None:
                resolved[label] = int(code)
        return resolved

    def _is_down(self, events: Mapping[int, int], label: str) -> bool:
        """True while the named key is held."""
        code = self._keys.get(label)
        if code is None:
            return False
        return bool(events.get(code, 0) & self._p.KEY_IS_DOWN)

    def _was_pressed(self, events: Mapping[int, int], label: str) -> bool:
        """True on the frame the named key goes down."""
        code = self._keys.get(label)
        if code is None:
            return False
        return bool(events.get(code, 0) & self._p.KEY_WAS_TRIGGERED)

    # ---- per-frame update ---------------------------------------------------

    def update(
        self,
        drone_position: np.ndarray | None = None,
        events: Mapping[int, int] | None = None,
    ) -> None:
        """Process input and reposition the camera. Call once per rendered frame.

        Parameters
        ----------
        drone_position:
            Point the tracking modes should follow.
        events:
            Pre-read keyboard events. PyBullet's ``getKeyboardEvents`` *drains*
            its buffer, so a caller that also reads keys (to bind its own
            shortcuts) must read once and pass the result in here - otherwise
            whichever of the two calls happens second sees an empty dict and
            those controls go dead. Omit it and the controller reads for itself,
            which is the right thing for a single-consumer caller.
        """
        now = time.perf_counter()
        # Clamp dt so a stalled frame cannot fling the camera across the map.
        dt = float(np.clip(now - self._last_time, 0.0, 0.1))
        self._last_time = now

        if events is None:
            try:
                events = self._p.getKeyboardEvents(physicsClientId=self.client_id)
            except Exception:  # pragma: no cover - DIRECT mode / disconnected
                events = {}

        self._handle_mode_switch(events)
        self._handle_movement(events, dt)
        self._handle_auto_motion(dt)
        if self.mode.tracks_drone and drone_position is not None:
            self._track(drone_position, dt)
        elif self.mode is CameraMode.CINEMATIC:
            self._track(np.array([0.0, 0.0, 6.0]), dt)
        self.apply()

    def _handle_mode_switch(self, events: Mapping[int, int]) -> None:
        """F1-F4 select a mode; F5 restores the initial view."""
        for label, mode in (
            ("f1", CameraMode.FREE),
            ("f2", CameraMode.FOLLOW),
            ("f3", CameraMode.ORBIT),
            ("f4", CameraMode.CINEMATIC),
        ):
            if self._was_pressed(events, label):
                self.mode = mode
        if self._was_pressed(events, "f5"):
            self.reset()

    def _handle_movement(self, events: Mapping[int, int], dt: float) -> None:
        """Arrows pan or (with Alt) orbit; modifiers raise/lower; PgUp/Dn zoom.

        Every axis is driven by easing a *velocity* toward what the keys ask for
        and then integrating it, so motion ramps up when a key goes down and
        coasts to rest when it is released. Releasing a key therefore decelerates
        rather than stopping dead, and the rig keeps gliding for a few frames.
        """
        alt_held = self._is_down(events, "alt")
        move_blend = self._blend(self.config.move_damping, dt)
        angle_blend = self._blend(self.config.angle_damping, dt)

        # --- desired angular velocity (Alt + arrows) ---
        wanted_yaw = 0.0
        wanted_pitch = 0.0
        if alt_held:
            if self._is_down(events, "left"):
                wanted_yaw -= self.config.angle_speed
            if self._is_down(events, "right"):
                wanted_yaw += self.config.angle_speed
            if self._is_down(events, "up"):
                wanted_pitch -= self.config.angle_speed
            if self._is_down(events, "down"):
                wanted_pitch += self.config.angle_speed

        # --- desired pan velocity, in the camera's own frame ---
        # Basis vectors come from the current yaw, so "up" is always "away from
        # the viewer" no matter which way the rig has been swung around.
        wanted_pan = np.zeros(3, dtype=np.float64)
        if not alt_held:
            yaw_rad = np.radians(self.yaw)
            forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
            right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0])
            if self._is_down(events, "up"):
                wanted_pan += forward
            if self._is_down(events, "down"):
                wanted_pan -= forward
            if self._is_down(events, "right"):
                wanted_pan += right
            if self._is_down(events, "left"):
                wanted_pan -= right
            magnitude = float(np.linalg.norm(wanted_pan))
            if magnitude > 0.0:
                # Normalise so a diagonal is not faster than a straight line.
                wanted_pan = wanted_pan / magnitude * self.config.move_speed
                if self.mode.tracks_drone:
                    self.mode = CameraMode.FREE  # panning means "give me control"

        # Vertical travel works in every mode, including while Alt is held.
        if self._is_down(events, "shift"):
            wanted_pan[2] += self.config.move_speed
        if self._is_down(events, "control"):
            wanted_pan[2] -= self.config.move_speed

        wanted_zoom = 0.0
        if self._is_down(events, "page_up"):
            wanted_zoom -= self.config.zoom_speed
        if self._is_down(events, "page_down"):
            wanted_zoom += self.config.zoom_speed

        self._pan_velocity += (wanted_pan - self._pan_velocity) * move_blend
        self._zoom_velocity += (wanted_zoom - self._zoom_velocity) * move_blend
        self._yaw_velocity += (wanted_yaw - self._yaw_velocity) * angle_blend
        self._pitch_velocity += (wanted_pitch - self._pitch_velocity) * angle_blend

        # Park sub-visible drift instead of easing toward zero forever.
        if not np.any(wanted_pan) and float(np.linalg.norm(self._pan_velocity)) < 1e-3:
            self._pan_velocity[:] = 0.0

        if np.any(self._pan_velocity):
            if self.mode.tracks_drone and np.any(self._pan_velocity[:2]):
                self.mode = CameraMode.FREE
            self.target = self.target + self._pan_velocity * dt
        self.distance += self._zoom_velocity * dt
        self.yaw += self._yaw_velocity * dt
        self.pitch += self._pitch_velocity * dt

        self._clamp()

    def _handle_auto_motion(self, dt: float) -> None:
        """Advance the automatic yaw sweep for the orbiting modes."""
        if self.mode is CameraMode.ORBIT:
            self.yaw += self.config.orbit_speed * dt
        elif self.mode is CameraMode.CINEMATIC:
            self.yaw += self.config.cinematic_speed * dt

    def _track(self, position: np.ndarray, dt: float) -> None:
        """Ease the focus point toward ``position``.

        Exponential smoothing keeps the camera from jittering on every small
        drone movement while still keeping it centred. ``follow_smoothing`` is
        the fraction of the gap left after one frame *at* ``reference_fps``; the
        per-frame blend is re-derived from the real ``dt`` so a viewer running at
        144 fps follows with the same lag as one running at 20, instead of
        becoming glued to the drone as the frame rate rises.
        """
        alpha = float(np.clip(self.config.follow_smoothing, 1e-4, 0.9999))
        rate = -np.log(alpha) * max(self.config.reference_fps, 1.0)
        blend = self._blend(rate, dt)
        self.target = self.target + (np.asarray(position, dtype=np.float64) - self.target) * blend

    def _clamp(self) -> None:
        """Keep distance, pitch and height inside usable ranges."""
        self.distance = float(np.clip(self.distance, self.config.min_distance,
                                      self.config.max_distance))
        self.pitch = float(np.clip(self.pitch, self.config.min_pitch, self.config.max_pitch))
        self.target[2] = max(self.config.min_height, float(self.target[2]))
        self.yaw = float(self.yaw % 360.0)

    # ---- output -------------------------------------------------------------

    def apply(self) -> None:
        """Push the current pose to the PyBullet visualiser."""
        try:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=self.distance,
                cameraYaw=self.yaw,
                cameraPitch=self.pitch,
                cameraTargetPosition=[float(v) for v in self.target],
                physicsClientId=self.client_id,
            )
        except Exception:  # pragma: no cover - disconnected client
            pass

    def reset(self) -> None:
        """Restore the initial placement and mode."""
        self.mode = self._initial.mode
        self.distance = float(self._initial.distance)
        self.yaw = float(self._initial.yaw)
        self.pitch = float(self._initial.pitch)
        self.target = np.array(self._initial.target, dtype=np.float64)

    def set_mode(self, mode: CameraMode | str) -> None:
        """Switch mode programmatically (used by the recording scripts)."""
        self.mode = mode if isinstance(mode, CameraMode) else CameraMode(str(mode).lower())

    @property
    def status_line(self) -> str:
        """One-line camera state for the HUD."""
        return (
            f"CAM {self.mode.value.upper():<9} "
            f"dist {self.distance:5.1f}  yaw {self.yaw:5.1f}  pitch {self.pitch:5.1f}"
        )

    def view_matrix(self) -> Sequence[float]:
        """View matrix for the current pose (works in DIRECT mode too)."""
        return self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[float(v) for v in self.target],
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=self.client_id,
        )
