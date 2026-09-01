"""Phase 3 - range sensors and the observation encoder.

Two separable pieces, kept apart on purpose so that Phase-14 successor
algorithms can swap either one without touching the environment:

* :class:`RangeSensorArray` - simulated depth/lidar rays fixed in the drone's
  body frame. Computed analytically against the city's AABBs (see
  ``geometry.raycast_boxes``); a PyBullet ``rayTestBatch`` produces the same
  numbers, which ``tests/test_geometry.py`` verifies.
* :class:`ObservationEncoder` - packs drone state, goal information and sensor
  readings into a single normalised float32 vector.

Replacing the encoder with an RGB/depth/latent front-end means implementing the
same three members used by the environment: ``space``, ``encode`` and
``labels``. No other code needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from drone_rl.environment.drone import Drone
from drone_rl.environment.geometry import norm3, raycast_boxes

__all__ = ["SensorConfig", "RangeSensorArray", "ObservationEncoder", "DEFAULT_RAY_DIRECTIONS"]


#: Body-frame unit directions for the default 7-ray array.
#: ``+x`` forward, ``+y`` left, ``+z`` up.
_SQRT_HALF = float(np.sqrt(0.5))
DEFAULT_RAY_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (1.0, 0.0, 0.0),
    "front_left": (_SQRT_HALF, _SQRT_HALF, 0.0),
    "front_right": (_SQRT_HALF, -_SQRT_HALF, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class SensorConfig:
    """Range-sensor parameters."""

    max_range: float = 12.0
    directions: tuple[str, ...] = tuple(DEFAULT_RAY_DIRECTIONS)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "SensorConfig":
        """Build from the ``sensors:`` YAML block."""
        if not mapping:
            return cls()
        directions = mapping.get("directions")
        if directions:
            unknown = [d for d in directions if d not in DEFAULT_RAY_DIRECTIONS]
            if unknown:
                raise KeyError(
                    f"Unknown sensor directions {unknown}; "
                    f"available: {sorted(DEFAULT_RAY_DIRECTIONS)}"
                )
            directions = tuple(directions)
        else:
            directions = tuple(DEFAULT_RAY_DIRECTIONS)
        return cls(max_range=float(mapping.get("max_range", 12.0)), directions=directions)


class RangeSensorArray:
    """Body-fixed ray sensors returning distance to the nearest obstacle.

    The rays rotate with the drone's yaw, so ``front`` always means "ahead of
    the nose". Readings are clipped to ``max_range``; a ray that hits nothing
    reports exactly ``max_range``.
    """

    def __init__(self, config: SensorConfig | None = None) -> None:
        self.config = config or SensorConfig()
        self._body_dirs = np.array(
            [DEFAULT_RAY_DIRECTIONS[name] for name in self.config.directions],
            dtype=np.float64,
        )

    @property
    def num_rays(self) -> int:
        return len(self.config.directions)

    @property
    def names(self) -> tuple[str, ...]:
        """Ray names in reading order."""
        return tuple(self.config.directions)

    def world_directions(self, yaw: float) -> np.ndarray:
        """Rotate the body-frame ray directions into world space by ``yaw``.

        A yaw rotation is orthonormal, so the result stays unit length - which
        lets :func:`raycast_boxes` skip re-normalising.
        """
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        body = self._body_dirs
        rotated = np.empty_like(body)
        rotated[:, 0] = body[:, 0] * cos_a - body[:, 1] * sin_a
        rotated[:, 1] = body[:, 0] * sin_a + body[:, 1] * cos_a
        rotated[:, 2] = body[:, 2]
        return rotated

    def read(
        self,
        position: np.ndarray,
        yaw: float,
        box_mins: np.ndarray,
        box_maxs: np.ndarray,
        ceiling_z: float | None = None,
    ) -> np.ndarray:
        """Return one distance per ray, in metres, clipped to ``max_range``."""
        directions = self.world_directions(yaw)
        origin = np.asarray(position, dtype=np.float64).reshape(1, 3)
        return raycast_boxes(
            origins=origin,
            directions=directions,
            box_mins=box_mins,
            box_maxs=box_maxs,
            max_range=self.config.max_range,
            ground_z=0.0,
            ceiling_z=ceiling_z,
            directions_are_unit=True,
        )

    def ray_endpoints(self, position: np.ndarray, yaw: float, distances: np.ndarray) -> np.ndarray:
        """World-space endpoints of each ray, for debug visualisation."""
        directions = self.world_directions(yaw)
        origin = np.asarray(position, dtype=np.float64).reshape(1, 3)
        return origin + directions * np.asarray(distances).reshape(-1, 1)


class ObservationEncoder:
    """Packs the world state into a normalised ``float32`` observation vector.

    Layout (all components roughly in ``[-1, 1]``):

    ==========================  ======  ====================================
    Component                   Size    Normalisation
    ==========================  ======  ====================================
    drone position x, y, z      3       by city half-extent / max altitude
    target relative x, y, z     3       in the drone's *body* frame, by
                                        the city diagonal
    distance to target          1       by the city diagonal
    velocity x, y, z            3       body frame, by max speed
    heading (cos, sin)          2       unit circle, avoids the wrap
                                        discontinuity of a raw angle
    altitude                    1       by max altitude
    battery                     1       fraction remaining
    range sensors               R       by sensor max range
    ==========================  ======  ====================================

    The relative-goal and velocity terms are expressed in the *body* frame so
    the policy is invariant to absolute world orientation - the same manoeuvre
    generalises regardless of which way the drone happens to be facing.

    Nothing here exposes the layout of the city beyond what the rays can see, so
    the agent is never handed a shortest path or privileged map information.
    """

    def __init__(
        self,
        sensors: RangeSensorArray,
        city_config,
        drone_config,
    ) -> None:
        self.sensors = sensors
        self._city_diagonal = float(
            np.linalg.norm([city_config.width, city_config.depth, city_config.max_altitude])
        )
        self._half_width = float(city_config.width * 0.5)
        self._half_depth = float(city_config.depth * 0.5)
        self._max_altitude = float(city_config.max_altitude)
        self._max_speed = float(drone_config.max_speed)
        self._sensor_range = float(sensors.config.max_range)

    @property
    def size(self) -> int:
        """Length of the observation vector."""
        return 14 + self.sensors.num_rays

    @property
    def space(self) -> spaces.Box:
        """Gymnasium observation space (bounded box)."""
        return spaces.Box(low=-2.0, high=2.0, shape=(self.size,), dtype=np.float32)

    @property
    def labels(self) -> list[str]:
        """Per-element names, used for debug printing and figures."""
        base = [
            "pos_x",
            "pos_y",
            "pos_z",
            "rel_target_forward",
            "rel_target_left",
            "rel_target_up",
            "distance_to_target",
            "vel_forward",
            "vel_left",
            "vel_up",
            "heading_cos",
            "heading_sin",
            "altitude",
            "battery",
        ]
        return base + [f"ray_{name}" for name in self.sensors.names]

    def _world_to_body(self, vector: np.ndarray, yaw: float) -> np.ndarray:
        """Rotate a world vector into the drone's body frame (inverse yaw)."""
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        x, y, z = vector
        return np.array(
            [x * cos_a + y * sin_a, -x * sin_a + y * cos_a, z],
            dtype=np.float64,
        )

    def encode(
        self,
        drone: Drone,
        target: np.ndarray,
        ray_distances: np.ndarray,
    ) -> np.ndarray:
        """Build the observation vector for the current state."""
        state = drone.state
        position = state.position
        to_target = target - position
        distance = norm3(to_target)

        rel_body = self._world_to_body(to_target, state.yaw) / max(self._city_diagonal, 1e-6)
        vel_body = self._world_to_body(state.velocity, state.yaw) / max(self._max_speed, 1e-6)

        obs = np.empty(self.size, dtype=np.float32)
        obs[0] = position[0] / max(self._half_width, 1e-6)
        obs[1] = position[1] / max(self._half_depth, 1e-6)
        obs[2] = position[2] / max(self._max_altitude, 1e-6)
        obs[3:6] = rel_body
        obs[6] = distance / max(self._city_diagonal, 1e-6)
        obs[7:10] = vel_body
        obs[10] = np.cos(state.yaw)
        obs[11] = np.sin(state.yaw)
        obs[12] = position[2] / max(self._max_altitude, 1e-6)
        obs[13] = drone.battery_fraction
        obs[14:] = np.asarray(ray_distances, dtype=np.float32) / max(self._sensor_range, 1e-6)

        return np.clip(obs, -2.0, 2.0, out=obs)
