"""Phase 2 - simplified kinematic drone model.

Deliberately *not* an aerodynamic simulation. The drone is a point mass with
first-order velocity tracking: an action requests a body-frame velocity, and the
actual velocity relaxes toward it with a configurable smoothing factor. This
gives visible momentum (the drone overshoots and cannot stop instantly, so
learning to brake before a wall matters) without the instability and tuning
burden of rotor-level dynamics.

State: position, velocity, yaw, battery. Energy is drawn per step from a
constant hover cost plus an action-dependent manoeuvre cost, so loitering
depletes the battery and the agent must trade time against energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from drone_rl.environment.spaces import ActionSpec

__all__ = ["DroneConfig", "DroneState", "Drone"]


@dataclass(frozen=True)
class DroneConfig:
    """Kinematic and energetic parameters of the drone."""

    max_speed: float = 3.0
    max_yaw_rate: float = 1.2  # rad/s
    velocity_smoothing: float = 0.55  # 0 = instant response, ->1 = heavy inertia
    collision_radius: float = 0.45
    battery_capacity: float = 100.0
    hover_energy_per_second: float = 0.55
    manoeuvre_energy_per_second: float = 0.9
    spawn_altitude_min: float = 1.5
    spawn_altitude_max: float = 6.0
    min_altitude: float = 0.35
    dt: float = 0.15  # seconds of simulated time per environment step

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DroneConfig":
        """Build from a YAML block, ignoring unrelated keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class DroneState:
    """Mutable drone state in world coordinates."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    yaw: float = 0.0
    battery: float = 100.0
    elapsed_time: float = 0.0

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            yaw=float(self.yaw),
            battery=float(self.battery),
            elapsed_time=float(self.elapsed_time),
        )


class Drone:
    """Kinematic drone with first-order velocity tracking and a battery.

    The drone owns *only* its own state. Collision handling and world bounds are
    the environment's responsibility: :meth:`propose_step` is a pure function
    returning where the drone *would* go, and the environment decides whether to
    accept that move, so a rejected (colliding) move never corrupts state.
    """

    def __init__(self, config: DroneConfig | None = None) -> None:
        self.config = config or DroneConfig()
        self.state = DroneState(battery=self.config.battery_capacity)
        # Validated once here rather than clamped on every step.
        self._velocity_alpha = float(np.clip(self.config.velocity_smoothing, 0.0, 0.99))

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, position: np.ndarray, yaw: float = 0.0) -> DroneState:
        """Place the drone at ``position`` with zero velocity and a full battery."""
        self.state = DroneState(
            position=np.asarray(position, dtype=np.float64).reshape(3).copy(),
            velocity=np.zeros(3, dtype=np.float64),
            yaw=float(yaw),
            battery=float(self.config.battery_capacity),
            elapsed_time=0.0,
        )
        return self.state

    # ---- kinematics --------------------------------------------------------

    def body_to_world(self, body_vector: np.ndarray, yaw: float | None = None) -> np.ndarray:
        """Rotate a body-frame vector into world coordinates about the z axis."""
        angle = self.state.yaw if yaw is None else yaw
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        forward, left, up = body_vector
        # Body +x is heading; body +y is 90 degrees counter-clockwise from it.
        return np.array(
            [
                forward * cos_a - left * sin_a,
                forward * sin_a + left * cos_a,
                up,
            ],
            dtype=np.float64,
        )

    def propose_step(self, action: ActionSpec) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute the post-action state *without* mutating the drone.

        Returns ``(position, velocity, yaw, energy_used)``. The environment calls
        this, tests the proposed position for collisions, and then either
        :meth:`commit` s the result or applies its own collision response.
        """
        cfg = self.config
        yaw = float(self.state.yaw + action.yaw_rate * cfg.max_yaw_rate * cfg.dt)
        yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))  # wrap to [-pi, pi]

        target_velocity = self.body_to_world(action.body_velocity * cfg.max_speed, yaw=yaw)
        # First-order lag: v <- alpha * v + (1 - alpha) * v_target.
        alpha = self._velocity_alpha
        velocity = alpha * self.state.velocity + (1.0 - alpha) * target_velocity

        position = self.state.position + velocity * cfg.dt
        energy_used = self.energy_for_action(action)
        return position, velocity, yaw, energy_used

    def energy_for_action(self, action: ActionSpec) -> float:
        """Battery units consumed by one step of ``action``.

        A constant hover draw plus a manoeuvre draw scaled by the action's
        ``energy_scale``. Ascending costs the most, descending the least.
        """
        cfg = self.config
        hover = cfg.hover_energy_per_second * cfg.dt
        manoeuvre = cfg.manoeuvre_energy_per_second * action.energy_scale * cfg.dt
        return float(hover + manoeuvre)

    def commit(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float,
        energy_used: float,
    ) -> DroneState:
        """Write an accepted transition into the drone state."""
        self.state.position = np.asarray(position, dtype=np.float64).reshape(3).copy()
        self.state.velocity = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
        self.state.yaw = float(yaw)
        self.state.battery = float(max(0.0, self.state.battery - energy_used))
        self.state.elapsed_time += self.config.dt
        return self.state

    # ---- derived quantities -------------------------------------------------

    @property
    def speed(self) -> float:
        """Current speed magnitude in m/s."""
        return float(np.linalg.norm(self.state.velocity))

    @property
    def altitude(self) -> float:
        """Height above the ground plane."""
        return float(self.state.position[2])

    @property
    def battery_fraction(self) -> float:
        """Remaining battery in ``[0, 1]``."""
        return float(self.state.battery / self.config.battery_capacity)

    @property
    def heading_vector(self) -> np.ndarray:
        """Unit vector pointing along the drone's yaw."""
        return np.array([np.cos(self.state.yaw), np.sin(self.state.yaw), 0.0], dtype=np.float64)

    def orientation_quaternion(self) -> tuple[float, float, float, float]:
        """Yaw-only orientation as an ``(x, y, z, w)`` quaternion for PyBullet."""
        half = self.state.yaw * 0.5
        return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))
