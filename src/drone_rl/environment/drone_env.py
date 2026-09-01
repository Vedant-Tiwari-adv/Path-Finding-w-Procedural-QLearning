"""Phase 5 - ``DroneCityEnv``, the Gymnasium environment.

Ties together the procedural city (Phase 1), the kinematic drone (Phase 2),
range sensors and the observation encoder (Phase 3), the discrete action set
(Phase 4) and the modular reward (Phase 6).

Termination semantics
---------------------
Gymnasium distinguishes two end conditions, and conflating them silently
corrupts DQN's bootstrap target. This environment is strict about it:

``terminated=True``
    An *environment-intrinsic* end state: target reached, crash, flat battery,
    or leaving the city volume. Value beyond this point is zero by definition,
    so the DQN target must **not** bootstrap.
``truncated=True``
    An externally imposed cutoff: the step limit. The underlying MDP would have
    continued, so the target **must** bootstrap from the final observation.

The environment therefore returns a *genuine* final observation on truncation
(not a zeroed placeholder), and the trainer stores ``terminated`` - never
``terminated or truncated`` - as the replay-buffer done flag.

Reproducibility
---------------
``reset(seed=s)`` seeds an internal ``numpy.random.Generator`` used for every
episode-level random draw (city seed selection, start, target, initial yaw). No
global RNG is consulted, so two environments constructed with the same config
and reset with the same seed produce byte-identical episodes regardless of what
else the process is doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_rl.environment.city import CityConfig, CityLayout, generate_city
from drone_rl.environment.drone import Drone, DroneConfig
from drone_rl.environment.geometry import (
    closest_distance_to_boxes,
    min_sq_distance_to_boxes,
    norm3,
)
from drone_rl.environment.rewards import RewardConfig, RewardFunction
from drone_rl.environment.sensors import ObservationEncoder, RangeSensorArray, SensorConfig
from drone_rl.environment.spaces import ActionSet, build_action_set
from drone_rl.utils.logging import get_logger

__all__ = ["DroneCityEnv", "EnvConfig", "EpisodeOutcome"]

_LOG = get_logger(__name__)


class EpisodeOutcome:
    """String constants for the terminal status of an episode."""

    RUNNING = "RUNNING"
    TARGET_REACHED = "TARGET REACHED"
    COLLISION = "COLLISION"
    BATTERY_EMPTY = "BATTERY EMPTY"
    OUT_OF_BOUNDS = "OUT OF BOUNDS"
    TIMEOUT = "TIMEOUT"


@dataclass
class EnvConfig:
    """Episode-level environment settings (the ``environment:`` YAML block)."""

    max_steps: int = 400
    target_radius: float = 2.0
    min_start_target_distance: float = 18.0
    #: Minimum distance from any building at which a start or target may be
    #: placed. Kept at or above the reward's ``safety_distance`` so a delivery
    #: point is never parked inside the obstacle-proximity penalty zone.
    spawn_clearance: float = 2.5
    target_altitude_min: float = 1.5
    target_altitude_max: float = 8.0
    collision_substeps: int = 4
    seed_pool: str = "train"
    debug: bool = False
    #: Fixed city seed for the whole run; ``None`` samples from the seed pool.
    fixed_city_seed: int | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "EnvConfig":
        if not mapping:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


class DroneCityEnv(gym.Env):
    """Navigate a kinematic drone from a start to a delivery target in a city.

    Parameters
    ----------
    config:
        Full experiment config dict. The relevant blocks are ``city``,
        ``drone``, ``sensors``, ``actions``, ``reward`` and ``environment``.
    seed_pool:
        Which city-seed range to draw from: ``"train"``, ``"validation"`` or
        ``"test"``. Overrides ``environment.seed_pool`` when given.
    render_mode:
        ``None`` (headless, the training default) or ``"human"`` to open a
        PyBullet GUI window.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        seed_pool: str | None = None,
        render_mode: str | None = None,
        render_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config = dict(config or {})
        self.raw_config = config

        self.city_config = CityConfig.from_mapping(config.get("city", {}))
        self.drone_config = DroneConfig.from_mapping(config.get("drone", {}))
        self.sensor_config = SensorConfig.from_mapping(config.get("sensors"))
        self.reward_config = RewardConfig.from_mapping(config.get("reward"))
        self.env_config = EnvConfig.from_mapping(config.get("environment"))
        if seed_pool is not None:
            self.env_config.seed_pool = seed_pool

        self._seed_ranges = self._resolve_seed_ranges(config.get("city", {}))

        self.action_set: ActionSet = build_action_set(config.get("actions"))
        self.sensors = RangeSensorArray(self.sensor_config)
        self.encoder = ObservationEncoder(self.sensors, self.city_config, self.drone_config)
        self.reward_function = RewardFunction(self.reward_config)
        self.drone = Drone(self.drone_config)

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = self.encoder.space

        # Episode state, populated by reset().
        self.city: CityLayout | None = None
        self._box_mins = np.zeros((0, 3), dtype=np.float64)
        self._box_maxs = np.zeros((0, 3), dtype=np.float64)
        self.start_position = np.zeros(3, dtype=np.float64)
        self.target_position = np.zeros(3, dtype=np.float64)
        self.trajectory: list[np.ndarray] = []
        self.status: str = EpisodeOutcome.RUNNING
        self.step_count = 0
        self.episode_reward = 0.0
        self.energy_used_total = 0.0
        self.path_length = 0.0
        self._previous_distance = 0.0
        self._last_ray_distances = np.zeros(self.sensors.num_rays, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._city_cache: dict[int, CityLayout] = {}

        self.render_mode = render_mode
        self._render_config = dict(render_config or {})
        self._renderer = None

    # ---- configuration helpers ---------------------------------------------

    @staticmethod
    def _resolve_seed_ranges(city_config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
        """Read the train/validation/test city-seed pools from config."""
        defaults = {
            "train": (0, 799),
            "validation": (800, 899),
            "test": (900, 999),
        }
        ranges = {}
        for pool, (lo_default, hi_default) in defaults.items():
            lo = int(city_config.get(f"{pool}_seed_start", lo_default))
            hi = int(city_config.get(f"{pool}_seed_end", hi_default))
            if hi < lo:
                raise ValueError(f"Empty {pool} seed range [{lo}, {hi}]")
            ranges[pool] = (lo, hi)
        return ranges

    @property
    def seed_pool(self) -> str:
        """Name of the city-seed pool this environment draws from."""
        return self.env_config.seed_pool

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.action_set.names

    @property
    def observation_labels(self) -> list[str]:
        return self.encoder.labels

    def _sample_city_seed(self) -> int:
        """Draw a city seed from the configured pool (or return the fixed one)."""
        if self.env_config.fixed_city_seed is not None:
            return int(self.env_config.fixed_city_seed)
        pool = self.env_config.seed_pool
        if pool not in self._seed_ranges:
            raise KeyError(f"Unknown seed pool {pool!r}; expected one of {sorted(self._seed_ranges)}")
        lo, hi = self._seed_ranges[pool]
        return int(self._rng.integers(lo, hi + 1))

    def _get_city(self, city_seed: int) -> CityLayout:
        """Return the (cached) layout for ``city_seed``.

        Generating a city costs a few milliseconds, which is significant when
        the seed pool is small and episodes are short, so layouts are memoised.
        The cache is bounded to keep memory flat over long runs.
        """
        cached = self._city_cache.get(city_seed)
        if cached is not None:
            return cached
        layout = generate_city(
            city_seed,
            self.city_config,
            clearance=max(
                self.env_config.spawn_clearance,
                self.drone_config.collision_radius + 0.6,
            ),
        )
        if len(self._city_cache) >= 128:
            self._city_cache.pop(next(iter(self._city_cache)))
        self._city_cache[city_seed] = layout
        return layout

    # ---- episode setup ------------------------------------------------------

    def _sample_start_and_target(self, layout: CityLayout) -> tuple[np.ndarray, np.ndarray]:
        """Pick a free start and a free target at least the min distance apart.

        Both are drawn from the city's precomputed free-space grid, so neither
        can spawn inside a building. The distance constraint is relaxed
        progressively if the city is too dense to satisfy it, which guarantees
        termination rather than looping forever.
        """
        free = layout.free_positions
        if free.shape[0] < 2:
            raise RuntimeError(f"City seed {layout.seed} has no free space to spawn in")

        min_distance = float(self.env_config.min_start_target_distance)
        for attempt in range(60):
            idx_start, idx_target = self._rng.choice(free.shape[0], size=2, replace=False)
            start_xy = free[idx_start]
            target_xy = free[idx_target]

            start_z = float(
                self._rng.uniform(
                    self.drone_config.spawn_altitude_min, self.drone_config.spawn_altitude_max
                )
            )
            target_z = float(
                self._rng.uniform(
                    self.env_config.target_altitude_min, self.env_config.target_altitude_max
                )
            )
            start = np.array([start_xy[0], start_xy[1], start_z], dtype=np.float64)
            target = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float64)

            if float(np.linalg.norm(target - start)) >= min_distance:
                return start, target
            # Relax the constraint every 15 failed attempts.
            if attempt > 0 and attempt % 15 == 0:
                min_distance *= 0.75

        return start, target  # best effort after relaxation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        ``options`` may carry ``city_seed``, ``start`` or ``target`` to pin any
        of them explicitly - used by ``compare_learning.py`` so every checkpoint
        flies the identical scenario.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})
        city_seed = int(options.get("city_seed", self._sample_city_seed()))
        self.city = self._get_city(city_seed)
        self._box_mins, self._box_maxs = self.city.box_arrays()

        start, target = self._sample_start_and_target(self.city)
        if options.get("start") is not None:
            start = np.asarray(options["start"], dtype=np.float64).reshape(3)
        if options.get("target") is not None:
            target = np.asarray(options["target"], dtype=np.float64).reshape(3)

        self.start_position = start.copy()
        self.target_position = target.copy()

        initial_yaw = float(options.get("yaw", self._rng.uniform(-np.pi, np.pi)))
        self.drone.reset(start, yaw=initial_yaw)

        self.step_count = 0
        self.episode_reward = 0.0
        self.energy_used_total = 0.0
        self.path_length = 0.0
        self.status = EpisodeOutcome.RUNNING
        self.trajectory = [start.copy()]
        self._previous_distance = float(np.linalg.norm(target - start))
        self._last_ray_distances = self._read_sensors()

        if self.render_mode is not None:
            self._ensure_renderer()
            self._renderer.build_city(self.city)
            self._renderer.set_markers(start, target, self.env_config.target_radius)

        observation = self.encoder.encode(self.drone, self.target_position, self._last_ray_distances)
        return observation, self._build_info(reward_breakdown=None)

    # ---- stepping -----------------------------------------------------------

    def _read_sensors(self) -> np.ndarray:
        """Current range-sensor readings in metres."""
        return self.sensors.read(
            self.drone.state.position,
            self.drone.state.yaw,
            self._box_mins,
            self._box_maxs,
            ceiling_z=self.city_config.max_altitude,
        )

    def _swept_sq_distances(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Squared obstacle distance at each swept sample along ``start->end``.

        Sampling the move rather than only its endpoint prevents tunnelling
        through thin walls when the per-step displacement is large. The final
        element corresponds to ``end``, which lets the caller reuse it as the
        post-move clearance.
        """
        steps = max(1, int(self.env_config.collision_substeps))
        # Skip fraction 0: the start pose was collision-free on the last step.
        fractions = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
        samples = start[None, :] + (end - start)[None, :] * fractions
        return min_sq_distance_to_boxes(samples, self._box_mins, self._box_maxs)

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        """True when the drone left the navigable volume."""
        assert self.city is not None
        low = self.city.bounds_min
        high = self.city.bounds_max
        below = position[2] < self.drone_config.min_altitude
        outside_xy = bool(np.any(position[:2] < low[:2]) or np.any(position[:2] > high[:2]))
        above = position[2] > high[2]
        return below or outside_xy or above

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one control step."""
        if self.city is None:
            raise RuntimeError("step() called before reset()")
        action_index = int(action)
        if not 0 <= action_index < len(self.action_set):
            raise ValueError(f"Action {action_index} outside [0, {len(self.action_set)})")

        spec = self.action_set[action_index]
        previous_position = self.drone.state.position.copy()
        proposed_position, velocity, yaw, energy_used = self.drone.propose_step(spec)

        # --- collision & bounds resolution ---------------------------------
        # A swept test prevents tunnelling through thin walls at high speed.
        # The same query yields the post-move clearance (the last sample is the
        # proposed position), so no second distance query is needed.
        radius = self.drone_config.collision_radius
        swept_sq = self._swept_sq_distances(previous_position, proposed_position)
        collided = bool((swept_sq <= radius * radius).any())
        out_of_bounds = self._out_of_bounds(proposed_position)

        if collided:
            # Stop at the pre-collision pose: the drone crashed, so it does not
            # continue into the building.
            self.drone.commit(previous_position, np.zeros(3), yaw, energy_used)
        elif out_of_bounds:
            self.drone.commit(previous_position, np.zeros(3), yaw, energy_used)
        else:
            self.drone.commit(proposed_position, velocity, yaw, energy_used)

        self.step_count += 1
        self.energy_used_total += float(energy_used)
        self.path_length += norm3(self.drone.state.position - previous_position)
        self.trajectory.append(self.drone.state.position.copy())

        self._last_ray_distances = self._read_sensors()
        current_distance = norm3(self.target_position - self.drone.state.position)

        # --- terminal conditions -------------------------------------------
        reached_target = current_distance <= self.env_config.target_radius and not collided
        battery_empty = self.drone.state.battery <= 0.0
        timed_out = self.step_count >= self.env_config.max_steps

        terminated = bool(reached_target or collided or battery_empty or out_of_bounds)
        # Truncation only applies when the episode did not end intrinsically.
        truncated = bool(timed_out and not terminated)

        if collided or out_of_bounds:
            # The drone was held at its previous pose, so the swept samples do
            # not describe where it actually is.
            clearance = closest_distance_to_boxes(
                self.drone.state.position, self._box_mins, self._box_maxs
            )
        else:
            clearance = float(np.sqrt(swept_sq[-1]))

        breakdown = self.reward_function.compute(
            previous_distance=self._previous_distance,
            current_distance=current_distance,
            energy_used=energy_used,
            obstacle_clearance=clearance,
            altitude=float(self.drone.state.position[2]),
            reached_target=reached_target,
            collided=collided,
            battery_empty=battery_empty and not reached_target,
            out_of_bounds=out_of_bounds,
            timed_out=truncated,
        )
        self._previous_distance = current_distance
        self.episode_reward += breakdown.total

        if reached_target:
            self.status = EpisodeOutcome.TARGET_REACHED
        elif collided:
            self.status = EpisodeOutcome.COLLISION
        elif out_of_bounds:
            self.status = EpisodeOutcome.OUT_OF_BOUNDS
        elif battery_empty:
            self.status = EpisodeOutcome.BATTERY_EMPTY
        elif truncated:
            self.status = EpisodeOutcome.TIMEOUT

        observation = self.encoder.encode(
            self.drone, self.target_position, self._last_ray_distances
        )
        info = self._build_info(
            reward_breakdown=breakdown,
            clearance=clearance,
            distance=current_distance,
            reached_target=reached_target,
            collided=collided,
            battery_empty=battery_empty,
            out_of_bounds=out_of_bounds,
            timed_out=truncated,
        )

        if self.env_config.debug:
            self._print_debug(action_index, breakdown, current_distance)
        if self.render_mode is not None:
            # Push the new pose into the scene, but do NOT grab a frame here:
            # in rgb_array mode the caller decides when to capture, and
            # rendering a frame per step only to discard it doubles the cost of
            # recording (frame capture dominates render time).
            self._sync_renderer()
            if collided:
                self._renderer.flash_event(self.drone.state.position, (1.0, 0.2, 0.15))
            elif reached_target:
                self._renderer.flash_event(self.drone.state.position, (0.2, 1.0, 0.3))

        return observation, float(breakdown.total), terminated, truncated, info

    # ---- info / debug -------------------------------------------------------

    def _build_info(
        self,
        reward_breakdown,
        clearance: float | None = None,
        distance: float | None = None,
        reached_target: bool = False,
        collided: bool = False,
        battery_empty: bool = False,
        out_of_bounds: bool = False,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        """Assemble the ``info`` dict, including the reward decomposition."""
        assert self.city is not None
        if distance is None:
            distance = norm3(self.target_position - self.drone.state.position)
        info: dict[str, Any] = {
            "city_seed": self.city.seed,
            "seed_pool": self.env_config.seed_pool,
            "step": self.step_count,
            "status": self.status,
            "distance_to_target": distance,
            "battery": float(self.drone.state.battery),
            "battery_fraction": self.drone.battery_fraction,
            "energy_used_total": float(self.energy_used_total),
            "path_length": float(self.path_length),
            "speed": self.drone.speed,
            "altitude": self.drone.altitude,
            "position": self.drone.state.position.copy(),
            "target": self.target_position.copy(),
            "min_ray_distance": float(np.min(self._last_ray_distances)),
            "clearance": float(clearance) if clearance is not None else None,
            "success": bool(reached_target),
            "collision": bool(collided),
            "battery_empty": bool(battery_empty),
            "out_of_bounds": bool(out_of_bounds),
            "timeout": bool(timed_out),
            "episode_reward": float(self.episode_reward),
        }
        if reward_breakdown is not None:
            info.update(reward_breakdown.as_dict())
            info["reward_components"] = dict(reward_breakdown.components)
        else:
            info.update(
                {
                    "step_penalty": 0.0,
                    "energy_cost": 0.0,
                    "progress_reward": 0.0,
                    "proximity_penalty": 0.0,
                    "altitude_penalty": 0.0,
                    "terminal_reward": 0.0,
                    "total": 0.0,
                }
            )
        return info

    def _print_debug(self, action_index: int, breakdown, distance: float) -> None:
        """Print a one-line reward decomposition (debug mode only)."""
        parts = " ".join(f"{k}={v:+.3f}" for k, v in breakdown.components.items() if v != 0.0)
        _LOG.info(
            "step=%3d action=%-10s d=%6.2f batt=%5.1f r=%+7.3f | %s",
            self.step_count,
            self.action_set[action_index].name,
            distance,
            self.drone.state.battery,
            breakdown.total,
            parts or "(no active components)",
        )

    # ---- episode summary ----------------------------------------------------

    def episode_summary(self) -> dict[str, Any]:
        """Aggregate metrics for the episode that just finished."""
        assert self.city is not None
        straight_line = float(np.linalg.norm(self.target_position - self.start_position))
        return {
            "city_seed": self.city.seed,
            "seed_pool": self.env_config.seed_pool,
            "status": self.status,
            "success": self.status == EpisodeOutcome.TARGET_REACHED,
            "collision": self.status == EpisodeOutcome.COLLISION,
            "battery_empty": self.status == EpisodeOutcome.BATTERY_EMPTY,
            "out_of_bounds": self.status == EpisodeOutcome.OUT_OF_BOUNDS,
            "timeout": self.status == EpisodeOutcome.TIMEOUT,
            "reward": float(self.episode_reward),
            "steps": int(self.step_count),
            "energy": float(self.energy_used_total),
            "battery_remaining": float(self.drone.state.battery),
            "path_length": float(self.path_length),
            "flight_time": float(self.step_count * self.drone_config.dt),
            "final_distance": float(
                np.linalg.norm(self.target_position - self.drone.state.position)
            ),
            "straight_line_distance": straight_line,
            "path_efficiency": (
                straight_line / self.path_length if self.path_length > 1e-6 else 0.0
            ),
        }

    def trajectory_array(self) -> np.ndarray:
        """Flown trajectory as an ``(T, 3)`` array."""
        return np.asarray(self.trajectory, dtype=np.float64)

    # ---- rendering ----------------------------------------------------------

    def _ensure_renderer(self) -> None:
        """Create the PyBullet renderer on first use."""
        if self._renderer is not None:
            return
        from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig

        cfg = RenderConfig(**self._render_config) if self._render_config else RenderConfig()
        if self.render_mode == "rgb_array":
            cfg.mode = "direct"
        self._renderer = PyBulletRenderer(cfg)

    def _sync_renderer(self) -> None:
        """Mirror the current drone state into the PyBullet scene (no capture)."""
        if self.render_mode is None:
            return
        self._ensure_renderer()
        assert self._renderer is not None

        endpoints = None
        hit_mask = None
        if self._renderer.config.show_rays:
            endpoints = self.sensors.ray_endpoints(
                self.drone.state.position, self.drone.state.yaw, self._last_ray_distances
            )
            hit_mask = [
                bool(d < self.sensor_config.max_range - 1e-6) for d in self._last_ray_distances
            ]
        self._renderer.update(
            self.drone.state.position,
            self.drone.state.yaw,
            ray_endpoints=endpoints,
            ray_hit_mask=hit_mask,
        )

    def render(self) -> np.ndarray | None:
        """Update the scene; return an RGB frame in ``rgb_array`` mode.

        Frame capture is expensive, so ``step`` deliberately does not call this
        - it calls :meth:`_sync_renderer` instead. Recording code calls
        ``render()`` (or the renderer's ``capture_frame``) at whatever rate it
        actually needs.
        """
        self._sync_renderer()
        if self.render_mode == "rgb_array" and self._renderer is not None:
            return self._renderer.capture_frame()
        return None

    @property
    def renderer(self):
        """The underlying :class:`PyBulletRenderer`, or ``None`` when headless."""
        return self._renderer

    def close(self) -> None:
        """Release the renderer, if any."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
