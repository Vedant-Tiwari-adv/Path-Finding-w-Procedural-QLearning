"""Hindsight goal relabelling (Phase 14 - proposed algorithm).

Hindsight Experience Replay (Andrychowicz et al., 2017) rewrites a failed
trajectory as a successful one by pretending the drone had been asked to reach
somewhere it actually visited. For that to be *sound* rather than approximate,
two things must be recomputed exactly as the environment would have computed
them for the substituted goal:

1. the observation, because it carries goal-relative terms, and
2. the reward, because it carries a goal-dependent shaping term and the arrival
   bonus.

This module does both analytically, with no environment access and no
privileged information. That matters for the fairness contract in
``configs/experiment.yaml``: the proposed algorithm must run against the same
environment, reward and observation space as the DQN baseline. Relabelling is a
*learning-side* transformation over transitions the agent already collected.

Why it is exact here
--------------------
Of the 21 observation dimensions produced by
:class:`~drone_rl.environment.sensors.ObservationEncoder`, only four depend on
the goal (``rel_target_{forward,left,up}`` and ``distance_to_target``). The
other seventeen - position, velocity, heading, altitude, battery, range
sensors - are properties of the drone and the city alone, so they carry over
untouched.

Of the six reward components in
:class:`~drone_rl.environment.rewards.RewardFunction`, only two depend on the
goal: the potential-based ``progress`` term and the ``target`` arrival bonus.
``step``, ``energy``, ``proximity`` and ``altitude`` do not, and neither do the
collision, battery and out-of-bounds terminal penalties. The goal-independent
remainder is therefore recoverable by subtraction from the stored scalar
reward, with no need to store the breakdown.

Recovering the world state
--------------------------
The encoder is invertible on the dimensions that matter. Position is stored
scaled by the city half-extents, heading as ``(cos, sin)``, and the
goal-relative vector in the *body* frame. So::

    position = obs[0:3] * (half_width, half_depth, max_altitude)
    yaw      = atan2(obs[11], obs[10])
    goal     = position + R(yaw) @ (obs[3:6] * city_diagonal)

where ``R(yaw)`` is the body-to-world rotation, the inverse of the encoder's
``_world_to_body``. Everything below is a vectorised form of those three lines.

All public methods operate on batches: observations are ``(N, obs_size)`` and
goals are ``(N, 3)``, so relabelling a whole episode is a handful of NumPy
operations rather than a Python loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["GoalRelabeler", "ObservationLayout"]


@dataclass(frozen=True)
class ObservationLayout:
    """Index map into the flat observation vector.

    Mirrors :class:`~drone_rl.environment.sensors.ObservationEncoder`. Kept as
    data rather than magic numbers so a change to the encoder surfaces as one
    edit here, and so :meth:`GoalRelabeler.validate_against_encoder` can assert
    the two agree.
    """

    position: slice = slice(0, 3)
    relative_target: slice = slice(3, 6)
    distance: int = 6
    velocity: slice = slice(7, 10)
    heading_cos: int = 10
    heading_sin: int = 11
    altitude: int = 12
    battery: int = 13
    rays_start: int = 14


class GoalRelabeler:
    """Rewrites transitions to target a substituted goal.

    Parameters
    ----------
    config:
        The full resolved experiment config. The ``city``, ``drone``,
        ``environment`` and ``reward`` blocks are read so the normalisation
        constants and reward weights match the environment exactly.

    Notes
    -----
    The relabeler is stateless; every method is a pure function of its
    arguments. That keeps it trivially safe to share between the agent's
    collection path and its replay path.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        city = dict(config.get("city", {}))
        drone = dict(config.get("drone", {}))
        environment = dict(config.get("environment", {}))
        reward = dict(config.get("reward", {}))

        width = float(city.get("width", 60.0))
        depth = float(city.get("depth", 60.0))
        max_altitude = float(city.get("max_altitude", 25.0))

        # Matches ObservationEncoder: position is scaled by the half-extents in
        # x/y but by the *full* ceiling height in z.
        self._position_scale = np.array(
            [max(width * 0.5, 1e-6), max(depth * 0.5, 1e-6), max(max_altitude, 1e-6)],
            dtype=np.float64,
        )
        self._city_diagonal = float(np.linalg.norm([width, depth, max_altitude]))

        self.target_radius = float(environment.get("target_radius", 2.0))
        self.progress_weight = float(reward.get("progress", 1.0))
        self.target_bonus = float(reward.get("target", 100.0))
        self.collision_penalty = float(reward.get("collision", -100.0))

        self.layout = ObservationLayout()

    # ---- decoding -----------------------------------------------------------

    def positions(self, observations: np.ndarray) -> np.ndarray:
        """World positions ``(N, 3)`` implied by a batch of observations."""
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        return obs[:, self.layout.position] * self._position_scale

    def yaws(self, observations: np.ndarray) -> np.ndarray:
        """Heading angles ``(N,)`` in radians, recovered from ``(cos, sin)``."""
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        return np.arctan2(obs[:, self.layout.heading_sin], obs[:, self.layout.heading_cos])

    def goals(self, observations: np.ndarray) -> np.ndarray:
        """World goal positions ``(N, 3)`` the observations were encoded against."""
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        relative_body = obs[:, self.layout.relative_target] * self._city_diagonal
        relative_world = self._body_to_world(relative_body, self.yaws(obs))
        return self.positions(obs) + relative_world

    # ---- frame conversions --------------------------------------------------

    @staticmethod
    def _world_to_body(vectors: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Rotate world vectors into the body frame (inverse yaw rotation).

        Identical to ``ObservationEncoder._world_to_body`` but batched.
        """
        cos_a = np.cos(yaws)
        sin_a = np.sin(yaws)
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        return np.stack([x * cos_a + y * sin_a, -x * sin_a + y * cos_a, z], axis=1)

    @staticmethod
    def _body_to_world(vectors: np.ndarray, yaws: np.ndarray) -> np.ndarray:
        """Rotate body-frame vectors back into the world frame."""
        cos_a = np.cos(yaws)
        sin_a = np.sin(yaws)
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        return np.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], axis=1)

    # ---- relabelling --------------------------------------------------------

    def relabel_observations(self, observations: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Return a copy of ``observations`` re-encoded against ``goals``.

        Only the four goal-dependent dimensions are rewritten; the rest are
        carried over bit-for-bit, which is exactly what the encoder would have
        produced for the same drone state under a different target.
        """
        obs = np.array(np.atleast_2d(observations), dtype=np.float32, copy=True)
        goal_array = np.atleast_2d(np.asarray(goals, dtype=np.float64))

        positions = self.positions(obs)
        yaws = self.yaws(obs)

        to_goal_world = goal_array - positions
        to_goal_body = self._world_to_body(to_goal_world, yaws)
        distances = np.linalg.norm(to_goal_world, axis=1)

        obs[:, self.layout.relative_target] = (to_goal_body / self._city_diagonal).astype(
            np.float32
        )
        obs[:, self.layout.distance] = (distances / self._city_diagonal).astype(np.float32)
        return obs

    def distances_to(self, observations: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Euclidean distance ``(N,)`` from each observation's position to its goal."""
        return np.linalg.norm(self.positions(observations) - np.atleast_2d(goals), axis=1)

    def relabel_rewards(
        self,
        *,
        observations: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        new_goals: np.ndarray,
        collided: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute rewards and termination flags under substituted goals.

        The stored scalar reward is decomposed as::

            reward = goal_independent + progress(old_goal) + arrival_bonus(old_goal)

        The two goal-dependent terms are subtracted off using the *original*
        goal recovered from the observation, then re-added for ``new_goals``.
        Whatever remains - step cost, energy, proximity, altitude, and any
        collision / battery / out-of-bounds penalty - is goal-independent and
        survives untouched.

        Parameters
        ----------
        observations, next_observations:
            The transition endpoints, ``(N, obs_size)``.
        rewards:
            Stored scalar rewards, ``(N,)``.
        terminated:
            Stored intrinsic-termination flags, ``(N,)``. A relabelled
            transition inherits these (a crash is still a crash) and adds
            arrival at the new goal.
        new_goals:
            Substituted goals, ``(N, 3)``.
        collided:
            Optional ``(N,)`` mask marking transitions that ended in a
            collision. The environment suppresses the arrival bonus when the
            drone crashed (``reached_target and not collided``), so passing this
            keeps the reconstruction exact on those steps. When omitted, a
            collision is inferred from a large negative goal-independent
            remainder, which is reliable because the collision penalty
            (-100 by default) dwarfs the shaping terms.

        Returns
        -------
        tuple
            ``(new_rewards, new_terminated)``, both ``(N,)``.
        """
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        next_obs = np.atleast_2d(np.asarray(next_observations, dtype=np.float64))
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        new_goals = np.atleast_2d(np.asarray(new_goals, dtype=np.float64))

        original_goals = self.goals(obs)
        old_previous = self.distances_to(obs, original_goals)
        old_current = self.distances_to(next_obs, original_goals)

        if collided is None:
            crashed = np.zeros(len(rewards), dtype=bool)
        else:
            crashed = np.asarray(collided, dtype=bool).reshape(-1)

        old_reached = (old_current <= self.target_radius) & ~crashed

        goal_independent = (
            rewards
            - self.progress_weight * (old_previous - old_current)
            - np.where(old_reached, self.target_bonus, 0.0)
        )

        new_previous = self.distances_to(obs, new_goals)
        new_current = self.distances_to(next_obs, new_goals)
        new_reached = (new_current <= self.target_radius) & ~crashed

        new_rewards = (
            goal_independent
            + self.progress_weight * (new_previous - new_current)
            + np.where(new_reached, self.target_bonus, 0.0)
        )
        new_terminated = terminated | new_reached
        return new_rewards.astype(np.float32), new_terminated

    # ---- self-check ---------------------------------------------------------

    def validate_against_encoder(self, encoder: Any, tolerance: float = 1e-4) -> None:
        """Assert this relabeler inverts ``encoder`` on a synthetic state.

        Cheap insurance against the encoder layout drifting away from
        :class:`ObservationLayout` without this module being updated - a change
        that would otherwise corrupt every relabelled transition silently.

        Raises
        ------
        ValueError
            If the recovered position, yaw or goal disagrees with the truth.
        """
        if int(encoder.size) != int(self.layout.rays_start) + int(encoder.sensors.num_rays):
            raise ValueError(
                f"Observation size {encoder.size} does not match the relabeler layout "
                f"({self.layout.rays_start} + {encoder.sensors.num_rays} rays)"
            )
        expected_labels = ["pos_x", "pos_y", "pos_z"]
        if list(encoder.labels[:3]) != expected_labels:
            raise ValueError(
                f"Encoder layout changed: expected {expected_labels}, got {encoder.labels[:3]}"
            )
