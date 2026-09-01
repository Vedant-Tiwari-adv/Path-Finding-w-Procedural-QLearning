"""Phase 6 - modular reward function with per-component reporting.

Every weight lives in YAML (``reward:`` block); nothing is hard-coded here.
:meth:`RewardFunction.compute` returns a :class:`RewardBreakdown` rather than a
bare float, so ``info`` can expose *why* the agent received what it did - which
is essential for diagnosing bad behaviour and is required by the project spec.

Design notes
------------
* **Progress shaping is potential-based.** The progress term is
  ``progress_weight * (d_prev - d_now)``, i.e. ``-w * (Phi(s') - Phi(s))`` with
  potential ``Phi(s) = w * d(s)``. Ng et al. (1999) show this form leaves the
  optimal policy unchanged, so the shaping accelerates learning without
  inventing a new objective. A naive ``-w * d(s)`` per-step penalty would *not*
  have that guarantee.
* **Proximity penalty.** Rewarding progress alone encourages flying straight at
  walls, because closing distance to the target pays on every step while the
  crash only costs once, at the end, discounted. The proximity term exists to
  cancel that, and it must be *sized* to do so: at the maximum closing speed the
  progress term pays ``progress * max_speed * dt`` per step, so the penalty has
  to exceed that before impact or the per-step reward of a beeline stays
  positive all the way into the building. With the shipped weights the total
  step reward turns negative below roughly 2.5 m of clearance, while leaving
  the centre of a road (about 3 m clear) unpenalised.

  It is measured against the **distance to the nearest building**, not the
  smallest sensor reading: the ray array also sees the ground and the ceiling,
  and penalising those here would conflate "too close to a wall" with "flying
  low", which the altitude term already handles.
* **Terminal rewards are not double-counted** with step costs: on a terminal
  step the event bonus/penalty is added to that step's shaped reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

__all__ = ["RewardConfig", "RewardBreakdown", "RewardFunction"]


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Mirrors the ``reward:`` block of the YAML config."""

    target: float = 100.0
    collision: float = -100.0
    battery_failure: float = -100.0
    out_of_bounds: float = -50.0
    timeout: float = 0.0
    step: float = -0.1
    energy: float = -0.05
    progress: float = 1.0
    proximity: float = -2.0
    safety_distance: float = 3.0
    altitude_penalty: float = -0.3
    min_safe_altitude: float = 2.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "RewardConfig":
        """Build from a YAML block, ignoring unrelated keys."""
        if not mapping:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in mapping.items() if k in known})

    def as_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class RewardBreakdown:
    """Per-component reward decomposition for one environment step."""

    step_penalty: float = 0.0
    energy_cost: float = 0.0
    progress_reward: float = 0.0
    proximity_penalty: float = 0.0
    altitude_penalty: float = 0.0
    terminal_reward: float = 0.0
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """Flat component dict.

        Built explicitly rather than via ``dataclasses.asdict``, which
        deep-copies and showed up as a measurable share of environment step
        time when called on every step.
        """
        return {
            "step_penalty": self.step_penalty,
            "energy_cost": self.energy_cost,
            "progress_reward": self.progress_reward,
            "proximity_penalty": self.proximity_penalty,
            "altitude_penalty": self.altitude_penalty,
            "terminal_reward": self.terminal_reward,
            "total": self.total,
        }


class RewardFunction:
    """Computes the shaped reward from a transition.

    The function is stateless with respect to episodes: the caller supplies the
    previous and current distance to target, so nothing needs resetting.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        *,
        previous_distance: float,
        current_distance: float,
        energy_used: float,
        obstacle_clearance: float,
        altitude: float,
        reached_target: bool,
        collided: bool,
        battery_empty: bool,
        out_of_bounds: bool,
        timed_out: bool,
    ) -> RewardBreakdown:
        """Return the reward decomposition for a single transition.

        Parameters
        ----------
        previous_distance, current_distance:
            Euclidean distance to the target before and after the move.
        energy_used:
            Battery units consumed this step.
        obstacle_clearance:
            Distance in metres from the drone to the nearest *building* surface
            (``inf`` in an empty city). Deliberately not the smallest sensor
            reading - see the module docstring.
        altitude:
            Current height above ground.
        reached_target, collided, battery_empty, out_of_bounds, timed_out:
            Terminal-event flags for this step.
        """
        cfg = self.config
        breakdown = RewardBreakdown()

        breakdown.step_penalty = float(cfg.step)
        breakdown.energy_cost = float(cfg.energy * energy_used)

        # Potential-based shaping: positive when the drone closed the gap.
        breakdown.progress_reward = float(cfg.progress * (previous_distance - current_distance))

        # Penalise flying closer to a building than the safety margin, scaled by
        # how deep the incursion is. Zero once the drone is clear of the margin.
        if np.isfinite(obstacle_clearance) and obstacle_clearance < cfg.safety_distance:
            severity = (cfg.safety_distance - obstacle_clearance) / max(
                cfg.safety_distance, 1e-6
            )
            breakdown.proximity_penalty = float(cfg.proximity * severity)

        # Discourage skimming the ground, which otherwise becomes a cheap way to
        # avoid buildings entirely on cities with short blocks.
        if altitude < cfg.min_safe_altitude:
            deficit = (cfg.min_safe_altitude - altitude) / max(cfg.min_safe_altitude, 1e-6)
            breakdown.altitude_penalty = float(cfg.altitude_penalty * deficit)

        terminal = 0.0
        if reached_target:
            terminal += cfg.target
        if collided:
            terminal += cfg.collision
        if battery_empty:
            terminal += cfg.battery_failure
        if out_of_bounds:
            terminal += cfg.out_of_bounds
        if timed_out:
            terminal += cfg.timeout
        breakdown.terminal_reward = float(terminal)

        breakdown.total = float(
            breakdown.step_penalty
            + breakdown.energy_cost
            + breakdown.progress_reward
            + breakdown.proximity_penalty
            + breakdown.altitude_penalty
            + breakdown.terminal_reward
        )
        breakdown.components = {
            "step_penalty": breakdown.step_penalty,
            "energy_cost": breakdown.energy_cost,
            "progress_reward": breakdown.progress_reward,
            "proximity_penalty": breakdown.proximity_penalty,
            "altitude_penalty": breakdown.altitude_penalty,
            "terminal_reward": breakdown.terminal_reward,
        }
        return breakdown
