"""Exploration schedules (Phase 8).

Epsilon is decayed against *environment steps* rather than episodes. Episode
length varies enormously here - an early crash lasts a handful of steps while a
timeout runs the full limit - so an episode-indexed schedule would make the
exploration rate depend on how badly the agent happens to be doing, which is
both unprincipled and hard to reproduce across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["LinearEpsilonSchedule", "EpsilonGreedy"]


@dataclass
class LinearEpsilonSchedule:
    """Linear decay from ``start`` to ``end`` over ``decay_steps`` env steps."""

    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 100_000

    def value(self, step: int) -> float:
        """Epsilon at global env step ``step``."""
        if self.decay_steps <= 0:
            return float(self.end)
        fraction = min(1.0, max(0.0, float(step) / float(self.decay_steps)))
        return float(self.start + fraction * (self.end - self.start))


class EpsilonGreedy:
    """Epsilon-greedy action selection with an explicit step counter.

    The policy owns its own ``numpy.random.Generator`` so exploration is
    reproducible independently of global RNG state, and the counter is part of
    :meth:`state_dict` so a resumed run continues the schedule rather than
    restarting exploration from scratch.
    """

    def __init__(
        self,
        num_actions: int,
        schedule: LinearEpsilonSchedule | None = None,
        seed: int | None = None,
    ) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.num_actions = int(num_actions)
        self.schedule = schedule or LinearEpsilonSchedule()
        self._rng = np.random.default_rng(seed)
        self._step = 0

    @classmethod
    def from_config(
        cls,
        num_actions: int,
        config: Mapping[str, Any],
        seed: int | None = None,
    ) -> "EpsilonGreedy":
        """Build from the ``dqn:`` YAML block."""
        schedule = LinearEpsilonSchedule(
            start=float(config.get("epsilon_start", 1.0)),
            end=float(config.get("epsilon_end", 0.05)),
            decay_steps=int(config.get("epsilon_decay_steps", 100_000)),
        )
        return cls(num_actions=num_actions, schedule=schedule, seed=seed)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.schedule.value(self._step)

    @property
    def step_count(self) -> int:
        """Environment steps counted by the schedule so far."""
        return self._step

    def increment(self, steps: int = 1) -> None:
        """Advance the schedule by ``steps`` environment steps."""
        self._step += int(steps)

    def should_explore(self) -> bool:
        """Draw whether this action should be random."""
        return bool(self._rng.random() < self.epsilon)

    def random_action(self) -> int:
        """Uniformly sample an action index."""
        return int(self._rng.integers(0, self.num_actions))

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "schedule": {
                "start": self.schedule.start,
                "end": self.schedule.end,
                "decay_steps": self.schedule.decay_steps,
            },
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._step = int(state.get("step", 0))
        schedule = state.get("schedule")
        if schedule:
            self.schedule = LinearEpsilonSchedule(
                start=float(schedule["start"]),
                end=float(schedule["end"]),
                decay_steps=int(schedule["decay_steps"]),
            )
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state
