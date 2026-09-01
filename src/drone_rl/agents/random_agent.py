"""Uniform random policy (Phase 7 baseline).

Serves two purposes: it validates the environment end-to-end before any
learning code is trusted, and it is the untrained reference in the Phase-11
learning-progression demo. It implements the same :class:`Agent` interface as
DQN, so the trainer, evaluator and demo run it without special-casing.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from drone_rl.agents.base import Agent

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    """Samples actions uniformly at random; learns nothing."""

    algorithm = "random"

    def __init__(self, num_actions: int, seed: int | None = None) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self.num_actions = int(num_actions)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, evaluation: bool = False) -> int:
        """Return a uniformly sampled action index."""
        return int(self._rng.integers(0, self.num_actions))

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "num_actions": self.num_actions,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state
