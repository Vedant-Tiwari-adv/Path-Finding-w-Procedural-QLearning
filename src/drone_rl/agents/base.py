"""Agent interface shared by every algorithm (Phase 14 extension point).

The trainer, evaluator and demo scripts talk to agents *only* through
:class:`Agent`. A future proposed algorithm (energy-aware DQN, VAE+DQN,
world-model DQN, ...) can therefore be dropped in by implementing this
interface and registering it, with **no changes to the environment, the reward
function, the training loop or the evaluation protocol** - which is exactly the
condition the experiment-fairness requirement demands.

Contract
--------
``act(obs, evaluation)``
    Choose an action. ``evaluation=True`` means "no exploration".
``observe(...)``
    Receive one transition. ``terminated`` is the *bootstrapping* flag and must
    exclude time-limit truncation.
``update()``
    Perform learning; return a dict of scalars for logging (may be empty when
    the agent is still warming up).
``state_dict`` / ``load_state_dict``
    Checkpoint round-trip, including optimiser and exploration state.
"""

from __future__ import annotations

import abc
from typing import Any, Mapping

import numpy as np

__all__ = ["Agent"]


class Agent(abc.ABC):
    """Abstract base class for all learning and non-learning agents."""

    #: Short identifier written into checkpoints, used to pick the right class
    #: when loading.
    algorithm: str = "base"

    @abc.abstractmethod
    def act(self, observation: np.ndarray, evaluation: bool = False) -> int:
        """Return a discrete action index for ``observation``."""

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one environment transition.

        ``terminated`` must be ``True`` only for intrinsic end states; a
        time-limit cutoff belongs in ``truncated`` so the learner still
        bootstraps from ``next_observation``.
        """
        return None

    def update(self) -> dict[str, float]:
        """Run one learning step; return scalars to log. Empty when idle."""
        return {}

    def on_episode_end(self, episode_index: int, summary: Mapping[str, Any]) -> None:
        """Hook called after each episode (schedules, statistics, curricula)."""
        return None

    def state_dict(self) -> dict[str, Any]:
        """Serialisable agent state for checkpointing."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore agent state produced by :meth:`state_dict`."""
        return None

    def train_mode(self) -> None:
        """Switch any networks into training mode."""
        return None

    def eval_mode(self) -> None:
        """Switch any networks into evaluation mode."""
        return None

    @property
    def diagnostics(self) -> dict[str, float]:
        """Scalars describing current agent state (e.g. epsilon), for logging."""
        return {}
