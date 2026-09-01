"""Experience replay (Phase 8).

A fixed-capacity circular buffer over pre-allocated NumPy arrays. Storing
transitions in contiguous arrays rather than a ``deque`` of tuples keeps
sampling to a single fancy-index per field, which matters because sampling
happens on every gradient step.

The stored ``terminated`` flag is the *bootstrapping* flag: it is ``True`` only
for intrinsic terminal states. Time-limit truncations are stored with
``terminated=False`` so the Bellman target still bootstraps from the successor
state, which is the correct handling of a time limit that is not part of the
MDP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Transition", "TransitionBatch", "ReplayBuffer"]


@dataclass(frozen=True)
class Transition:
    """A single environment transition."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool


@dataclass(frozen=True)
class TransitionBatch:
    """A minibatch of transitions as torch tensors on the target device."""

    observations: torch.Tensor  # (B, obs)
    actions: torch.Tensor  # (B, 1) int64
    rewards: torch.Tensor  # (B, 1)
    next_observations: torch.Tensor  # (B, obs)
    terminated: torch.Tensor  # (B, 1) float, 1.0 = do not bootstrap

    def __len__(self) -> int:
        return int(self.observations.shape[0])


class ReplayBuffer:
    """Uniform-sampling circular replay buffer.

    Parameters
    ----------
    capacity:
        Maximum transitions retained; oldest are overwritten.
    observation_size:
        Length of the flat observation vector.
    device:
        Device the sampled tensors are moved to.
    seed:
        Seed for the buffer's own sampling RNG, kept separate from global state
        so replay sampling is reproducible.
    """

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_size = int(observation_size)
        self.device = torch.device(device)

        self._observations = np.zeros((self.capacity, self.observation_size), dtype=np.float32)
        self._next_observations = np.zeros((self.capacity, self.observation_size), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=np.float32)

        self._cursor = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest when full.

        ``terminated`` must exclude time-limit truncation - see module docstring.
        """
        index = self._cursor
        self._observations[index] = observation
        self._next_observations[index] = next_observation
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._terminated[index] = float(bool(terminated))

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample ``batch_size`` transitions uniformly with replacement."""
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices: np.ndarray) -> TransitionBatch:
        """Materialise a :class:`TransitionBatch` for the given row indices."""
        to_device = lambda array: torch.as_tensor(array, device=self.device)  # noqa: E731
        return TransitionBatch(
            observations=to_device(self._observations[indices]),
            actions=to_device(self._actions[indices]).unsqueeze(1),
            rewards=to_device(self._rewards[indices]).unsqueeze(1),
            next_observations=to_device(self._next_observations[indices]),
            terminated=to_device(self._terminated[indices]).unsqueeze(1),
        )

    def state_dict(self) -> dict[str, object]:
        """Buffer contents and cursor, for resumable checkpoints."""
        return {
            "observations": self._observations[: self._size].copy(),
            "next_observations": self._next_observations[: self._size].copy(),
            "actions": self._actions[: self._size].copy(),
            "rewards": self._rewards[: self._size].copy(),
            "terminated": self._terminated[: self._size].copy(),
            "cursor": self._cursor,
            "size": self._size,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore buffer contents saved by :meth:`state_dict`."""
        size = int(state["size"])  # type: ignore[arg-type]
        if size > self.capacity:
            raise ValueError(
                f"Saved buffer holds {size} transitions but capacity is {self.capacity}"
            )
        self._observations[:size] = state["observations"]  # type: ignore[assignment]
        self._next_observations[:size] = state["next_observations"]  # type: ignore[assignment]
        self._actions[:size] = state["actions"]  # type: ignore[assignment]
        self._rewards[:size] = state["rewards"]  # type: ignore[assignment]
        self._terminated[:size] = state["terminated"]  # type: ignore[assignment]
        self._size = size
        self._cursor = int(state["cursor"]) % self.capacity  # type: ignore[arg-type]
