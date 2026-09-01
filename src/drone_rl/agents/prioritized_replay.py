"""Prioritised n-step replay (Phase 14 - proposed algorithm).

Two departures from the baseline's uniform 1-step buffer in
:mod:`drone_rl.agents.replay_buffer`, both aimed at the failure modes measured
on the DQN baseline run:

**n-step returns.** The baseline bootstraps one step at a time with
``gamma=0.995``. A successful delivery takes 110-220 steps, so the ``+100``
arrival bonus has to propagate back through nearly two hundred Bellman updates
before it can influence the first action of an episode. n-step returns carry
the reward back ``n`` transitions per update, cutting that chain by the same
factor and measurably reducing how long the agent spends unable to see any
consequence of reaching the target.

**Proportional prioritisation** (Schaul et al., 2016). Successes were 5-15% of
episodes on the baseline, so a uniform batch of 128 contains only a handful of
transitions from a trajectory that ever earned the bonus. Sampling in
proportion to TD error concentrates updates on the transitions the network is
currently wrong about, which in practice are exactly the rare rewarding ones.
Importance-sampling weights correct the bias this introduces.

Storage layout
--------------
Rather than store raw transitions and assemble n-step windows at sample time,
the caller supplies an already-accumulated window::

    (observation, action, n_step_return, bootstrap_observation, bootstrap_discount)

``bootstrap_discount`` is ``gamma**k`` for the ``k`` steps actually accumulated,
and **zero** when the window ran into an intrinsic terminal. Folding
termination into the discount means the Bellman target is a single unbranched
expression::

    y = n_step_return + bootstrap_discount * Q_target(bootstrap_observation)

which removes the ``(1 - done)`` factor that is easy to get wrong, and lets
truncated windows at an episode boundary be represented without a special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["SumTree", "NStepBatch", "PrioritizedReplayBuffer"]


class SumTree:
    """Fixed-capacity binary indexed tree over non-negative priorities.

    Supports ``O(log n)`` point updates and prefix-sum search, which is what
    proportional prioritisation needs: sampling a transition means drawing a
    uniform value in ``[0, total)`` and finding the leaf whose cumulative range
    contains it.

    The search is *vectorised over a whole batch*. Because the tree is a
    complete binary tree stored in a flat array, every element of a batch sits
    at the same depth after the same number of descent steps, so the descent is
    a fixed number of NumPy operations rather than a Python loop per sample.
    That matters here: sampling happens on every gradient step.

    Parameters
    ----------
    capacity:
        Number of leaves. Rounded up to the next power of two so the tree is
        complete; the padding leaves keep priority zero and are never selected.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = 1 << (int(capacity) - 1).bit_length()
        self._tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self._depth = int(self.capacity).bit_length() - 1

    @property
    def total(self) -> float:
        """Sum of all leaf priorities."""
        return float(self._tree[1])

    @property
    def max_priority(self) -> float:
        """Largest single leaf priority, or 0.0 when the tree is empty."""
        leaves = self._tree[self.capacity :]
        return float(leaves.max()) if leaves.size else 0.0

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Set the priorities of ``indices`` and repair the internal nodes.

        Duplicate indices are handled: the *last* value wins, matching the
        semantics of scalar assignment, and the internal nodes are rebuilt from
        the leaves rather than by delta-propagation so a repeated index cannot
        double-count.
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1) + self.capacity
        values = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if np.any(values < 0.0):
            raise ValueError("priorities must be non-negative")
        self._tree[idx] = values

        # Walk up recomputing parents from children. Recomputing (rather than
        # adding a delta) keeps the tree exact under duplicate indices and
        # avoids floating-point drift accumulating over millions of updates.
        parents = np.unique(idx >> 1)
        while parents.size and parents[0] >= 1:
            self._tree[parents] = self._tree[2 * parents] + self._tree[2 * parents + 1]
            if parents[0] == 1:
                break
            parents = np.unique(parents >> 1)

    def sample(self, values: np.ndarray) -> np.ndarray:
        """Return leaf indices whose prefix-sum ranges contain ``values``.

        ``values`` must lie in ``[0, total)``.
        """
        cursor = np.ones(len(values), dtype=np.int64)
        remaining = np.asarray(values, dtype=np.float64).copy()
        for _ in range(self._depth):
            left = 2 * cursor
            left_sum = self._tree[left]
            go_right = remaining > left_sum
            remaining -= np.where(go_right, left_sum, 0.0)
            cursor = left + go_right.astype(np.int64)
        return cursor - self.capacity


@dataclass(frozen=True)
class NStepBatch:
    """A minibatch of n-step windows as tensors on the target device."""

    observations: torch.Tensor  # (B, obs)
    actions: torch.Tensor  # (B, 1) int64
    returns: torch.Tensor  # (B, 1) accumulated n-step reward
    bootstrap_observations: torch.Tensor  # (B, obs)
    bootstrap_discounts: torch.Tensor  # (B, 1) gamma**k, 0 if terminal
    weights: torch.Tensor  # (B, 1) importance-sampling weights
    indices: np.ndarray  # (B,) buffer rows, for priority updates

    def __len__(self) -> int:
        return int(self.observations.shape[0])


class PrioritizedReplayBuffer:
    """Circular buffer of n-step windows with proportional prioritisation.

    Parameters
    ----------
    capacity:
        Maximum windows retained; oldest are overwritten.
    observation_size:
        Length of the flat observation vector.
    alpha:
        Prioritisation exponent. ``0`` recovers uniform sampling, ``1`` samples
        strictly in proportion to TD error.
    epsilon:
        Floor added to every priority so a transition with zero TD error can
        still be revisited.
    device:
        Device sampled tensors are moved to.
    seed:
        Seed for the buffer's sampling RNG, kept separate from global state.
    """

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        alpha: float = 0.6,
        epsilon: float = 1e-3,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_size = int(observation_size)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.device = torch.device(device)

        self._observations = np.zeros((self.capacity, self.observation_size), dtype=np.float32)
        self._bootstrap_observations = np.zeros(
            (self.capacity, self.observation_size), dtype=np.float32
        )
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._returns = np.zeros(self.capacity, dtype=np.float32)
        self._bootstrap_discounts = np.zeros(self.capacity, dtype=np.float32)

        self._tree = SumTree(self.capacity)
        self._cursor = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def add_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        bootstrap_observations: np.ndarray,
        bootstrap_discounts: np.ndarray,
        priorities: np.ndarray | None = None,
    ) -> None:
        """Insert many windows at once, wrapping the cursor as needed.

        Batched insertion exists because hindsight relabelling produces a whole
        episode's worth of synthetic windows at a time; adding them one by one
        would spend more time in Python than in the relabelling itself.

        New windows enter at the current maximum priority so each is replayed at
        least once before its priority is refined by an actual TD error.
        """
        observations = np.asarray(observations, dtype=np.float32)
        count = len(observations)
        if count == 0:
            return
        if count > self.capacity:
            # Keep the most recent windows; the older ones would be overwritten
            # within this same call anyway.
            observations = observations[-self.capacity :]
            actions = np.asarray(actions)[-self.capacity :]
            returns = np.asarray(returns)[-self.capacity :]
            bootstrap_observations = np.asarray(bootstrap_observations)[-self.capacity :]
            bootstrap_discounts = np.asarray(bootstrap_discounts)[-self.capacity :]
            if priorities is not None:
                priorities = np.asarray(priorities)[-self.capacity :]
            count = self.capacity

        rows = (self._cursor + np.arange(count)) % self.capacity
        self._observations[rows] = observations
        self._bootstrap_observations[rows] = np.asarray(
            bootstrap_observations, dtype=np.float32
        )
        self._actions[rows] = np.asarray(actions, dtype=np.int64)
        self._returns[rows] = np.asarray(returns, dtype=np.float32)
        self._bootstrap_discounts[rows] = np.asarray(bootstrap_discounts, dtype=np.float32)

        if priorities is None:
            seed_priority = max(self._tree.max_priority, self.epsilon)
            priority_values = np.full(count, seed_priority, dtype=np.float64)
        else:
            priority_values = (
                np.abs(np.asarray(priorities, dtype=np.float64)) + self.epsilon
            ) ** self.alpha
        self._tree.update(rows, priority_values)

        self._cursor = int((self._cursor + count) % self.capacity)
        self._size = int(min(self._size + count, self.capacity))

    def sample(self, batch_size: int, beta: float = 0.4) -> NStepBatch:
        """Draw ``batch_size`` windows in proportion to priority.

        Parameters
        ----------
        batch_size:
            Number of windows.
        beta:
            Importance-sampling exponent. ``0`` applies no correction, ``1``
            fully corrects the sampling bias. Annealing it towards 1 over
            training is standard: early updates tolerate bias in exchange for
            faster progress, later ones need the unbiased gradient.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        total = self._tree.total
        if not np.isfinite(total) or total <= 0.0:
            indices = self._rng.integers(0, self._size, size=int(batch_size))
            probabilities = np.full(len(indices), 1.0 / self._size, dtype=np.float64)
        else:
            # Stratified sampling: one draw per equal-probability stratum gives
            # lower-variance batches than independent draws over the whole range.
            edges = np.linspace(0.0, total, int(batch_size) + 1)
            values = self._rng.uniform(edges[:-1], edges[1:])
            indices = self._tree.sample(values)
            # A leaf can only be selected if it holds priority, but clamp anyway
            # so a floating-point edge case cannot index unwritten rows.
            indices = np.clip(indices, 0, max(self._size - 1, 0))
            probabilities = self._leaf_priorities(indices) / total

        weights = (self._size * np.maximum(probabilities, 1e-12)) ** (-float(beta))
        weights /= max(weights.max(), 1e-12)

        to_device = lambda array: torch.as_tensor(array, device=self.device)  # noqa: E731
        return NStepBatch(
            observations=to_device(self._observations[indices]),
            actions=to_device(self._actions[indices]).unsqueeze(1),
            returns=to_device(self._returns[indices]).unsqueeze(1),
            bootstrap_observations=to_device(self._bootstrap_observations[indices]),
            bootstrap_discounts=to_device(self._bootstrap_discounts[indices]).unsqueeze(1),
            weights=to_device(weights.astype(np.float32)).unsqueeze(1),
            indices=indices,
        )

    def _leaf_priorities(self, indices: np.ndarray) -> np.ndarray:
        """Raw stored priorities for the given rows."""
        return self._tree._tree[np.asarray(indices, dtype=np.int64) + self._tree.capacity]

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Refresh priorities from fresh absolute TD errors."""
        priorities = (
            np.abs(np.asarray(td_errors, dtype=np.float64)).reshape(-1) + self.epsilon
        ) ** self.alpha
        self._tree.update(np.asarray(indices, dtype=np.int64).reshape(-1), priorities)

    def state_dict(self) -> dict[str, object]:
        """Buffer contents and cursor, for resumable checkpoints."""
        return {
            "observations": self._observations[: self._size].copy(),
            "bootstrap_observations": self._bootstrap_observations[: self._size].copy(),
            "actions": self._actions[: self._size].copy(),
            "returns": self._returns[: self._size].copy(),
            "bootstrap_discounts": self._bootstrap_discounts[: self._size].copy(),
            "priorities": self._leaf_priorities(np.arange(self._size)).copy(),
            "cursor": self._cursor,
            "size": self._size,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore buffer contents saved by :meth:`state_dict`."""
        size = int(state["size"])  # type: ignore[arg-type]
        if size > self.capacity:
            raise ValueError(
                f"Saved buffer holds {size} windows but capacity is {self.capacity}"
            )
        self._observations[:size] = state["observations"]  # type: ignore[assignment]
        self._bootstrap_observations[:size] = state["bootstrap_observations"]  # type: ignore[assignment]
        self._actions[:size] = state["actions"]  # type: ignore[assignment]
        self._returns[:size] = state["returns"]  # type: ignore[assignment]
        self._bootstrap_discounts[:size] = state["bootstrap_discounts"]  # type: ignore[assignment]
        self._size = size
        self._cursor = int(state["cursor"]) % self.capacity  # type: ignore[arg-type]
        if size:
            self._tree.update(np.arange(size), np.asarray(state["priorities"]))
