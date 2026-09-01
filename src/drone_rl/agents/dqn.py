"""Deep Q-Network agent, implemented from scratch in PyTorch (Phase 8).

No RL library is used: the replay buffer, target network, epsilon-greedy policy
and Bellman update are all written here so the algorithm can be modified for
the research phase.

The update
----------
For a minibatch of transitions ``(s, a, r, s', d)`` where ``d`` is the
*intrinsic* termination flag:

.. math::

    y = r + \\gamma \\, (1 - d) \\, \\max_{a'} Q_{\\text{target}}(s', a')

    \\mathcal{L} = \\text{SmoothL1}\\big(Q_\\theta(s, a),\\; y\\big)

Two correctness points that are easy to get wrong and are handled explicitly:

1. **No bootstrapping from terminal states.** The ``(1 - d)`` factor zeroes the
   future term for intrinsic terminals. Because the environment reports
   time-limit cutoffs as ``truncated`` rather than ``terminated``, and the
   trainer stores only ``terminated``, a timeout correctly *keeps* its
   bootstrap - treating it as terminal would teach the agent that the world
   ends after ``max_steps``.
2. **The target is detached.** ``y`` is computed under ``torch.no_grad()`` from
   the frozen target network, so gradients never flow into the bootstrap.

``double_dqn`` is available as a config flag but defaults to ``False``: the
project's stated baseline is plain DQN.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from drone_rl.agents.base import Agent
from drone_rl.agents.exploration import EpsilonGreedy
from drone_rl.agents.network import QNetwork
from drone_rl.agents.replay_buffer import ReplayBuffer
from drone_rl.utils.logging import get_logger

__all__ = ["DQNConfig", "DQNAgent"]

_LOG = get_logger(__name__)


@dataclass
class DQNConfig:
    """DQN hyper-parameters (the ``dqn:`` YAML block)."""

    learning_rate: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 200_000
    min_replay_size: int = 2_000
    target_update_frequency: int = 1_000
    soft_update_tau: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    double_dqn: bool = False
    dueling: bool = False
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "DQNConfig":
        """Build from a YAML block, ignoring unrelated keys."""
        if not mapping:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in mapping.items() if k in known}
        if "hidden_sizes" in values:
            values["hidden_sizes"] = tuple(int(h) for h in values["hidden_sizes"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}
        data["hidden_sizes"] = list(data["hidden_sizes"])
        return data


class DQNAgent(Agent):
    """Deep Q-Network with experience replay and a target network.

    Parameters
    ----------
    observation_size:
        Length of the flat observation vector.
    num_actions:
        Size of the discrete action space.
    config:
        Hyper-parameters.
    device:
        Torch device for the networks and sampled batches.
    seed:
        Seeds the replay sampler and the exploration policy.
    """

    algorithm = "dqn"

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: DQNConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.observation_size = int(observation_size)
        self.num_actions = int(num_actions)

        self.q_network = QNetwork(
            observation_size=self.observation_size,
            num_actions=self.num_actions,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            dueling=self.config.dueling,
        ).to(self.device)

        # The target network is a frozen periodic copy; it is never optimised.
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss(beta=self.config.huber_delta)

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            observation_size=self.observation_size,
            device=self.device,
            seed=seed,
        )
        self.policy = EpsilonGreedy.from_config(
            num_actions=self.num_actions,
            config=self.config.as_dict(),
            seed=None if seed is None else seed + 1,
        )

        self._gradient_steps = 0
        self._env_steps = 0
        self._last_loss: float | None = None
        self._last_q_mean: float | None = None

    # ---- acting -------------------------------------------------------------

    def act(self, observation: np.ndarray, evaluation: bool = False) -> int:
        """Epsilon-greedy during training, greedy during evaluation."""
        if not evaluation and self.policy.should_explore():
            return self.policy.random_action()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                np.asarray(observation, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            q_values = self.q_network(obs_tensor)
            return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Q-values for one observation, for debugging and GUI overlays."""
        obs_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        return self.q_network(obs_tensor).squeeze(0).cpu().numpy()

    # ---- learning -----------------------------------------------------------

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store a transition.

        Only ``terminated`` is stored. ``truncated`` is deliberately discarded:
        a time-limit cutoff is not part of the MDP, so its successor state must
        still be bootstrapped from.
        """
        self.replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
        )
        self._env_steps += 1
        self.policy.increment()

    def ready_to_train(self) -> bool:
        """True once the replay buffer holds enough transitions to learn from."""
        return len(self.replay_buffer) >= max(
            self.config.min_replay_size, self.config.batch_size
        )

    def update(self) -> dict[str, float]:
        """Perform one gradient step on the Bellman error.

        Returns a dict with ``loss``, ``q_mean``, ``target_mean`` and ``epsilon``,
        or an empty dict while the buffer is still warming up.
        """
        if not self.ready_to_train():
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size)

        # Q(s, a) for the actions actually taken.
        q_selected = self.q_network(batch.observations).gather(1, batch.actions)

        with torch.no_grad():
            if self.config.double_dqn:
                # Online net picks the action, target net scores it.
                next_actions = self.q_network(batch.next_observations).argmax(dim=1, keepdim=True)
                next_q = self.target_network(batch.next_observations).gather(1, next_actions)
            else:
                next_q = self.target_network(batch.next_observations).max(dim=1, keepdim=True)[0]

            # (1 - terminated) removes the bootstrap for intrinsic terminals.
            target = batch.rewards + self.config.gamma * (1.0 - batch.terminated) * next_q

        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self._gradient_steps += 1
        self._sync_target_network()

        self._last_loss = float(loss.item())
        self._last_q_mean = float(q_selected.mean().item())
        return {
            "loss": self._last_loss,
            "q_mean": self._last_q_mean,
            "target_mean": float(target.mean().item()),
            "epsilon": self.policy.epsilon,
        }

    def _sync_target_network(self) -> None:
        """Hard-copy or Polyak-average the online weights into the target net."""
        tau = self.config.soft_update_tau
        if tau > 0.0:
            with torch.no_grad():
                for target_param, online_param in zip(
                    self.target_network.parameters(), self.q_network.parameters()
                ):
                    target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)
            return
        if self._gradient_steps % max(1, self.config.target_update_frequency) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # ---- bookkeeping --------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.policy.epsilon

    @property
    def gradient_steps(self) -> int:
        return self._gradient_steps

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def diagnostics(self) -> dict[str, float]:
        """Scalars logged every episode."""
        return {
            "epsilon": self.policy.epsilon,
            "replay_size": float(len(self.replay_buffer)),
            "gradient_steps": float(self._gradient_steps),
            "loss": float(self._last_loss) if self._last_loss is not None else float("nan"),
            "q_mean": float(self._last_q_mean) if self._last_q_mean is not None else float("nan"),
        }

    def train_mode(self) -> None:
        self.q_network.train()

    def eval_mode(self) -> None:
        self.q_network.eval()

    def state_dict(self, include_replay: bool = False) -> dict[str, Any]:
        """Checkpoint payload. Replay contents are excluded by default (large)."""
        state: dict[str, Any] = {
            "algorithm": self.algorithm,
            "observation_size": self.observation_size,
            "num_actions": self.num_actions,
            "config": self.config.as_dict(),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy": self.policy.state_dict(),
            "gradient_steps": self._gradient_steps,
            "env_steps": self._env_steps,
        }
        if include_replay:
            state["replay_buffer"] = self.replay_buffer.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore from a checkpoint produced by :meth:`state_dict`."""
        expected = (self.observation_size, self.num_actions)
        found = (int(state["observation_size"]), int(state["num_actions"]))
        if expected != found:
            raise ValueError(
                f"Checkpoint shape mismatch: agent expects obs/actions {expected}, "
                f"checkpoint holds {found}."
            )
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state.get("target_network", state["q_network"]))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "policy" in state:
            self.policy.load_state_dict(state["policy"])
        self._gradient_steps = int(state.get("gradient_steps", 0))
        self._env_steps = int(state.get("env_steps", 0))
        if "replay_buffer" in state:
            self.replay_buffer.load_state_dict(state["replay_buffer"])

    @classmethod
    def from_checkpoint(
        cls,
        state: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> "DQNAgent":
        """Rebuild an agent directly from a checkpoint payload."""
        agent = cls(
            observation_size=int(state["observation_size"]),
            num_actions=int(state["num_actions"]),
            config=DQNConfig.from_mapping(state.get("config")),
            device=device,
        )
        agent.load_state_dict(state)
        return agent
