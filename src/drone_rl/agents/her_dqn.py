"""Hindsight-relabelled DQN - the proposed algorithm (Phase 14).

Implements :class:`~drone_rl.agents.base.Agent`, so the environment, reward,
observation space, training loop and evaluation protocol are shared verbatim
with the DQN baseline. Everything below is a change to *how the agent learns
from transitions it has already collected*, never to what it is allowed to see.

Motivation, from the measured baseline run
------------------------------------------
The 4000-episode DQN baseline (``outputs/logs/dqn_gpu_run``) exhibits two
distinct pathologies, and this agent targets each with a specific mechanism:

1. **Value divergence.** Mean predicted Q rose monotonically from 2.3 to 63.6
   while realised episode returns averaged -71, and the Huber loss *rose*
   throughout training (0.16 to 0.71) instead of falling. That is the classic
   max-operator overestimation of plain DQN, compounded over the ~200-step
   bootstrap chain implied by ``gamma=0.995``. Validation success peaked at
   14.8% near episode 1500 and then decayed to ~5% as the value function drifted
   further from the returns it was supposed to predict.

   -> ``double_dqn`` decouples action selection from action evaluation, which
   removes the systematic upward bias; ``layer_norm`` bounds the feature scale;
   ``soft_update_tau`` replaces the 1000-step hard target copy with Polyak
   averaging so the regression target moves smoothly.

2. **Reward too sparse to propagate.** Only 5-15% of episodes ever earn the
   ``+100`` arrival bonus, and with 1-step returns that bonus must survive ~200
   sequential Bellman updates to reach the actions that set the approach up. A
   uniform batch of 128 therefore contains almost no signal about succeeding.

   -> ``n_step`` returns shorten the credit path by a factor of n; prioritised
   replay concentrates updates on high-TD-error transitions, which are
   disproportionately the rewarding ones; and hindsight relabelling manufactures
   successful goal-reaching data out of *every* episode, including the failures.

Hindsight relabelling
---------------------
This task is goal-conditioned in the strict sense the technique requires: the
goal enters only through four observation dimensions and two reward terms, and
:class:`~drone_rl.agents.goal_relabeling.GoalRelabeler` recomputes both exactly
(verified to float32 precision against the live encoder). So a trajectory that
missed its assigned target is still a *correct, on-distribution* demonstration
of flying to wherever it actually ended up. With ``her_k=4`` the agent extracts
four such relabelled windows per real one, giving it dense arrival signal from
episode one instead of waiting for the policy to stumble into a success.

The ``future`` strategy is used: the substituted goal for a transition at time
``t`` is a position the drone actually occupied at some time ``>= t`` in the
same episode, so the relabelled goal is reachable by construction from that
state, under the dynamics the agent actually experienced.

Every mechanism is individually toggleable from YAML so the contribution can be
ablated - which the fairness contract in ``configs/experiment.yaml`` expects of
any claimed improvement.
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
from drone_rl.agents.goal_relabeling import GoalRelabeler
from drone_rl.agents.network import QNetwork
from drone_rl.agents.prioritized_replay import PrioritizedReplayBuffer
from drone_rl.utils.logging import get_logger

__all__ = ["HERDQNConfig", "HERDQNAgent"]

_LOG = get_logger(__name__)


@dataclass
class HERDQNConfig:
    """Hyper-parameters for the proposed agent (the ``her_dqn:`` YAML block)."""

    # --- optimisation ---
    learning_rate: float = 3e-4
    gamma: float = 0.995
    batch_size: int = 128
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0

    # --- replay ---
    buffer_size: int = 400_000
    min_replay_size: int = 5_000

    # --- target network ---
    #: Polyak coefficient. Preferred over the baseline's hard copy: a target
    #: that moves a little every step cannot present the sudden jump in
    #: regression target that a 1000-step hard sync does.
    soft_update_tau: float = 0.005
    target_update_frequency: int = 1_000  # only used when soft_update_tau == 0

    # --- exploration ---
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 400_000

    # --- architecture ---
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    double_dqn: bool = True
    dueling: bool = True
    layer_norm: bool = True

    # --- n-step returns ---
    n_step: int = 3

    # --- prioritised replay ---
    prioritized: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 800_000
    per_epsilon: float = 1e-3

    # --- hindsight experience replay ---
    her_enabled: bool = True
    #: Relabelled windows generated per real transition. 4 is the value from the
    #: original HER paper and gives a 1:4 real-to-relabelled ratio.
    her_k: int = 4

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "HERDQNConfig":
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


class HERDQNAgent(Agent):
    """Double/dueling DQN with n-step returns, prioritised replay and HER.

    Parameters
    ----------
    observation_size:
        Length of the flat observation vector.
    num_actions:
        Size of the discrete action space.
    config:
        Hyper-parameters.
    relabel_config:
        The ``city`` / ``drone`` / ``environment`` / ``reward`` blocks, used to
        build the :class:`GoalRelabeler`. Stored in checkpoints so a loaded
        agent can relabel identically.
    device:
        Torch device for the networks and sampled batches.
    seed:
        Seeds the replay sampler, the hindsight goal sampler and exploration.

    Notes
    -----
    Transitions are buffered per episode and only flushed to replay in
    :meth:`on_episode_end`, because hindsight goals must be drawn from states
    the drone reached *later* in the same episode. The replay buffer therefore
    lags collection by at most one episode, which the trainer's
    ``learning_starts`` warm-up already absorbs.
    """

    algorithm = "her_dqn"

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: HERDQNConfig | None = None,
        relabel_config: Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.config = config or HERDQNConfig()
        self.device = torch.device(device)
        self.observation_size = int(observation_size)
        self.num_actions = int(num_actions)

        self._relabel_config = {
            key: dict(value)
            for key, value in (relabel_config or {}).items()
            if key in {"city", "drone", "environment", "reward"}
        }
        self.relabeler = GoalRelabeler(self._relabel_config)

        self.q_network = QNetwork(
            observation_size=self.observation_size,
            num_actions=self.num_actions,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            dueling=self.config.dueling,
            layer_norm=self.config.layer_norm,
        ).to(self.device)

        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )
        # Reduction is 'none' so per-sample errors can be weighted by the
        # importance-sampling weights before averaging.
        self.loss_fn = nn.SmoothL1Loss(beta=self.config.huber_delta, reduction="none")

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_size,
            observation_size=self.observation_size,
            alpha=self.config.per_alpha if self.config.prioritized else 0.0,
            epsilon=self.config.per_epsilon,
            device=self.device,
            seed=seed,
        )
        self.policy = EpsilonGreedy.from_config(
            num_actions=self.num_actions,
            config=self.config.as_dict(),
            seed=None if seed is None else seed + 1,
        )
        self._goal_rng = np.random.default_rng(None if seed is None else seed + 2)

        # Transitions of the episode currently being collected.
        self._episode: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

        self._gradient_steps = 0
        self._env_steps = 0
        self._last_loss: float | None = None
        self._last_q_mean: float | None = None
        self._last_relabelled = 0

    # ---- acting -------------------------------------------------------------

    def act(self, observation: np.ndarray, evaluation: bool = False) -> int:
        """Epsilon-greedy during training, greedy during evaluation."""
        if not evaluation and self.policy.should_explore():
            return self.policy.random_action()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                np.asarray(observation, dtype=np.float32), device=self.device
            ).unsqueeze(0)
            return int(self.q_network(obs_tensor).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Q-values for one observation, for debugging and GUI overlays."""
        obs_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        return self.q_network(obs_tensor).squeeze(0).cpu().numpy()

    # ---- collection ---------------------------------------------------------

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Buffer one transition until the episode completes.

        ``truncated`` is deliberately not stored: a time-limit cutoff is not
        part of the MDP, so its successor state must still be bootstrapped from.
        """
        self._episode.append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(terminated),
            )
        )
        self._env_steps += 1
        self.policy.increment()

    def on_episode_end(self, episode_index: int, summary: Mapping[str, Any]) -> None:
        """Convert the finished episode into n-step windows and store them."""
        if not self._episode:
            return
        transitions = self._episode
        self._episode = []

        observations = np.stack([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float64)
        next_observations = np.stack([t[3] for t in transitions])
        terminated = np.array([t[4] for t in transitions], dtype=bool)

        # Only the final transition can be a collision, and the summary reports
        # it exactly - so the reward reconstruction never has to guess.
        collided = np.zeros(len(transitions), dtype=bool)
        if bool(summary.get("collision", False)):
            collided[-1] = True

        self._store_real_windows(
            observations, actions, rewards, next_observations, terminated
        )
        if self.config.her_enabled and self.config.her_k > 0:
            self._store_hindsight_windows(
                observations, actions, rewards, next_observations, terminated, collided
            )

    def _store_real_windows(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        """Accumulate and store n-step windows for the trajectory as flown."""
        length = len(rewards)
        n = max(1, int(self.config.n_step))
        gamma = float(self.config.gamma)

        offsets = np.arange(n)
        starts = np.arange(length)
        indices = np.minimum(starts[:, None] + offsets[None, :], length - 1)
        valid = (starts[:, None] + offsets[None, :]) < length
        discounts = gamma ** offsets

        returns = (rewards[indices] * discounts[None, :] * valid).sum(axis=1)
        window_lengths = np.minimum(n, length - starts)
        last = starts + window_lengths - 1
        bootstrap_discounts = np.where(terminated[last], 0.0, gamma**window_lengths)

        self.replay_buffer.add_batch(
            observations=observations,
            actions=actions,
            returns=returns,
            bootstrap_observations=next_observations[last],
            bootstrap_discounts=bootstrap_discounts,
        )

    def _store_hindsight_windows(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminated: np.ndarray,
        collided: np.ndarray,
    ) -> None:
        """Relabel the episode against achieved goals and store the result.

        For every transition ``t`` this draws ``her_k`` future indices
        ``f >= t`` and substitutes the position the drone occupied at ``f`` as
        the goal. Each relabelled window is then re-accumulated from scratch,
        because substituting the goal changes both the per-step progress reward
        and *where the window terminates*: a relabelled episode ends the moment
        the substituted goal is reached, which is usually earlier than the real
        episode ended.
        """
        length = len(rewards)
        n = max(1, int(self.config.n_step))
        k = int(self.config.her_k)
        gamma = float(self.config.gamma)

        # Achieved goal at each step = where the drone actually ended up.
        achieved = self.relabeler.positions(next_observations)

        starts = np.repeat(np.arange(length), k)
        spans = length - starts
        futures = starts + (self._goal_rng.random(len(starts)) * spans).astype(np.int64)
        futures = np.minimum(futures, length - 1)
        goals = achieved[futures]

        offsets = np.arange(n)
        step_indices = np.minimum(starts[:, None] + offsets[None, :], length - 1)
        valid = (starts[:, None] + offsets[None, :]) < length

        flat = step_indices.ravel()
        goals_flat = np.repeat(goals, n, axis=0)
        new_rewards, new_terminated = self.relabeler.relabel_rewards(
            observations=observations[flat],
            next_observations=next_observations[flat],
            rewards=rewards[flat],
            terminated=terminated[flat],
            new_goals=goals_flat,
            collided=collided[flat],
        )
        new_rewards = new_rewards.reshape(len(starts), n)
        new_terminated = new_terminated.reshape(len(starts), n)

        # Truncate each window at the first step that terminates under the new
        # goal; steps past the end of the episode are not eligible.
        stops = new_terminated & valid
        has_stop = stops.any(axis=1)
        first_stop = stops.argmax(axis=1)
        available = np.minimum(n, length - starts)
        window_lengths = np.where(has_stop, first_stop + 1, available)

        mask = offsets[None, :] < window_lengths[:, None]
        discounts = gamma**offsets
        returns = (new_rewards * discounts[None, :] * mask).sum(axis=1)

        last_offset = window_lengths - 1
        rows = np.arange(len(starts))
        terminal_at_last = new_terminated[rows, last_offset]
        bootstrap_discounts = np.where(terminal_at_last, 0.0, gamma**window_lengths)

        last_indices = starts + last_offset
        start_observations = self.relabeler.relabel_observations(
            observations[starts], goals
        )
        bootstrap_observations = self.relabeler.relabel_observations(
            next_observations[last_indices], goals
        )

        self.replay_buffer.add_batch(
            observations=start_observations,
            actions=actions[starts],
            returns=returns,
            bootstrap_observations=bootstrap_observations,
            bootstrap_discounts=bootstrap_discounts,
        )
        self._last_relabelled = len(starts)

    # ---- learning -----------------------------------------------------------

    def ready_to_train(self) -> bool:
        """True once replay holds enough windows to learn from."""
        return len(self.replay_buffer) >= max(
            self.config.min_replay_size, self.config.batch_size
        )

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent, annealed towards its end value."""
        cfg = self.config
        if not cfg.prioritized:
            return 1.0
        steps = max(1, int(cfg.per_beta_steps))
        fraction = min(1.0, self._env_steps / steps)
        return float(cfg.per_beta_start + fraction * (cfg.per_beta_end - cfg.per_beta_start))

    def update(self) -> dict[str, float]:
        """One gradient step on the prioritised, importance-weighted n-step error."""
        if not self.ready_to_train():
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size, beta=self.beta)

        q_selected = self.q_network(batch.observations).gather(1, batch.actions)

        with torch.no_grad():
            if self.config.double_dqn:
                # Online net picks the action, target net scores it. This is the
                # correction for the overestimation that destabilised the baseline.
                next_actions = self.q_network(batch.bootstrap_observations).argmax(
                    dim=1, keepdim=True
                )
                next_q = self.target_network(batch.bootstrap_observations).gather(
                    1, next_actions
                )
            else:
                next_q = self.target_network(batch.bootstrap_observations).max(
                    dim=1, keepdim=True
                )[0]

            # bootstrap_discounts is gamma**k, and exactly zero when the window
            # ended in an intrinsic terminal - so no (1 - done) factor is needed.
            target = batch.returns + batch.bootstrap_discounts * next_q

        elementwise = self.loss_fn(q_selected, target)
        loss = (elementwise * batch.weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.config.prioritized:
            td_errors = (target - q_selected).detach().abs().squeeze(1).cpu().numpy()
            self.replay_buffer.update_priorities(batch.indices, td_errors)

        self._gradient_steps += 1
        self._sync_target_network()

        self._last_loss = float(loss.item())
        self._last_q_mean = float(q_selected.mean().item())
        return {
            "loss": self._last_loss,
            "q_mean": self._last_q_mean,
            "target_mean": float(target.mean().item()),
            "epsilon": self.policy.epsilon,
            "beta": self.beta,
        }

    def _sync_target_network(self) -> None:
        """Polyak-average (default) or hard-copy the online weights."""
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
            "beta": self.beta,
            "replay_size": float(len(self.replay_buffer)),
            "gradient_steps": float(self._gradient_steps),
            "relabelled_last_episode": float(self._last_relabelled),
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
            "relabel_config": self._relabel_config,
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
    ) -> "HERDQNAgent":
        """Rebuild an agent directly from a checkpoint payload."""
        agent = cls(
            observation_size=int(state["observation_size"]),
            num_actions=int(state["num_actions"]),
            config=HERDQNConfig.from_mapping(state.get("config")),
            relabel_config=state.get("relabel_config"),
            device=device,
        )
        agent.load_state_dict(state)
        return agent
