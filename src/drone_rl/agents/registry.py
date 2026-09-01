"""Algorithm registry - the Phase-14 plug-in point.

Adding a proposed algorithm means writing a class that implements
:class:`~drone_rl.agents.base.Agent` and calling :func:`register_agent`. Every
script then reaches it through ``--config`` (``experiment.algorithm: my_method``)
with no other change, which is what keeps the baseline-vs-proposed comparison
mechanically fair: the environment, reward, observation space and evaluation
protocol are literally the same code paths.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from drone_rl.agents.base import Agent
from drone_rl.agents.dqn import DQNAgent, DQNConfig
from drone_rl.agents.her_dqn import HERDQNAgent, HERDQNConfig
from drone_rl.agents.random_agent import RandomAgent

__all__ = ["register_agent", "build_agent", "available_algorithms", "agent_from_checkpoint"]

#: Factory signature: ``(observation_size, num_actions, config, device, seed) -> Agent``
AgentFactory = Callable[..., Agent]

_REGISTRY: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Register an agent factory under ``name`` (case-insensitive)."""
    key = name.strip().lower()
    if key in _REGISTRY:
        raise KeyError(f"Algorithm {name!r} is already registered")
    _REGISTRY[key] = factory


def available_algorithms() -> list[str]:
    """Names of all registered algorithms."""
    return sorted(_REGISTRY)


def _build_dqn(
    observation_size: int,
    num_actions: int,
    config: Mapping[str, Any],
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> DQNAgent:
    return DQNAgent(
        observation_size=observation_size,
        num_actions=num_actions,
        config=DQNConfig.from_mapping(config.get("dqn")),
        device=device,
        seed=seed,
    )


def _build_her_dqn(
    observation_size: int,
    num_actions: int,
    config: Mapping[str, Any],
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> HERDQNAgent:
    # The whole config is forwarded as `relabel_config`: hindsight relabelling
    # needs the same city extents, target radius and reward weights the
    # environment was built with, or the reconstructed rewards would not match.
    return HERDQNAgent(
        observation_size=observation_size,
        num_actions=num_actions,
        config=HERDQNConfig.from_mapping(config.get("her_dqn")),
        relabel_config=config,
        device=device,
        seed=seed,
    )


def _build_random(
    observation_size: int,
    num_actions: int,
    config: Mapping[str, Any],
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> RandomAgent:
    del observation_size, config, device
    return RandomAgent(num_actions=num_actions, seed=seed)


register_agent("dqn", _build_dqn)
register_agent("her_dqn", _build_her_dqn)
register_agent("random", _build_random)


def build_agent(
    algorithm: str,
    observation_size: int,
    num_actions: int,
    config: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> Agent:
    """Instantiate the agent named ``algorithm``."""
    key = (algorithm or "dqn").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown algorithm {algorithm!r}; available: {available_algorithms()}")
    return _REGISTRY[key](
        observation_size=observation_size,
        num_actions=num_actions,
        config=dict(config or {}),
        device=device,
        seed=seed,
    )


def agent_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> Agent:
    """Rebuild an agent from a checkpoint's ``agent`` payload."""
    agent_state = checkpoint.get("agent", checkpoint)
    algorithm = str(agent_state.get("algorithm", "dqn")).lower()
    if algorithm == "dqn":
        return DQNAgent.from_checkpoint(agent_state, device=device)
    if algorithm == "her_dqn":
        return HERDQNAgent.from_checkpoint(agent_state, device=device)
    if algorithm == "random":
        agent = RandomAgent(num_actions=int(agent_state["num_actions"]))
        agent.load_state_dict(agent_state)
        return agent
    raise KeyError(
        f"Checkpoint declares algorithm {algorithm!r}, which has no loader. "
        f"Known: ['dqn', 'her_dqn', 'random']"
    )
