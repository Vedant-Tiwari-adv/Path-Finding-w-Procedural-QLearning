"""Learning algorithms. All agents implement the :class:`Agent` interface."""

from drone_rl.agents.base import Agent
from drone_rl.agents.dqn import DQNAgent, DQNConfig
from drone_rl.agents.exploration import EpsilonGreedy, LinearEpsilonSchedule
from drone_rl.agents.goal_relabeling import GoalRelabeler, ObservationLayout
from drone_rl.agents.her_dqn import HERDQNAgent, HERDQNConfig
from drone_rl.agents.network import QNetwork
from drone_rl.agents.prioritized_replay import (
    NStepBatch,
    PrioritizedReplayBuffer,
    SumTree,
)
from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import (
    agent_from_checkpoint,
    available_algorithms,
    build_agent,
    register_agent,
)
from drone_rl.agents.replay_buffer import ReplayBuffer, Transition, TransitionBatch

__all__ = [
    "Agent",
    "DQNAgent",
    "DQNConfig",
    "EpsilonGreedy",
    "GoalRelabeler",
    "HERDQNAgent",
    "HERDQNConfig",
    "LinearEpsilonSchedule",
    "NStepBatch",
    "ObservationLayout",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "RandomAgent",
    "ReplayBuffer",
    "SumTree",
    "Transition",
    "TransitionBatch",
    "agent_from_checkpoint",
    "available_algorithms",
    "build_agent",
    "register_agent",
]
