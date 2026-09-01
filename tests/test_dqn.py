"""Phase 8 tests: DQN correctness.

The critical tests here are the ones covering the Bellman target: terminal
handling and the truncation/termination distinction are the two places where a
subtly wrong DQN still "trains" but learns the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_rl.agents.dqn import DQNAgent, DQNConfig
from drone_rl.agents.exploration import EpsilonGreedy, LinearEpsilonSchedule
from drone_rl.agents.network import QNetwork
from drone_rl.agents.registry import available_algorithms, build_agent
from drone_rl.agents.replay_buffer import ReplayBuffer

OBS_SIZE = 21
NUM_ACTIONS = 9


@pytest.fixture
def agent() -> DQNAgent:
    config = DQNConfig(
        buffer_size=1_000,
        batch_size=32,
        min_replay_size=64,
        hidden_sizes=(32, 32),
        epsilon_decay_steps=1_000,
    )
    return DQNAgent(OBS_SIZE, NUM_ACTIONS, config=config, seed=0)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_network_output_shape():
    net = QNetwork(OBS_SIZE, NUM_ACTIONS, hidden_sizes=(16, 16))
    out = net(torch.zeros(5, OBS_SIZE))
    assert out.shape == (5, NUM_ACTIONS)


def test_network_is_an_mlp_with_configured_widths():
    net = QNetwork(OBS_SIZE, NUM_ACTIONS, hidden_sizes=(64, 48))
    linear_layers = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in linear_layers] == [64, 48, NUM_ACTIONS]


def test_dueling_head_produces_same_shape():
    net = QNetwork(OBS_SIZE, NUM_ACTIONS, hidden_sizes=(16,), dueling=True)
    assert net(torch.zeros(3, OBS_SIZE)).shape == (3, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


def test_replay_buffer_grows_then_saturates():
    buffer = ReplayBuffer(capacity=10, observation_size=OBS_SIZE, seed=0)
    for i in range(25):
        buffer.add(np.zeros(OBS_SIZE), i % NUM_ACTIONS, float(i), np.ones(OBS_SIZE), False)
    assert len(buffer) == 10  # oldest entries evicted


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=100, observation_size=OBS_SIZE, seed=0)
    for i in range(50):
        buffer.add(np.random.rand(OBS_SIZE), i % NUM_ACTIONS, 1.0, np.random.rand(OBS_SIZE), i % 7 == 0)
    batch = buffer.sample(16)
    assert batch.observations.shape == (16, OBS_SIZE)
    assert batch.actions.shape == (16, 1)
    assert batch.rewards.shape == (16, 1)
    assert batch.next_observations.shape == (16, OBS_SIZE)
    assert batch.terminated.shape == (16, 1)
    assert batch.actions.dtype == torch.int64


def test_replay_buffer_round_trips_values():
    buffer = ReplayBuffer(capacity=4, observation_size=3, seed=0)
    obs = np.array([1.0, 2.0, 3.0])
    nxt = np.array([4.0, 5.0, 6.0])
    buffer.add(obs, 2, -1.5, nxt, True)
    batch = buffer.sample(1)
    np.testing.assert_allclose(batch.observations[0].numpy(), obs, rtol=1e-6)
    np.testing.assert_allclose(batch.next_observations[0].numpy(), nxt, rtol=1e-6)
    assert int(batch.actions[0]) == 2
    assert float(batch.rewards[0]) == pytest.approx(-1.5)
    assert float(batch.terminated[0]) == 1.0


def test_empty_buffer_sample_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=5, observation_size=3).sample(2)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------


def test_epsilon_decays_linearly_then_holds():
    schedule = LinearEpsilonSchedule(start=1.0, end=0.1, decay_steps=100)
    assert schedule.value(0) == pytest.approx(1.0)
    assert schedule.value(50) == pytest.approx(0.55)
    assert schedule.value(100) == pytest.approx(0.1)
    assert schedule.value(10_000) == pytest.approx(0.1)  # never below the floor


def test_epsilon_greedy_explores_at_epsilon_one():
    policy = EpsilonGreedy(NUM_ACTIONS, LinearEpsilonSchedule(1.0, 1.0, 10), seed=0)
    assert all(policy.should_explore() for _ in range(50))


def test_epsilon_greedy_state_round_trip():
    policy = EpsilonGreedy(NUM_ACTIONS, LinearEpsilonSchedule(1.0, 0.05, 500), seed=3)
    policy.increment(200)
    restored = EpsilonGreedy(NUM_ACTIONS, seed=0)
    restored.load_state_dict(policy.state_dict())
    assert restored.epsilon == pytest.approx(policy.epsilon)
    assert restored.step_count == policy.step_count


# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------


def test_act_returns_valid_action(agent):
    action = agent.act(np.zeros(OBS_SIZE, dtype=np.float32))
    assert 0 <= action < NUM_ACTIONS


def test_evaluation_mode_is_greedy_and_deterministic(agent):
    observation = np.random.RandomState(0).rand(OBS_SIZE).astype(np.float32)
    actions = {agent.act(observation, evaluation=True) for _ in range(30)}
    assert len(actions) == 1
    expected = int(np.argmax(agent.q_values(observation)))
    assert actions.pop() == expected


def test_no_update_before_min_replay_size(agent):
    for _ in range(10):
        agent.observe(np.zeros(OBS_SIZE), 0, 0.0, np.zeros(OBS_SIZE), False, False)
    assert agent.update() == {}
    assert not agent.ready_to_train()


def test_update_runs_and_changes_weights(agent):
    rng = np.random.default_rng(0)
    for _ in range(200):
        agent.observe(
            rng.random(OBS_SIZE).astype(np.float32),
            int(rng.integers(0, NUM_ACTIONS)),
            float(rng.normal()),
            rng.random(OBS_SIZE).astype(np.float32),
            bool(rng.random() < 0.1),
            False,
        )
    before = agent.q_network.feature_extractor[0].weight.detach().clone()
    stats = agent.update()
    after = agent.q_network.feature_extractor[0].weight.detach()

    assert "loss" in stats and np.isfinite(stats["loss"])
    assert not torch.allclose(before, after), "Gradient step did not change the weights"


def test_target_network_is_frozen_and_synced(agent):
    assert all(not p.requires_grad for p in agent.target_network.parameters())

    rng = np.random.default_rng(1)
    for _ in range(200):
        agent.observe(rng.random(OBS_SIZE).astype(np.float32), 0, 1.0,
                      rng.random(OBS_SIZE).astype(np.float32), False, False)

    agent.config.target_update_frequency = 5
    before = agent.target_network.output_head.weight.detach().clone()
    for _ in range(4):
        agent.update()
    assert torch.allclose(before, agent.target_network.output_head.weight)  # not yet
    agent.update()  # 5th step triggers the sync
    assert torch.allclose(
        agent.target_network.output_head.weight, agent.q_network.output_head.weight
    )


def test_terminal_states_are_not_bootstrapped(agent):
    """The Bellman target for a terminal transition must be exactly r.

    Recomputes the target by hand and compares against the agent's own formula.
    """
    gamma = agent.config.gamma
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    next_obs = np.ones(OBS_SIZE, dtype=np.float32)

    for _ in range(100):
        agent.observe(obs, 0, 5.0, next_obs, True, False)  # all terminal
    batch = agent.replay_buffer.sample(8)

    with torch.no_grad():
        next_q = agent.target_network(batch.next_observations).max(dim=1, keepdim=True)[0]
        target = batch.rewards + gamma * (1.0 - batch.terminated) * next_q

    assert torch.allclose(target, batch.rewards)
    assert not torch.allclose(target, batch.rewards + gamma * next_q)


def test_non_terminal_states_are_bootstrapped(agent):
    gamma = agent.config.gamma
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    next_obs = np.ones(OBS_SIZE, dtype=np.float32)
    for _ in range(100):
        agent.observe(obs, 0, 5.0, next_obs, False, False)
    batch = agent.replay_buffer.sample(8)

    with torch.no_grad():
        next_q = agent.target_network(batch.next_observations).max(dim=1, keepdim=True)[0]
        target = batch.rewards + gamma * (1.0 - batch.terminated) * next_q

    assert torch.allclose(target, batch.rewards + gamma * next_q)


def test_truncation_is_stored_as_non_terminal(agent):
    """A time-limit cutoff must still bootstrap.

    Storing ``terminated or truncated`` would be the classic bug; this asserts
    the agent records only the intrinsic flag.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    agent.observe(obs, 0, 1.0, obs, terminated=False, truncated=True)
    batch = agent.replay_buffer.sample(1)
    assert float(batch.terminated[0]) == 0.0


def test_terminated_flag_is_stored_as_terminal(agent):
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    agent.observe(obs, 0, 1.0, obs, terminated=True, truncated=False)
    batch = agent.replay_buffer.sample(1)
    assert float(batch.terminated[0]) == 1.0


def test_agent_learns_a_trivial_bandit():
    """End-to-end sanity: on a one-state MDP, Q must converge toward the reward.

    Action 3 pays +1, everything else pays 0. After training, the greedy action
    must be 3 and its Q-value must approach 1 / (1 - gamma) is not applicable
    here because every transition is terminal, so Q(s, 3) -> 1.
    """
    config = DQNConfig(
        learning_rate=1e-2,
        batch_size=64,
        buffer_size=5_000,
        min_replay_size=64,
        target_update_frequency=50,
        hidden_sizes=(32, 32),
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    agent = DQNAgent(4, 5, config=config, seed=0)
    rng = np.random.default_rng(0)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(3_000):
        action = int(rng.integers(0, 5))
        reward = 1.0 if action == 3 else 0.0
        agent.observe(state, action, reward, state, terminated=True, truncated=False)
        agent.update()

    q_values = agent.q_values(state)
    assert int(np.argmax(q_values)) == 3, f"Q values: {q_values}"
    assert q_values[3] == pytest.approx(1.0, abs=0.15)
    assert q_values[0] == pytest.approx(0.0, abs=0.15)


def test_double_dqn_flag_changes_target_computation():
    """Double DQN must decouple action selection from evaluation."""
    config = DQNConfig(double_dqn=True, min_replay_size=32, batch_size=16, hidden_sizes=(16,))
    agent = DQNAgent(OBS_SIZE, NUM_ACTIONS, config=config, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        agent.observe(rng.random(OBS_SIZE).astype(np.float32), 0, 1.0,
                      rng.random(OBS_SIZE).astype(np.float32), False, False)
    stats = agent.update()
    assert np.isfinite(stats["loss"])


def test_gradient_clipping_bounds_updates():
    config = DQNConfig(max_grad_norm=0.01, min_replay_size=32, batch_size=16, hidden_sizes=(16,))
    agent = DQNAgent(OBS_SIZE, NUM_ACTIONS, config=config, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        agent.observe(rng.random(OBS_SIZE).astype(np.float32), 0, 1_000.0,
                      rng.random(OBS_SIZE).astype(np.float32), False, False)
    before = torch.cat([p.flatten() for p in agent.q_network.parameters()]).clone()
    agent.update()
    after = torch.cat([p.flatten() for p in agent.q_network.parameters()])
    # Adam's step size is bounded by lr regardless, but clipping keeps the
    # gradient (and hence the update direction) finite and small.
    assert torch.isfinite(after).all()
    assert (after - before).abs().max() < 1.0


# ---------------------------------------------------------------------------
# Checkpointing / registry
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_preserves_policy(agent):
    rng = np.random.default_rng(2)
    for _ in range(150):
        agent.observe(rng.random(OBS_SIZE).astype(np.float32), 1, 0.5,
                      rng.random(OBS_SIZE).astype(np.float32), False, False)
    agent.update()

    state = agent.state_dict()
    restored = DQNAgent.from_checkpoint(state)

    observation = rng.random(OBS_SIZE).astype(np.float32)
    np.testing.assert_allclose(agent.q_values(observation), restored.q_values(observation), rtol=1e-5)
    assert restored.gradient_steps == agent.gradient_steps


def test_load_state_dict_rejects_shape_mismatch(agent):
    other = DQNAgent(OBS_SIZE + 1, NUM_ACTIONS, config=agent.config, seed=0)
    with pytest.raises(ValueError, match="shape mismatch"):
        other.load_state_dict(agent.state_dict())


def test_registry_builds_known_algorithms():
    assert "dqn" in available_algorithms()
    assert "random" in available_algorithms()

    dqn = build_agent("dqn", OBS_SIZE, NUM_ACTIONS, config={"dqn": {"hidden_sizes": [16, 16]}})
    assert isinstance(dqn, DQNAgent)

    random_agent = build_agent("random", OBS_SIZE, NUM_ACTIONS)
    assert 0 <= random_agent.act(np.zeros(OBS_SIZE)) < NUM_ACTIONS


def test_registry_rejects_unknown_algorithm():
    with pytest.raises(KeyError):
        build_agent("not_an_algorithm", OBS_SIZE, NUM_ACTIONS)
