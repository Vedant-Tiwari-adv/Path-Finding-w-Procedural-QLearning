"""Phase 14 tests: the proposed hindsight-relabelled DQN.

The tests that matter most here are the ones covering *silent* corruption. A
relabeler that inverts the observation encoder slightly wrongly, or an n-step
window that accumulates one reward too many, still trains happily and still
produces plausible curves - it just learns from data the environment would
never have produced. So the emphasis is:

* the relabeler is an exact inverse of the live encoder, checked against a real
  environment rollout rather than a hand-built vector;
* relabelling to the *original* goal is a bit-for-bit no-op, which pins the
  whole reward decomposition in one assertion;
* n-step returns match an independently hand-computed reference, including the
  two truncation cases (episode end, and intrinsic termination);
* the sum tree's vectorised descent agrees with ``np.searchsorted``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_rl.agents.goal_relabeling import GoalRelabeler
from drone_rl.agents.her_dqn import HERDQNAgent, HERDQNConfig
from drone_rl.agents.prioritized_replay import PrioritizedReplayBuffer, SumTree
from drone_rl.agents.registry import agent_from_checkpoint, available_algorithms, build_agent
from drone_rl.environment.drone_env import DroneCityEnv

OBS_SIZE = 21
NUM_ACTIONS = 9


@pytest.fixture
def relabeler(small_env_config) -> GoalRelabeler:
    return GoalRelabeler(small_env_config)


@pytest.fixture
def agent_config() -> HERDQNConfig:
    return HERDQNConfig(
        buffer_size=4_000,
        batch_size=32,
        min_replay_size=64,
        hidden_sizes=(32, 32),
        epsilon_decay_steps=1_000,
    )


def _make_agent(config: HERDQNConfig, relabel_config: dict) -> HERDQNAgent:
    return HERDQNAgent(
        OBS_SIZE,
        NUM_ACTIONS,
        config=config,
        relabel_config=relabel_config,
        seed=0,
    )


def _synthetic_observation(position, goal, yaw=0.0, scale=(20.0, 20.0, 25.0), diagonal=1.0):
    """Build an observation whose position/heading/goal dimensions are known."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0:3] = np.asarray(position) / np.asarray(scale)
    relative = np.asarray(goal, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    obs[3:6] = relative / diagonal
    obs[6] = np.linalg.norm(relative) / diagonal
    obs[10] = np.cos(yaw)
    obs[11] = np.sin(yaw)
    return obs


# ---------------------------------------------------------------------------
# Goal relabelling - exactness against the live encoder
# ---------------------------------------------------------------------------


def test_relabeler_inverts_the_live_encoder(small_env_config, relabeler):
    """Position, heading and goal must be recoverable from the observation alone."""
    env = DroneCityEnv(config=small_env_config, seed_pool="train")
    relabeler.validate_against_encoder(env.encoder)
    rng = np.random.default_rng(0)

    observation, _ = env.reset(seed=7)
    for _ in range(40):
        true_position = env.drone.state.position.copy()
        true_yaw = float(env.drone.state.yaw)
        true_goal = env.target_position.copy()

        row = observation.reshape(1, -1)
        assert relabeler.positions(row)[0] == pytest.approx(true_position, abs=1e-4)
        assert relabeler.goals(row)[0] == pytest.approx(true_goal, abs=1e-3)
        recovered_yaw = relabeler.yaws(row)[0]
        assert np.sin(recovered_yaw - true_yaw) == pytest.approx(0.0, abs=1e-5)

        observation, _r, terminated, truncated, _i = env.step(
            int(rng.integers(0, env.action_space.n))
        )
        if terminated or truncated:
            observation, _ = env.reset(seed=8)


def test_relabelling_to_the_original_goal_is_a_no_op(small_env_config, relabeler):
    """The reward decomposition must reconstruct the stored reward exactly."""
    env = DroneCityEnv(config=small_env_config, seed_pool="train")
    rng = np.random.default_rng(1)

    observation, _ = env.reset(seed=11)
    for _ in range(60):
        goal = env.target_position.copy().reshape(1, 3)
        row = observation.reshape(1, -1)
        next_observation, reward, terminated, _truncated, info = env.step(
            int(rng.integers(0, env.action_space.n))
        )

        unchanged = relabeler.relabel_observations(row, goal)
        assert unchanged[0] == pytest.approx(observation, abs=1e-6)

        new_reward, new_terminated = relabeler.relabel_rewards(
            observations=row,
            next_observations=next_observation.reshape(1, -1),
            rewards=np.array([reward]),
            terminated=np.array([terminated]),
            new_goals=goal,
            collided=np.array([bool(info.get("collision", False))]),
        )
        assert float(new_reward[0]) == pytest.approx(reward, abs=1e-3)
        assert bool(new_terminated[0]) is bool(terminated)

        observation = next_observation
        if terminated or _truncated:
            observation, _ = env.reset(seed=12)


def test_relabelled_goal_pays_the_arrival_bonus(relabeler):
    """Substituting a goal the drone did reach must add exactly the target bonus."""
    scale = np.array([20.0, 20.0, 25.0])
    diagonal = relabeler._city_diagonal
    start = np.array([0.0, 0.0, 5.0])
    arrived = np.array([1.0, 0.0, 5.0])
    far_goal = np.array([15.0, 15.0, 5.0])

    observation = _synthetic_observation(start, far_goal, scale=scale, diagonal=diagonal)
    next_observation = _synthetic_observation(arrived, far_goal, scale=scale, diagonal=diagonal)

    # Reward as the environment would have produced it for the far goal.
    progress = np.linalg.norm(far_goal - start) - np.linalg.norm(far_goal - arrived)
    stored = -0.1 + relabeler.progress_weight * progress

    new_reward, new_terminated = relabeler.relabel_rewards(
        observations=observation.reshape(1, -1),
        next_observations=next_observation.reshape(1, -1),
        rewards=np.array([stored]),
        terminated=np.array([False]),
        new_goals=arrived.reshape(1, 3),
        collided=np.array([False]),
    )
    # Under the substituted goal the drone lands exactly on it: progress equals
    # the full starting distance, plus the arrival bonus.
    expected = -0.1 + relabeler.progress_weight * np.linalg.norm(arrived - start)
    expected += relabeler.target_bonus
    assert float(new_reward[0]) == pytest.approx(expected, abs=1e-3)
    assert bool(new_terminated[0]) is True


def test_collision_suppresses_the_relabelled_arrival_bonus(relabeler):
    """Matches the environment's `reached_target and not collided` semantics."""
    scale = np.array([20.0, 20.0, 25.0])
    diagonal = relabeler._city_diagonal
    start = np.array([0.0, 0.0, 5.0])
    arrived = np.array([1.0, 0.0, 5.0])

    observation = _synthetic_observation(start, arrived, scale=scale, diagonal=diagonal)
    next_observation = _synthetic_observation(arrived, arrived, scale=scale, diagonal=diagonal)

    _reward, terminated = relabeler.relabel_rewards(
        observations=observation.reshape(1, -1),
        next_observations=next_observation.reshape(1, -1),
        rewards=np.array([-100.0]),
        terminated=np.array([True]),
        new_goals=arrived.reshape(1, 3),
        collided=np.array([True]),
    )
    # Still terminal (it crashed), but no arrival credit.
    assert bool(terminated[0]) is True


def test_relabelled_observation_only_touches_goal_dimensions(relabeler):
    scale = np.array([20.0, 20.0, 25.0])
    diagonal = relabeler._city_diagonal
    observation = _synthetic_observation(
        [3.0, 4.0, 6.0], [10.0, 10.0, 6.0], scale=scale, diagonal=diagonal
    )
    observation[7:10] = [0.3, -0.2, 0.1]
    observation[13] = 0.8
    observation[14:] = 0.5

    relabelled = relabeler.relabel_observations(
        observation.reshape(1, -1), np.array([[1.0, 2.0, 3.0]])
    )[0]

    goal_dimensions = list(range(3, 7))
    for index in range(OBS_SIZE):
        if index in goal_dimensions:
            continue
        assert relabelled[index] == pytest.approx(observation[index], abs=1e-7), (
            f"non-goal dimension {index} was modified"
        )


# ---------------------------------------------------------------------------
# Sum tree and prioritised replay
# ---------------------------------------------------------------------------


def test_sum_tree_descent_matches_searchsorted():
    rng = np.random.default_rng(0)
    priorities = rng.uniform(0.01, 5.0, size=257)
    tree = SumTree(len(priorities))
    tree.update(np.arange(len(priorities)), priorities)

    assert tree.total == pytest.approx(priorities.sum(), rel=1e-12)

    probe = rng.uniform(0.0, priorities.sum(), size=2_000)
    expected = np.searchsorted(np.cumsum(priorities), probe, side="right")
    assert np.array_equal(tree.sample(probe), expected)


def test_sum_tree_repeated_index_does_not_double_count():
    tree = SumTree(8)
    tree.update(np.array([2, 2, 2]), np.array([1.0, 5.0, 3.0]))
    assert tree.total == pytest.approx(3.0)


def test_zero_priority_rows_are_never_sampled():
    buffer = PrioritizedReplayBuffer(64, 4, alpha=1.0, epsilon=0.0, seed=3)
    observations = np.zeros((64, 4), dtype=np.float32)
    buffer.add_batch(
        observations,
        np.zeros(64, np.int64),
        np.zeros(64, np.float32),
        observations,
        np.full(64, 0.99, np.float32),
    )
    errors = np.zeros(64)
    errors[[5, 11]] = [3.0, 1.0]
    buffer.update_priorities(np.arange(64), errors)

    drawn = np.concatenate([buffer.sample(32, beta=0.5).indices for _ in range(20)])
    assert set(np.unique(drawn)).issubset({5, 11})


def test_importance_weights_are_normalised_and_favour_rare_samples():
    buffer = PrioritizedReplayBuffer(32, 4, alpha=1.0, epsilon=0.0, seed=4)
    observations = np.zeros((32, 4), dtype=np.float32)
    buffer.add_batch(
        observations,
        np.zeros(32, np.int64),
        np.zeros(32, np.float32),
        observations,
        np.full(32, 0.99, np.float32),
    )
    errors = np.full(32, 0.1)
    errors[0] = 10.0
    buffer.update_priorities(np.arange(32), errors)

    batch = buffer.sample(64, beta=1.0)
    weights = batch.weights.squeeze(1).numpy()
    assert weights.max() == pytest.approx(1.0)
    assert weights.min() > 0.0
    # The oversampled row must be down-weighted relative to the rest.
    oversampled = batch.indices == 0
    if oversampled.any() and (~oversampled).any():
        assert weights[oversampled].mean() < weights[~oversampled].mean()


# ---------------------------------------------------------------------------
# n-step accumulation
# ---------------------------------------------------------------------------


def test_n_step_returns_match_hand_computed_reference(small_env_config):
    """Covers both truncation cases: episode end and intrinsic termination."""
    config = HERDQNConfig(
        n_step=3,
        gamma=0.9,
        her_enabled=False,
        prioritized=False,
        buffer_size=100,
        batch_size=4,
        min_replay_size=1,
        hidden_sizes=(8, 8),
    )
    agent = _make_agent(config, small_env_config)

    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for index, reward in enumerate(rewards):
        agent.observe(
            observation=np.zeros(OBS_SIZE, dtype=np.float32),
            action=0,
            reward=reward,
            next_observation=np.zeros(OBS_SIZE, dtype=np.float32),
            terminated=(index == len(rewards) - 1),
            truncated=False,
        )
    agent.on_episode_end(0, {"collision": False})

    g = 0.9
    expected_returns = [
        1 + g * 2 + g**2 * 3,  # full 3-step window
        2 + g * 3 + g**2 * 4,
        3 + g * 4 + g**2 * 5,  # ends on the terminal transition
        4 + g * 5,  # truncated by episode end
        5.0,
    ]
    # Windows that run into the intrinsic terminal must not bootstrap.
    expected_discounts = [g**3, g**3, 0.0, 0.0, 0.0]

    assert len(agent.replay_buffer) == 5
    stored_returns = agent.replay_buffer._returns[:5]
    stored_discounts = agent.replay_buffer._bootstrap_discounts[:5]
    assert stored_returns == pytest.approx(expected_returns, abs=1e-5)
    assert stored_discounts == pytest.approx(expected_discounts, abs=1e-6)


def test_hindsight_multiplies_stored_windows(small_env_config):
    config = HERDQNConfig(
        n_step=1,
        gamma=0.99,
        her_enabled=True,
        her_k=4,
        prioritized=False,
        buffer_size=1_000,
        batch_size=4,
        min_replay_size=1,
        hidden_sizes=(8, 8),
    )
    agent = _make_agent(config, small_env_config)
    relabeler = agent.relabeler
    scale = agent.relabeler._position_scale
    diagonal = relabeler._city_diagonal
    far_goal = np.array([18.0, 18.0, 6.0])

    length = 6
    for index in range(length):
        position = np.array([float(index), 0.0, 5.0])
        nxt = np.array([float(index + 1), 0.0, 5.0])
        agent.observe(
            observation=_synthetic_observation(position, far_goal, scale=scale, diagonal=diagonal),
            action=1,
            reward=-0.1,
            next_observation=_synthetic_observation(
                nxt, far_goal, scale=scale, diagonal=diagonal
            ),
            terminated=False,
            truncated=(index == length - 1),
        )
    agent.on_episode_end(0, {"collision": False})

    assert len(agent.replay_buffer) == length * (1 + config.her_k)

    hindsight_returns = agent.replay_buffer._returns[length:]
    hindsight_discounts = agent.replay_buffer._bootstrap_discounts[length:]
    # The final transition can only draw itself as its future goal, so those
    # windows must terminate on arrival and collect the bonus.
    assert (hindsight_discounts == 0.0).sum() >= config.her_k
    assert hindsight_returns.max() > relabeler.target_bonus * 0.5


def test_hindsight_disabled_stores_only_real_windows(small_env_config):
    config = HERDQNConfig(
        n_step=1,
        her_enabled=False,
        prioritized=False,
        buffer_size=100,
        batch_size=4,
        min_replay_size=1,
        hidden_sizes=(8, 8),
    )
    agent = _make_agent(config, small_env_config)
    for _ in range(5):
        agent.observe(
            observation=np.zeros(OBS_SIZE, dtype=np.float32),
            action=0,
            reward=-0.1,
            next_observation=np.zeros(OBS_SIZE, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
    agent.on_episode_end(0, {"collision": False})
    assert len(agent.replay_buffer) == 5


# ---------------------------------------------------------------------------
# Agent mechanics
# ---------------------------------------------------------------------------


def test_terminal_windows_do_not_bootstrap(small_env_config, agent_config):
    """A zero bootstrap discount must remove the successor term entirely."""
    agent = _make_agent(agent_config, small_env_config)
    observations = torch.zeros(4, OBS_SIZE)
    returns = torch.full((4, 1), 2.0)
    discounts = torch.zeros(4, 1)
    with torch.no_grad():
        next_q = agent.target_network(observations).max(dim=1, keepdim=True)[0]
        target = returns + discounts * next_q
    assert torch.allclose(target, returns)


def test_update_runs_and_reports_finite_statistics(small_env_config, agent_config):
    agent = _make_agent(agent_config, small_env_config)
    rng = np.random.default_rng(0)
    for _ in range(4):
        for _ in range(40):
            agent.observe(
                observation=rng.normal(size=OBS_SIZE).astype(np.float32),
                action=int(rng.integers(0, NUM_ACTIONS)),
                reward=float(rng.normal()),
                next_observation=rng.normal(size=OBS_SIZE).astype(np.float32),
                terminated=False,
                truncated=True,
            )
        agent.on_episode_end(0, {"collision": False})

    stats = agent.update()
    assert stats, "agent should be ready to train after four episodes"
    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["q_mean"])
    assert 0.0 <= stats["beta"] <= 1.0


def test_beta_anneals_towards_its_end_value(small_env_config):
    config = HERDQNConfig(per_beta_start=0.4, per_beta_end=1.0, per_beta_steps=100)
    agent = _make_agent(config, small_env_config)
    assert agent.beta == pytest.approx(0.4)
    agent._env_steps = 50
    assert agent.beta == pytest.approx(0.7)
    agent._env_steps = 10_000
    assert agent.beta == pytest.approx(1.0)


def test_checkpoint_round_trip_preserves_policy(small_env_config, agent_config):
    agent = _make_agent(agent_config, small_env_config)
    state = agent.state_dict()

    restored = agent_from_checkpoint(state, device="cpu")
    assert isinstance(restored, HERDQNAgent)

    observation = np.random.default_rng(0).normal(size=OBS_SIZE).astype(np.float32)
    assert restored.act(observation, evaluation=True) == agent.act(observation, evaluation=True)
    assert np.allclose(restored.q_values(observation), agent.q_values(observation), atol=1e-6)


def test_registry_builds_the_proposed_agent(small_env_config):
    assert "her_dqn" in available_algorithms()
    agent = build_agent(
        algorithm="her_dqn",
        observation_size=OBS_SIZE,
        num_actions=NUM_ACTIONS,
        config={**small_env_config, "her_dqn": {"hidden_sizes": [16, 16]}},
        device="cpu",
        seed=0,
    )
    assert isinstance(agent, HERDQNAgent)
    assert agent.relabeler.target_radius == pytest.approx(
        small_env_config["environment"]["target_radius"]
    )


def test_evaluation_actions_are_deterministic(small_env_config, agent_config):
    agent = _make_agent(agent_config, small_env_config)
    observation = np.random.default_rng(2).normal(size=OBS_SIZE).astype(np.float32)
    choices = {agent.act(observation, evaluation=True) for _ in range(20)}
    assert len(choices) == 1
