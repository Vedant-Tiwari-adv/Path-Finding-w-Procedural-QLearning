"""Phase 5 tests: Gymnasium environment contract, determinism, termination."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from drone_rl.environment.drone_env import DroneCityEnv, EpisodeOutcome


@pytest.fixture
def env(small_env_config):
    environment = DroneCityEnv(config=small_env_config)
    yield environment
    environment.close()


def test_spaces_are_well_formed(env):
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.action_space.n == 9
    # 14 state features + one reading per sensor ray.
    assert env.observation_space.shape == (14 + env.sensors.num_rays,)


def test_reset_returns_valid_observation(env):
    observation, info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert info["status"] == EpisodeOutcome.RUNNING


def test_step_returns_five_tuple_with_correct_types(env):
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_same_seed_reproduces_identical_episode(env, small_env_config):
    """Reproducibility: identical seed and actions must replay exactly."""
    other = DroneCityEnv(config=small_env_config)
    try:
        obs_a, _ = env.reset(seed=1234)
        obs_b, _ = other.reset(seed=1234)
        np.testing.assert_allclose(obs_a, obs_b)

        actions = [1, 1, 5, 3, 1, 7, 1, 2, 6, 1]
        for action in actions:
            a = env.step(action)
            b = other.step(action)
            np.testing.assert_allclose(a[0], b[0], atol=0)
            assert a[1] == pytest.approx(b[1])
            assert a[2] == b[2] and a[3] == b[3]
            if a[2] or a[3]:
                break
        assert env.city.seed == other.city.seed
        np.testing.assert_allclose(env.start_position, other.start_position)
        np.testing.assert_allclose(env.target_position, other.target_position)
    finally:
        other.close()


def test_different_seeds_give_different_episodes(env):
    env.reset(seed=1)
    first = (env.city.seed, env.start_position.copy(), env.target_position.copy())
    env.reset(seed=2)
    second = (env.city.seed, env.start_position.copy(), env.target_position.copy())
    assert first[0] != second[0] or not np.allclose(first[1], second[1])


def test_environment_does_not_use_global_numpy_seed(env, small_env_config):
    """Episode sampling must not depend on global RNG state."""
    np.random.seed(999)
    obs_a, _ = env.reset(seed=77)
    np.random.seed(1)
    obs_b, _ = env.reset(seed=77)
    np.testing.assert_allclose(obs_a, obs_b)


def test_start_and_target_are_collision_free(env):
    """Neither endpoint may spawn inside a building."""
    from drone_rl.environment.geometry import sphere_hits_boxes

    for seed in range(12):
        env.reset(seed=seed)
        mins, maxs = env.city.box_arrays()
        radius = env.drone_config.collision_radius
        assert not sphere_hits_boxes(env.start_position, radius, mins, maxs)
        assert not sphere_hits_boxes(env.target_position, radius, mins, maxs)


def test_truncation_is_not_termination(small_env_config):
    """A time limit must set truncated, never terminated.

    Bootstrapping depends on this distinction: treating a timeout as terminal
    would teach the agent the world ends at max_steps.
    """
    config = {**small_env_config, "environment": {**small_env_config["environment"], "max_steps": 8}}
    env = DroneCityEnv(config=config)
    try:
        # Hovering cannot reach the target, crash, or drain the battery in 8
        # steps, so the only possible ending is the time limit.
        env.reset(seed=3)
        hover = env.action_set.index("hover")
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _obs, _r, terminated, truncated, _info = env.step(hover)
            steps += 1
        assert steps == 8
        assert truncated is True
        assert terminated is False
        assert env.status == EpisodeOutcome.TIMEOUT
    finally:
        env.close()


def test_collision_terminates_and_is_reported(small_env_config):
    """Flying into a building must terminate with a collision status."""
    env = DroneCityEnv(config={**small_env_config,
                               "environment": {**small_env_config["environment"], "max_steps": 400}})
    try:
        found = False
        for seed in range(40):
            env.reset(seed=seed)
            # Aim the drone at the nearest building and fly straight at it.
            building = min(
                env.city.buildings,
                key=lambda b: np.hypot(
                    b.center_x - env.drone.state.position[0],
                    b.center_y - env.drone.state.position[1],
                ),
            )
            delta = np.array([building.center_x, building.center_y]) - env.drone.state.position[:2]
            env.drone.state.yaw = float(np.arctan2(delta[1], delta[0]))

            forward = env.action_set.index("forward")
            for _ in range(120):
                _o, _r, terminated, truncated, info = env.step(forward)
                if terminated and info["collision"]:
                    found = True
                    break
                if terminated or truncated:
                    break
            if found:
                break
        assert found, "Flying straight at buildings never produced a collision"
        assert env.status == EpisodeOutcome.COLLISION
        assert info["success"] is False
    finally:
        env.close()


def test_collision_stops_the_drone_outside_the_building(small_env_config):
    """After a crash the drone must not be left inside geometry."""
    from drone_rl.environment.geometry import sphere_hits_boxes

    env = DroneCityEnv(config=small_env_config)
    try:
        rng = np.random.default_rng(0)
        for seed in range(30):
            env.reset(seed=seed)
            for _ in range(60):
                _o, _r, terminated, _t, info = env.step(int(rng.integers(0, 9)))
                if terminated and info["collision"]:
                    mins, maxs = env.city.box_arrays()
                    assert not sphere_hits_boxes(
                        env.drone.state.position,
                        env.drone_config.collision_radius * 0.999,
                        mins,
                        maxs,
                    )
                    return
    finally:
        env.close()


def test_battery_depletion_terminates(small_env_config):
    """A tiny battery must end the episode with BATTERY EMPTY."""
    config = {
        **small_env_config,
        "drone": {**small_env_config["drone"], "battery_capacity": 1.2},
        "environment": {**small_env_config["environment"], "max_steps": 400},
    }
    env = DroneCityEnv(config=config)
    try:
        env.reset(seed=5)
        hover = env.action_set.index("hover")
        terminated = truncated = False
        while not (terminated or truncated):
            _o, _r, terminated, truncated, info = env.step(hover)
        assert terminated is True
        assert info["battery_empty"] is True
        assert env.status == EpisodeOutcome.BATTERY_EMPTY
    finally:
        env.close()


def test_reaching_target_terminates_with_success(small_env_config):
    """Spawning inside the target radius must immediately succeed."""
    env = DroneCityEnv(config=small_env_config)
    try:
        env.reset(seed=9)
        # Place the drone one small step away from the target.
        target = env.target_position.copy()
        env.drone.state.position = target + np.array([0.0, 0.0, 0.05])
        env._previous_distance = 0.05
        hover = env.action_set.index("hover")
        _o, reward, terminated, truncated, info = env.step(hover)

        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        assert reward > 90
        assert env.status == EpisodeOutcome.TARGET_REACHED
    finally:
        env.close()


def test_out_of_bounds_terminates(small_env_config):
    env = DroneCityEnv(config=small_env_config)
    try:
        env.reset(seed=4)
        half_width = env.city_config.width / 2
        env.drone.state.position = np.array([half_width - 0.2, 0.0, 5.0])
        env.drone.state.yaw = 0.0
        forward = env.action_set.index("forward")
        terminated = truncated = False
        for _ in range(20):
            _o, _r, terminated, truncated, info = env.step(forward)
            if terminated or truncated:
                break
        assert terminated is True
        assert info["out_of_bounds"] is True
        assert env.status == EpisodeOutcome.OUT_OF_BOUNDS
    finally:
        env.close()


def test_info_exposes_reward_components(env):
    """Reward debugging requirement: components must be visible in info."""
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(1)
    for key in (
        "distance_to_target",
        "progress_reward",
        "energy_cost",
        "collision",
        "success",
        "battery",
        "step_penalty",
        "proximity_penalty",
        "terminal_reward",
        "total",
    ):
        assert key in info, f"info is missing {key!r}"
    assert info["total"] == pytest.approx(_r)


def test_reward_components_sum_to_returned_reward(env):
    env.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(25):
        _o, reward, terminated, truncated, info = env.step(int(rng.integers(0, 9)))
        assert sum(info["reward_components"].values()) == pytest.approx(reward)
        if terminated or truncated:
            break


def test_observation_stays_in_bounds_over_long_rollout(env):
    """Normalisation must hold even in extreme states."""
    rng = np.random.default_rng(3)
    for seed in range(6):
        observation, _ = env.reset(seed=seed)
        assert env.observation_space.contains(observation)
        for _ in range(60):
            observation, _r, terminated, truncated, _i = env.step(int(rng.integers(0, 9)))
            assert env.observation_space.contains(observation), observation
            assert np.all(np.isfinite(observation))
            if terminated or truncated:
                break


def test_observation_does_not_leak_privileged_information(env):
    """The agent must not receive the city layout or a shortest path."""
    labels = env.observation_labels
    assert len(labels) == env.observation_space.shape[0]
    forbidden = ("path", "route", "waypoint", "building", "layout", "map", "seed")
    for label in labels:
        assert not any(token in label.lower() for token in forbidden), label


def test_seed_pools_are_disjoint(small_env_config):
    """Test seeds must never be reachable from the training pool."""
    train = DroneCityEnv(config=small_env_config, seed_pool="train")
    test = DroneCityEnv(config=small_env_config, seed_pool="test")
    try:
        train_seeds = {train.reset(seed=s)[1]["city_seed"] for s in range(120)}
        test_seeds = {test.reset(seed=s)[1]["city_seed"] for s in range(120)}
        assert train_seeds.isdisjoint(test_seeds)
        assert max(train_seeds) <= small_env_config["city"]["train_seed_end"]
        assert min(test_seeds) >= small_env_config["city"]["test_seed_start"]
    finally:
        train.close()
        test.close()


def test_options_pin_scenario_for_fair_comparison(env):
    """compare_learning.py relies on pinning city/start/target explicitly."""
    start = np.array([2.0, 3.0, 4.0])
    target = np.array([-6.0, -5.0, 3.0])
    _obs, info = env.reset(seed=0, options={"city_seed": 123, "start": start, "target": target})
    assert info["city_seed"] == 123
    np.testing.assert_allclose(env.start_position, start)
    np.testing.assert_allclose(env.target_position, target)


def test_episode_summary_is_consistent(env):
    env.reset(seed=8)
    rng = np.random.default_rng(1)
    terminated = truncated = False
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(int(rng.integers(0, 9)))
    summary = env.episode_summary()

    assert summary["steps"] == env.step_count
    assert summary["reward"] == pytest.approx(env.episode_reward)
    assert summary["flight_time"] == pytest.approx(env.step_count * env.drone_config.dt)
    assert summary["path_length"] >= 0
    # Exactly one terminal category must be set.
    flags = ["success", "collision", "battery_empty", "out_of_bounds", "timeout"]
    assert sum(bool(summary[f]) for f in flags) == 1


def test_step_before_reset_raises(small_env_config):
    env = DroneCityEnv(config=small_env_config)
    try:
        with pytest.raises(RuntimeError):
            env.step(0)
    finally:
        env.close()


def test_invalid_action_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)
