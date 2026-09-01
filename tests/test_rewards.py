"""Phase 6 tests: reward function correctness and configurability."""

from __future__ import annotations

import pytest

from drone_rl.environment.rewards import RewardConfig, RewardFunction


def base_kwargs(**overrides):
    """Neutral transition: no progress, no events, clear of obstacles."""
    kwargs = dict(
        previous_distance=10.0,
        current_distance=10.0,
        energy_used=0.0,
        obstacle_clearance=99.0,
        altitude=5.0,
        reached_target=False,
        collided=False,
        battery_empty=False,
        out_of_bounds=False,
        timed_out=False,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def reward_fn() -> RewardFunction:
    return RewardFunction(RewardConfig())


def test_config_comes_from_yaml_mapping():
    """Weights must be configurable, not hard-coded."""
    config = RewardConfig.from_mapping({"target": 250.0, "collision": -75.0, "step": -0.5})
    assert config.target == 250.0
    assert config.collision == -75.0
    assert config.step == -0.5
    # Unspecified keys keep their defaults.
    assert config.progress == RewardConfig().progress


def test_reaching_target_awards_configured_bonus(reward_fn):
    result = reward_fn.compute(**base_kwargs(reached_target=True))
    assert result.terminal_reward == pytest.approx(reward_fn.config.target)
    assert result.total > 90


def test_collision_applies_configured_penalty(reward_fn):
    result = reward_fn.compute(**base_kwargs(collided=True))
    assert result.terminal_reward == pytest.approx(reward_fn.config.collision)
    assert result.total < -90


def test_battery_failure_penalty(reward_fn):
    result = reward_fn.compute(**base_kwargs(battery_empty=True))
    assert result.terminal_reward == pytest.approx(reward_fn.config.battery_failure)


def test_out_of_bounds_penalty(reward_fn):
    result = reward_fn.compute(**base_kwargs(out_of_bounds=True))
    assert result.terminal_reward == pytest.approx(reward_fn.config.out_of_bounds)


def test_progress_reward_is_symmetric(reward_fn):
    """Potential-based shaping: closing and opening the same gap must cancel."""
    closer = reward_fn.compute(**base_kwargs(previous_distance=10.0, current_distance=8.0))
    further = reward_fn.compute(**base_kwargs(previous_distance=8.0, current_distance=10.0))
    assert closer.progress_reward == pytest.approx(-further.progress_reward)
    assert closer.progress_reward > 0


def test_progress_shaping_telescopes_to_zero_on_a_round_trip(reward_fn):
    """A path that returns to its start accrues zero net shaping reward.

    This is the property that makes the shaping potential-based, and hence
    policy-invariant: the agent cannot farm reward by oscillating.
    """
    distances = [20.0, 17.0, 12.0, 15.0, 20.0]
    total = 0.0
    for previous, current in zip(distances, distances[1:]):
        total += reward_fn.compute(
            **base_kwargs(previous_distance=previous, current_distance=current)
        ).progress_reward
    assert total == pytest.approx(0.0, abs=1e-9)


def test_step_penalty_is_always_applied(reward_fn):
    result = reward_fn.compute(**base_kwargs())
    assert result.step_penalty == pytest.approx(reward_fn.config.step)


def test_energy_cost_scales_with_consumption(reward_fn):
    low = reward_fn.compute(**base_kwargs(energy_used=1.0))
    high = reward_fn.compute(**base_kwargs(energy_used=4.0))
    assert high.energy_cost == pytest.approx(4.0 * low.energy_cost)
    assert high.energy_cost < 0


def test_proximity_penalty_grows_as_obstacle_nears(reward_fn):
    safe = reward_fn.compute(**base_kwargs(obstacle_clearance=5.0))
    near = reward_fn.compute(**base_kwargs(obstacle_clearance=1.0))
    nearer = reward_fn.compute(**base_kwargs(obstacle_clearance=0.2))

    assert safe.proximity_penalty == 0.0  # beyond safety_distance
    assert near.proximity_penalty < 0
    assert nearer.proximity_penalty < near.proximity_penalty


def test_beelining_into_a_wall_is_penalised_before_impact(reward_fn):
    """Regression: approaching a building must stop paying before the crash.

    The proximity term has to be sized against the progress term. If it is not,
    closing on a wall is net positive on every step and the only thing opposing
    it is a terminal penalty many discounted steps away - and DQN duly learns to
    fly straight into buildings (observed: 96% collision rate by episode 500
    with the original weights).
    """
    max_speed, dt = 3.0, 0.15
    closing_per_step = max_speed * dt  # best case progress in one step

    def total_at(clearance: float) -> float:
        return reward_fn.compute(
            **base_kwargs(
                previous_distance=20.0,
                current_distance=20.0 - closing_per_step,
                energy_used=0.2175,
                obstacle_clearance=clearance,
            )
        ).total

    # Far from buildings, making progress must pay.
    assert total_at(6.0) > 0
    # Close to a building, it must not - even at maximum closing speed.
    assert total_at(2.0) < 0
    assert total_at(1.0) < total_at(2.0)


def test_road_centre_is_not_penalised(reward_fn):
    """Roads must stay usable: ~3 m of clearance should cost nothing.

    A penalty that reaches into the middle of a road would make the free
    corridors as expensive as the buildings and remove the way through.
    """
    assert reward_fn.compute(**base_kwargs(obstacle_clearance=3.0)).proximity_penalty == 0.0


def test_altitude_penalty_only_below_threshold(reward_fn):
    high = reward_fn.compute(**base_kwargs(altitude=5.0))
    low = reward_fn.compute(**base_kwargs(altitude=0.2))
    assert high.altitude_penalty == 0.0
    assert low.altitude_penalty < 0


def test_components_sum_to_total(reward_fn):
    result = reward_fn.compute(
        **base_kwargs(
            previous_distance=12.0,
            current_distance=9.5,
            energy_used=0.3,
            obstacle_clearance=0.9,
            altitude=0.5,
            collided=True,
        )
    )
    assert sum(result.components.values()) == pytest.approx(result.total)


def test_breakdown_exposes_all_components_for_debugging(reward_fn):
    """The info dict must explain *why* a reward was given."""
    result = reward_fn.compute(**base_kwargs())
    expected = {
        "step_penalty",
        "energy_cost",
        "progress_reward",
        "proximity_penalty",
        "altitude_penalty",
        "terminal_reward",
    }
    assert expected == set(result.components)
    assert expected | {"total"} == set(result.as_dict())


def test_custom_weights_change_outcome():
    generous = RewardFunction(RewardConfig(target=1000.0))
    default = RewardFunction(RewardConfig())
    kwargs = base_kwargs(reached_target=True)
    assert generous.compute(**kwargs).total > default.compute(**kwargs).total + 800
