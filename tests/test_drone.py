"""Phase 2 tests: kinematic drone model."""

from __future__ import annotations

import numpy as np
import pytest

from drone_rl.environment.drone import Drone, DroneConfig
from drone_rl.environment.spaces import DEFAULT_ACTIONS, ActionSet, build_action_set


@pytest.fixture
def drone() -> Drone:
    return Drone(DroneConfig(velocity_smoothing=0.0, dt=0.1, max_speed=2.0))


@pytest.fixture
def actions() -> ActionSet:
    return ActionSet(DEFAULT_ACTIONS)


def test_reset_places_drone_with_full_battery(drone):
    state = drone.reset(np.array([1.0, 2.0, 3.0]), yaw=0.5)
    assert state.position == pytest.approx([1.0, 2.0, 3.0])
    assert state.velocity == pytest.approx([0.0, 0.0, 0.0])
    assert state.battery == pytest.approx(drone.config.battery_capacity)
    assert drone.battery_fraction == pytest.approx(1.0)


def test_forward_action_moves_along_heading(drone, actions):
    """With yaw=0 the body +x axis is the world +x axis."""
    drone.reset(np.zeros(3), yaw=0.0)
    position, velocity, yaw, _energy = drone.propose_step(actions[actions.index("forward")])

    assert position[0] > 0
    assert position[1] == pytest.approx(0.0, abs=1e-9)
    assert position[2] == pytest.approx(0.0, abs=1e-9)
    # No smoothing, so velocity equals the commanded max speed.
    assert velocity[0] == pytest.approx(drone.config.max_speed)


def test_forward_action_respects_yaw(drone, actions):
    """Rotated 90 degrees, 'forward' must move along world +y."""
    drone.reset(np.zeros(3), yaw=np.pi / 2)
    position, _velocity, _yaw, _e = drone.propose_step(actions[actions.index("forward")])
    assert position[0] == pytest.approx(0.0, abs=1e-9)
    assert position[1] > 0


def test_left_action_is_ninety_degrees_from_forward(drone, actions):
    drone.reset(np.zeros(3), yaw=0.0)
    left_pos, *_ = drone.propose_step(actions[actions.index("left")])
    assert left_pos[1] > 0  # body +y is world +y at yaw 0
    assert left_pos[0] == pytest.approx(0.0, abs=1e-9)


def test_ascend_and_descend_change_altitude(drone, actions):
    drone.reset(np.array([0.0, 0.0, 5.0]), yaw=0.0)
    up, *_ = drone.propose_step(actions[actions.index("ascend")])
    down, *_ = drone.propose_step(actions[actions.index("descend")])
    assert up[2] > 5.0
    assert down[2] < 5.0


def test_yaw_actions_rotate_and_wrap(drone, actions):
    drone.reset(np.zeros(3), yaw=0.0)
    _p, _v, yaw_left, _e = drone.propose_step(actions[actions.index("yaw_left")])
    _p, _v, yaw_right, _e = drone.propose_step(actions[actions.index("yaw_right")])
    assert yaw_left > 0
    assert yaw_right < 0

    # Yaw must stay wrapped to [-pi, pi] rather than growing without bound.
    drone.reset(np.zeros(3), yaw=np.pi - 0.01)
    for _ in range(50):
        _p, _v, yaw, _e = drone.propose_step(actions[actions.index("yaw_left")])
        drone.commit(drone.state.position, drone.state.velocity, yaw, 0.0)
    assert -np.pi <= drone.state.yaw <= np.pi


def test_propose_step_does_not_mutate_state(drone, actions):
    drone.reset(np.array([1.0, 1.0, 1.0]), yaw=0.3)
    before = drone.state.copy()
    drone.propose_step(actions[actions.index("forward")])
    assert drone.state.position == pytest.approx(before.position)
    assert drone.state.battery == pytest.approx(before.battery)
    assert drone.state.yaw == pytest.approx(before.yaw)


def test_commit_applies_transition_and_drains_battery(drone, actions):
    drone.reset(np.zeros(3), yaw=0.0)
    position, velocity, yaw, energy = drone.propose_step(actions[actions.index("forward")])
    drone.commit(position, velocity, yaw, energy)

    assert drone.state.position == pytest.approx(position)
    assert drone.state.battery < drone.config.battery_capacity
    assert drone.state.elapsed_time == pytest.approx(drone.config.dt)


def test_battery_never_goes_negative(drone, actions):
    drone.reset(np.zeros(3), yaw=0.0)
    for _ in range(10_000):
        p, v, y, e = drone.propose_step(actions[actions.index("ascend")])
        drone.commit(p, v, y, e)
        if drone.state.battery <= 0:
            break
    assert drone.state.battery == pytest.approx(0.0)


def test_hover_costs_less_energy_than_manoeuvring(drone, actions):
    hover = drone.energy_for_action(actions[actions.index("hover")])
    forward = drone.energy_for_action(actions[actions.index("forward")])
    ascend = drone.energy_for_action(actions[actions.index("ascend")])
    assert hover < forward < ascend
    assert hover > 0  # hovering is not free


def test_velocity_smoothing_creates_momentum():
    """With smoothing the drone cannot reach commanded speed in one step."""
    smooth = Drone(DroneConfig(velocity_smoothing=0.8, dt=0.1, max_speed=3.0))
    smooth.reset(np.zeros(3), yaw=0.0)
    forward = ActionSet(DEFAULT_ACTIONS)[1]

    _p, velocity, _y, _e = smooth.propose_step(forward)
    assert velocity[0] < smooth.config.max_speed * 0.5

    # Repeated commands asymptotically approach the target speed.
    for _ in range(50):
        p, v, y, e = smooth.propose_step(forward)
        smooth.commit(p, v, y, e)
    assert smooth.speed == pytest.approx(smooth.config.max_speed, rel=0.05)


def test_body_to_world_round_trip(drone):
    drone.reset(np.zeros(3), yaw=0.7)
    body = np.array([1.0, 2.0, 3.0])
    world = drone.body_to_world(body)
    assert np.linalg.norm(world) == pytest.approx(np.linalg.norm(body))


def test_orientation_quaternion_is_unit(drone):
    drone.reset(np.zeros(3), yaw=1.1)
    quat = np.array(drone.orientation_quaternion())
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_action_set_from_config_restricts_actions():
    action_set = build_action_set({"use": ["hover", "forward", "ascend"]})
    assert len(action_set) == 3
    assert action_set.names == ("hover", "forward", "ascend")


def test_action_set_rejects_unknown_names():
    with pytest.raises(KeyError):
        build_action_set({"use": ["teleport"]})


def test_custom_action_set_from_config():
    action_set = build_action_set(
        {"custom": [{"name": "dash", "forward": 1.0, "energy_scale": 2.0}]}
    )
    assert action_set.names == ("dash",)
    assert action_set[0].energy_scale == 2.0
