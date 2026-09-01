"""Simulation environment: procedural city, kinematic drone, sensors, rewards."""

from drone_rl.environment.city import Building, CityConfig, CityLayout, generate_city
from drone_rl.environment.drone import Drone, DroneConfig, DroneState
from drone_rl.environment.drone_env import DroneCityEnv, EnvConfig, EpisodeOutcome
from drone_rl.environment.rewards import RewardBreakdown, RewardConfig, RewardFunction
from drone_rl.environment.sensors import ObservationEncoder, RangeSensorArray, SensorConfig
from drone_rl.environment.spaces import ActionSet, ActionSpec, DEFAULT_ACTIONS, build_action_set

__all__ = [
    "Building",
    "CityConfig",
    "CityLayout",
    "generate_city",
    "Drone",
    "DroneConfig",
    "DroneState",
    "DroneCityEnv",
    "EnvConfig",
    "EpisodeOutcome",
    "RewardBreakdown",
    "RewardConfig",
    "RewardFunction",
    "ObservationEncoder",
    "RangeSensorArray",
    "SensorConfig",
    "ActionSet",
    "ActionSpec",
    "DEFAULT_ACTIONS",
    "build_action_set",
]
