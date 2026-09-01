"""Shared utilities: configuration, device selection, logging, reproducibility."""

from drone_rl.utils.config import apply_overrides, deep_merge, get_in, load_config, save_config
from drone_rl.utils.device import DeviceInfo, describe_device, resolve_device
from drone_rl.utils.logging import MetricsLogger, configure_logging, get_logger
from drone_rl.utils.reproducibility import (
    ExperimentRecord,
    SeedRanges,
    seed_ranges_from_config,
    set_global_seeds,
)

__all__ = [
    "apply_overrides",
    "deep_merge",
    "get_in",
    "load_config",
    "save_config",
    "DeviceInfo",
    "describe_device",
    "resolve_device",
    "MetricsLogger",
    "configure_logging",
    "get_logger",
    "ExperimentRecord",
    "SeedRanges",
    "seed_ranges_from_config",
    "set_global_seeds",
]
