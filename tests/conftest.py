"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_rl.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "environment.yaml"


@pytest.fixture(scope="session")
def env_config() -> dict:
    """The project's environment configuration."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def small_env_config(env_config) -> dict:
    """A smaller/faster city, so tests stay quick without changing semantics."""
    config = {k: dict(v) if isinstance(v, dict) else v for k, v in env_config.items()}
    config["city"] = {**config["city"], "width": 40.0, "depth": 40.0}
    config["environment"] = {**config["environment"], "max_steps": 60}
    return config


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gui: requires a PyBullet GUI window")
    config.addinivalue_line("markers", "slow: long-running test")
