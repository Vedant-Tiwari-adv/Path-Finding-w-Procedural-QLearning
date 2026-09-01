"""Trainer tests, focused on the run lock and end-to-end loop wiring.

The lock covers a failure seen twice during development: a backgrounded run
outlived the shell that launched it, a second run was started with the same
``--name``, and the two interleaved their episodes into one ``metrics.csv``
while overwriting each other's checkpoints. The blended result looks plausible,
which is what makes it dangerous.
"""

from __future__ import annotations

import json
import os

import pytest

from drone_rl.training.trainer import ConcurrentRunError, Trainer, _process_is_alive


def test_process_liveness_detects_self_and_missing_pids():
    """The probe must be correct on this platform, and must not kill anything.

    On Windows ``os.kill(pid, 0)`` terminates the target rather than probing it,
    so this asserts the probe both reports correctly and leaves the process
    running afterwards.
    """
    assert _process_is_alive(os.getpid())
    assert _process_is_alive(os.getpid()), "probe must be repeatable, not destructive"
    assert not _process_is_alive(4_194_304)  # above every platform's pid_max
    assert not _process_is_alive(0)
    assert not _process_is_alive(-1)


def test_process_liveness_does_not_kill_a_real_child():
    """Probing a live child process must leave it running."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _process_is_alive(child.pid)
        assert child.poll() is None, "liveness probe terminated the child process"
        assert _process_is_alive(child.pid)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=10)
    assert not _process_is_alive(child.pid)


@pytest.fixture
def tiny_config(small_env_config) -> dict:
    """A config small enough to train for a handful of episodes in a test."""
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in small_env_config.items()}
    config["environment"] = {**config["environment"], "max_steps": 15}
    config["experiment"] = {"name": "unit_test", "algorithm": "dqn", "seed": 0}
    config["training"] = {
        "total_episodes": 3,
        "learning_starts": 10,
        "train_frequency": 1,
        "eval_frequency": 10_000,       # never during the test
        "checkpoint_frequency": 10_000,
        "log_frequency": 10_000,
    }
    config["dqn"] = {
        "buffer_size": 500,
        "batch_size": 8,
        "min_replay_size": 10,
        "hidden_sizes": [16, 16],
        "epsilon_decay_steps": 100,
    }
    return config


def test_training_loop_runs_and_logs(tiny_config, tmp_path):
    trainer = Trainer(config=tiny_config, output_dir=tmp_path)
    metrics = trainer.train()

    assert len(metrics) == 3
    assert trainer.global_step > 0
    assert trainer.metrics_logger.csv_path.is_file()
    # Every episode must land in exactly one terminal category.
    for record in metrics:
        flags = [record.success, record.collision, record.battery_empty,
                 record.out_of_bounds, record.timeout]
        assert sum(bool(f) for f in flags) == 1


def test_checkpoint_round_trip(tiny_config, tmp_path):
    trainer = Trainer(config=tiny_config, output_dir=tmp_path)
    trainer.train()

    path = trainer.save_checkpoint(tmp_path / "ckpt.pt", episode=3)
    assert path.is_file()

    import torch

    from drone_rl.agents.registry import agent_from_checkpoint

    payload = torch.load(path, map_location="cpu", weights_only=False)
    # A checkpoint must carry enough provenance to reproduce the run.
    for key in ("agent", "config", "seed", "device", "action_names", "episode"):
        assert key in payload
    assert agent_from_checkpoint(payload, device="cpu") is not None


def test_second_concurrent_run_is_refused(tiny_config, tmp_path):
    """A live run must block a second trainer on the same output directory."""
    first = Trainer(config=tiny_config, output_dir=tmp_path)
    try:
        with pytest.raises(ConcurrentRunError, match="already holds a training run"):
            Trainer(config=tiny_config, output_dir=tmp_path)
    finally:
        first._release_run_lock()


def test_different_names_do_not_conflict(tiny_config, tmp_path):
    """The documented workaround must actually work."""
    first = Trainer(config=tiny_config, output_dir=tmp_path)
    other = {**tiny_config, "experiment": {**tiny_config["experiment"], "name": "unit_test_b"}}
    try:
        second = Trainer(config=other, output_dir=tmp_path)
        second._release_run_lock()
    finally:
        first._release_run_lock()


def test_stale_lock_is_reclaimed(tiny_config, tmp_path):
    """A lock naming a dead PID must not block a new run forever."""
    log_dir = tmp_path / "logs" / "unit_test"
    log_dir.mkdir(parents=True, exist_ok=True)
    # PID 2^22 is above every platform's pid_max, so it cannot be running.
    (log_dir / "RUNNING.lock").write_text(
        json.dumps({"pid": 4_194_304, "experiment": "unit_test"}), encoding="utf-8"
    )

    trainer = Trainer(config=tiny_config, output_dir=tmp_path)
    try:
        owner = json.loads((log_dir / "RUNNING.lock").read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
    finally:
        trainer._release_run_lock()


def test_lock_released_after_training(tiny_config, tmp_path):
    trainer = Trainer(config=tiny_config, output_dir=tmp_path)
    trainer.train()
    assert not trainer._lock_path.exists()


def test_truncation_is_not_stored_as_terminal(tiny_config, tmp_path):
    """The trainer must pass `terminated` (not `terminated or truncated`).

    With `max_steps` this small most episodes end by truncation, so the replay
    buffer would be almost entirely (wrongly) terminal if this were broken.
    """
    trainer = Trainer(config=tiny_config, output_dir=tmp_path)
    try:
        trainer.train()
        buffer = trainer.agent.replay_buffer
        stored = buffer._terminated[: len(buffer)]
        assert stored.min() == 0.0, "No non-terminal transitions were stored"
    finally:
        trainer._release_run_lock()
