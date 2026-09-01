"""Training loop (Phase 9).

Algorithm-agnostic by construction: the loop only calls the
:class:`~drone_rl.agents.base.Agent` interface, so swapping DQN for a proposed
method changes nothing here. That is what makes the eventual
baseline-vs-proposed comparison an apples-to-apples one.

Per episode the trainer records reward, success, collision, steps, energy,
distance, epsilon and loss to CSV, JSON and TensorBoard, and delegates
checkpointing and periodic validation to callbacks.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from drone_rl.agents.base import Agent
from drone_rl.agents.registry import build_agent
from drone_rl.environment.drone_env import DroneCityEnv
from drone_rl.evaluation.metrics import METRIC_FIELDS, EpisodeMetrics
from drone_rl.training.callbacks import (
    Callback,
    CheckpointCallback,
    EvaluationCallback,
    ProgressCallback,
)
from drone_rl.utils.device import describe_device, resolve_device
from drone_rl.utils.logging import MetricsLogger, get_logger
from drone_rl.utils.reproducibility import ExperimentRecord, set_global_seeds

__all__ = ["Trainer", "TrainingConfig", "ConcurrentRunError"]

_LOG = get_logger(__name__)

#: Name of the lock file written into a run's log directory while it is active.
_LOCK_FILENAME = "RUNNING.lock"


class ConcurrentRunError(RuntimeError):
    """Raised when another live training run already owns this output directory."""


#: Windows ``GetExitCodeProcess`` returns this while a process is still running.
_WINDOWS_STILL_ACTIVE = 259
#: ``PROCESS_QUERY_LIMITED_INFORMATION`` - enough to read the exit code, and
#: obtainable for processes this one may not fully control.
_WINDOWS_QUERY_LIMITED_INFORMATION = 0x1000


def _process_is_alive(pid: int) -> bool:
    """Return whether ``pid`` refers to a running process.

    **Do not** implement this as ``os.kill(pid, 0)``. That is the POSIX idiom,
    but on Windows CPython implements ``os.kill`` as ``OpenProcess`` followed by
    ``TerminateProcess(handle, sig)`` - so ``os.kill(pid, 0)`` *terminates* the
    target with exit code 0 rather than probing it. Using it here would have let
    the run-lock check kill the very training run it exists to protect.

    Windows therefore goes through ``OpenProcess`` +  ``GetExitCodeProcess``.
    A process that genuinely exits with code 259 is indistinguishable from a
    running one; that is the accepted limitation of this standard technique, and
    the cost here is only a spuriously held lock.
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_WINDOWS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False  # no such process (or not visible to us)
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return exit_code.value == _WINDOWS_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)  # POSIX: signal 0 probes without delivering anything
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


@dataclass
class TrainingConfig:
    """Loop-level settings (the ``training:`` YAML block)."""

    total_episodes: int = 3_000
    learning_starts: int = 2_000
    train_frequency: int = 1
    gradient_steps: int = 1
    eval_frequency: int = 250
    eval_episodes: int = 30
    checkpoint_frequency: int = 250
    log_frequency: int = 10
    save_trajectories: bool = False
    device: str = "auto"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "TrainingConfig":
        if not mapping:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


class Trainer:
    """Runs an agent against :class:`DroneCityEnv` for a fixed episode budget.

    Parameters
    ----------
    config:
        Full merged experiment config.
    output_dir:
        Root for checkpoints, logs and metrics; defaults to
        ``experiment.output_dir``.
    device:
        Overrides ``training.device``.
    callbacks:
        Replaces the default checkpoint/eval/progress callbacks when given.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: str | Path | None = None,
        device: str | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        self.config = dict(config)
        self.experiment_config = dict(self.config.get("experiment", {}))
        self.training_config = TrainingConfig.from_mapping(self.config.get("training"))
        if device is not None:
            self.training_config.device = device

        self.experiment_name = str(self.experiment_config.get("name", "experiment"))
        self.algorithm = str(self.experiment_config.get("algorithm", "dqn"))
        self.seed = int(self.experiment_config.get("seed", 42))

        root = Path(output_dir or self.experiment_config.get("output_dir", "outputs"))
        self.output_dir = root
        self.checkpoint_dir = root / "checkpoints"
        self.log_dir = root / "logs" / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Claim the output directory before anything is written into it.
        self._lock_path = self.log_dir / _LOCK_FILENAME
        self._acquire_run_lock()

        set_global_seeds(
            self.seed,
            deterministic_torch=bool(self.experiment_config.get("deterministic_torch", False)),
        )
        self.device = resolve_device(self.training_config.device)
        self.device_info = describe_device(self.device)

        # The training environment always draws from the *train* seed pool.
        self.env = DroneCityEnv(config=self.config, seed_pool="train")
        self.agent: Agent = build_agent(
            algorithm=self.algorithm,
            observation_size=int(self.env.observation_space.shape[0]),
            num_actions=int(self.env.action_space.n),
            config=self.config,
            device=self.device,
            seed=self.seed,
        )

        self.metrics_logger = MetricsLogger(
            output_dir=self.log_dir,
            tensorboard_dir=self.log_dir / "tensorboard",
            fieldnames=list(METRIC_FIELDS),
        )
        self.callbacks: list[Callback] = (
            list(callbacks)
            if callbacks is not None
            else self._default_callbacks()
        )

        self.global_step = 0
        self.episode_index = 0
        self.episode_metrics: list[EpisodeMetrics] = []
        self._evaluation_log: list[dict[str, Any]] = []
        self._started_at = 0.0

    # ---- run lock ------------------------------------------------------------

    def _acquire_run_lock(self) -> None:
        """Refuse to start if a live run already owns this output directory.

        Two trainers sharing one ``--name`` interleave their episodes into the
        same ``metrics.csv`` and overwrite each other's checkpoints, producing
        results that look plausible but are a blend of two runs. That happened
        twice during development (a backgrounded run outliving the shell that
        launched it), so the failure is guarded rather than merely documented.

        A lock naming a dead PID is stale and is reclaimed automatically.
        """
        if self._lock_path.exists():
            try:
                existing = json.loads(self._lock_path.read_text(encoding="utf-8"))
                owner_pid = int(existing.get("pid", -1))
            except (OSError, ValueError, json.JSONDecodeError):
                owner_pid = -1

            # A live owner blocks the run even when it is this same process:
            # two Trainer instances sharing an output directory corrupt it just
            # as thoroughly as two processes do.
            if owner_pid > 0 and _process_is_alive(owner_pid):
                who = "This process" if owner_pid == os.getpid() else f"PID {owner_pid}"
                raise ConcurrentRunError(
                    f"{who} already holds a training run writing to {self.log_dir}.\n"
                    f"Stop it first, or pass a different --name so the two runs use "
                    f"separate output directories.\n"
                    f"If you are certain that run is dead, delete {self._lock_path}."
                )
            _LOG.warning("Reclaiming stale run lock from PID %s", owner_pid)

        payload = {
            "pid": os.getpid(),
            "experiment": self.experiment_name,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        self._lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Covers normal exit and unhandled exceptions; KeyboardInterrupt is
        # additionally handled by train()'s finally block.
        atexit.register(self._release_run_lock)

    def _release_run_lock(self) -> None:
        """Remove this process's run lock, if it still owns it."""
        try:
            if not self._lock_path.exists():
                return
            payload = json.loads(self._lock_path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                self._lock_path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def _default_callbacks(self) -> list[Callback]:
        """Checkpointing, periodic validation and console progress."""
        prefix = self.algorithm
        return [
            ProgressCallback(frequency=self.training_config.log_frequency),
            EvaluationCallback(
                directory=self.checkpoint_dir,
                prefix=prefix,
                frequency=self.training_config.eval_frequency,
                episodes=self.training_config.eval_episodes,
                pool="validation",
            ),
            CheckpointCallback(
                directory=self.checkpoint_dir,
                prefix=prefix,
                frequency=self.training_config.checkpoint_frequency,
            ),
        ]

    # ---- main loop -----------------------------------------------------------

    def train(self, total_episodes: int | None = None) -> list[EpisodeMetrics]:
        """Run the training loop and return the per-episode metrics."""
        episodes = int(total_episodes or self.training_config.total_episodes)
        self._started_at = time.perf_counter()
        self.agent.train_mode()

        _LOG.info(
            "Training %s for %d episodes on device=%s (obs=%d, actions=%d)",
            self.algorithm,
            episodes,
            self.device,
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        )
        self.save_experiment_record()
        for callback in self.callbacks:
            callback.on_training_start(self)

        try:
            for episode in range(1, episodes + 1):
                self.episode_index = episode
                metrics = self._run_episode(episode)
                self.episode_metrics.append(metrics)
                self.metrics_logger.log(metrics.as_row(), step=episode)
                for callback in self.callbacks:
                    callback.on_episode_end(self, episode, metrics)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            _LOG.warning("Training interrupted at episode %d; saving state.", self.episode_index)
        finally:
            for callback in self.callbacks:
                callback.on_training_end(self)
            self.write_summary()
            self.metrics_logger.close()
            self.env.close()
            self._release_run_lock()

        return self.episode_metrics

    def _run_episode(self, episode: int) -> EpisodeMetrics:
        """Roll out one training episode, learning as it goes."""
        # Seeding each episode from (seed, episode) keeps the whole training run
        # reproducible while still visiting a different scenario every episode.
        episode_seed = self.seed * 1_000_003 + episode
        observation, _info = self.env.reset(seed=episode_seed)

        losses: list[float] = []
        q_means: list[float] = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = self.agent.act(observation, evaluation=False)
            next_observation, reward, terminated, truncated, _info = self.env.step(action)

            # `terminated` alone is the bootstrap flag: a time-limit truncation
            # must keep bootstrapping from next_observation.
            self.agent.observe(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
            observation = next_observation
            self.global_step += 1

            if self._should_update():
                for _ in range(max(1, self.training_config.gradient_steps)):
                    stats = self.agent.update()
                    if stats:
                        losses.append(stats.get("loss", float("nan")))
                        q_means.append(stats.get("q_mean", float("nan")))

        summary = self.env.episode_summary()
        metrics = EpisodeMetrics.from_summary(summary, episode=episode)
        metrics.status = summary["status"]
        diagnostics = self.agent.diagnostics
        metrics.epsilon = float(diagnostics.get("epsilon", float("nan")))
        metrics.loss = float(np.mean(losses)) if losses else float("nan")
        metrics.q_mean = float(np.mean(q_means)) if q_means else float("nan")

        self.agent.on_episode_end(episode, summary)
        return metrics

    def _should_update(self) -> bool:
        """True when a gradient step is due at the current env step."""
        if self.global_step < self.training_config.learning_starts:
            return False
        return self.global_step % max(1, self.training_config.train_frequency) == 0

    # ---- persistence ---------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        episode: int,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a checkpoint carrying the agent plus its full provenance.

        The config, seeds and device are embedded so a checkpoint alone is
        enough to reproduce or fairly compare the run.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": self.agent.state_dict(),
            "algorithm": self.algorithm,
            "episode": int(episode),
            "global_step": int(self.global_step),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "config": self.config,
            "device": self.device_info.as_dict(),
            "observation_labels": self.env.observation_labels,
            "action_names": list(self.env.action_names),
        }
        if extra:
            payload["extra"] = dict(extra)
        torch.save(payload, out_path)
        return out_path

    def save_experiment_record(self) -> Path:
        """Write the provenance JSON next to the checkpoints."""
        record = ExperimentRecord(
            name=self.experiment_name,
            seed=self.seed,
            config=self.config,
            device=self.device_info.as_dict(),
            extra={
                "algorithm": self.algorithm,
                "observation_size": int(self.env.observation_space.shape[0]),
                "num_actions": int(self.env.action_space.n),
                "action_names": list(self.env.action_names),
            },
        )
        return record.save(self.log_dir / "experiment.json")

    def log_evaluation(self, record: Mapping[str, Any], pool: str) -> None:
        """Record one validation sweep (called by :class:`EvaluationCallback`)."""
        entry = {"pool": pool, **dict(record)}
        self._evaluation_log.append(entry)
        path = self.log_dir / "evaluations.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._evaluation_log, handle, indent=2, default=str)

    def write_summary(self) -> Path:
        """Write the end-of-run JSON summary."""
        from drone_rl.evaluation.metrics import aggregate_metrics

        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        last_100 = self.episode_metrics[-100:]
        summary = {
            "experiment_name": self.experiment_name,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "episodes_completed": len(self.episode_metrics),
            "env_steps": self.global_step,
            "wall_clock_seconds": round(elapsed, 2),
            "steps_per_second": round(self.global_step / max(elapsed, 1e-6), 1),
            "device": self.device_info.as_dict(),
            "final_100_episodes": aggregate_metrics(last_100),
            "all_episodes": aggregate_metrics(self.episode_metrics),
            "validation_history": self._evaluation_log,
            "config": self.config,
        }
        return self.metrics_logger.write_summary(summary)
