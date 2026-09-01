"""Training callbacks (Phase 9).

Checkpointing, periodic validation and console progress are separated from the
training loop so the loop itself stays readable and so a research variant can
add behaviour (curiosity statistics, curriculum changes, latent-space probes)
without editing :class:`~drone_rl.training.trainer.Trainer`.

Callbacks receive the trainer instance, so they can read anything they need
without the trainer having to anticipate it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from drone_rl.evaluation.evaluate import evaluate_agent
from drone_rl.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from drone_rl.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from drone_rl.training.trainer import Trainer

__all__ = ["Callback", "CheckpointCallback", "EvaluationCallback", "ProgressCallback"]

_LOG = get_logger(__name__)


class Callback:
    """Base class; override the hooks you need."""

    def on_training_start(self, trainer: "Trainer") -> None:
        return None

    def on_episode_end(
        self, trainer: "Trainer", episode: int, metrics: EpisodeMetrics
    ) -> None:
        return None

    def on_training_end(self, trainer: "Trainer") -> None:
        return None


class CheckpointCallback(Callback):
    """Writes periodic and final checkpoints.

    Files are named ``{prefix}_episode_{n}.pt``, matching the naming the demo
    and comparison scripts expect. ``{prefix}_last.pt`` always points at the
    most recent state so a run can be resumed or demoed at any time.
    """

    def __init__(
        self,
        directory: str | Path,
        prefix: str = "dqn",
        frequency: int = 250,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.frequency = max(1, int(frequency))

    def on_episode_end(self, trainer: "Trainer", episode: int, metrics: EpisodeMetrics) -> None:
        if episode % self.frequency == 0:
            path = self.directory / f"{self.prefix}_episode_{episode}.pt"
            trainer.save_checkpoint(path, episode=episode)
            _LOG.info("Saved checkpoint %s", path.name)
        trainer.save_checkpoint(self.directory / f"{self.prefix}_last.pt", episode=episode)

    def on_training_end(self, trainer: "Trainer") -> None:
        path = self.directory / f"{self.prefix}_final.pt"
        trainer.save_checkpoint(path, episode=trainer.episode_index)
        _LOG.info("Saved final checkpoint %s", path.name)


class EvaluationCallback(Callback):
    """Periodically evaluates on the validation seed pool and tracks the best model.

    "Best" is ranked by validation **success rate**, with mean reward as the
    tie-breaker. Validation cities are disjoint from training cities, so this
    selects for generalisation rather than for memorising training layouts -
    and it never touches the test pool.
    """

    def __init__(
        self,
        directory: str | Path,
        prefix: str = "dqn",
        frequency: int = 250,
        episodes: int = 30,
        pool: str = "validation",
        eval_seed: int = 777_000,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.frequency = max(1, int(frequency))
        self.episodes = int(episodes)
        self.pool = pool
        self.eval_seed = int(eval_seed)
        self.best_score: tuple[float, float] = (-np.inf, -np.inf)
        self.history: list[dict[str, Any]] = []

    def on_episode_end(self, trainer: "Trainer", episode: int, metrics: EpisodeMetrics) -> None:
        if episode % self.frequency != 0:
            return
        started = time.perf_counter()
        result = evaluate_agent(
            agent=trainer.agent,
            config=trainer.config,
            pool=self.pool,
            episodes=self.episodes,
            seed=self.eval_seed,
            deterministic=True,
        )
        summary = result.summary
        elapsed = time.perf_counter() - started

        record = {
            "episode": episode,
            "env_steps": trainer.global_step,
            "elapsed_s": round(elapsed, 2),
            **{k: v for k, v in summary.items() if k != "pool"},
        }
        self.history.append(record)
        trainer.log_evaluation(record, pool=self.pool)

        _LOG.info(
            "[eval @ ep %d] %s: success=%.1f%% collision=%.1f%% timeout=%.1f%% reward=%.2f (%.1fs)",
            episode,
            self.pool,
            100 * summary["success_rate"],
            100 * summary["collision_rate"],
            100 * summary["timeout_rate"],
            summary["mean_reward"],
            elapsed,
        )

        score = (float(summary["success_rate"]), float(summary["mean_reward"]))
        if score > self.best_score:
            self.best_score = score
            path = self.directory / f"{self.prefix}_best.pt"
            trainer.save_checkpoint(path, episode=episode, extra={"validation": record})
            _LOG.info(
                "New best model (success=%.1f%%, reward=%.2f) -> %s",
                100 * score[0],
                score[1],
                path.name,
            )
        # Restore training mode; evaluate_agent switched the network to eval.
        trainer.agent.train_mode()


class ProgressCallback(Callback):
    """Logs a rolling summary of recent training episodes to the console."""

    def __init__(self, frequency: int = 10, window: int = 50) -> None:
        self.frequency = max(1, int(frequency))
        self.window = max(1, int(window))
        self._recent: list[EpisodeMetrics] = []
        self._started = time.perf_counter()

    def on_training_start(self, trainer: "Trainer") -> None:
        self._started = time.perf_counter()

    def on_episode_end(self, trainer: "Trainer", episode: int, metrics: EpisodeMetrics) -> None:
        self._recent.append(metrics)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if episode % self.frequency != 0:
            return

        stats = aggregate_metrics(self._recent)
        elapsed = time.perf_counter() - self._started
        steps_per_second = trainer.global_step / max(elapsed, 1e-6)
        _LOG.info(
            "ep %5d | steps %8d | R %8.2f | succ %5.1f%% | coll %5.1f%% | "
            "to %5.1f%% | eps %.3f | loss %s | %.0f sps",
            episode,
            trainer.global_step,
            stats["mean_reward"],
            100 * stats["success_rate"],
            100 * stats["collision_rate"],
            100 * stats["timeout_rate"],
            trainer.agent.diagnostics.get("epsilon", float("nan")),
            _format_loss(trainer.agent.diagnostics.get("loss")),
            steps_per_second,
        )


def _format_loss(value: float | None) -> str:
    """Render a possibly-NaN loss compactly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "  n/a "
    return f"{value:6.4f}"
