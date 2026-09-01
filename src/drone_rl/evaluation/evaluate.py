"""Evaluation pipeline (Phase 12).

Rolls a policy out for a fixed number of episodes on one or more city-seed
pools and reports the metric table. The evaluation episode sampler is seeded
from ``fairness.evaluation_seed``, so *every* algorithm sees the identical
sequence of cities, starts and targets - the comparison differs only in the
policy.

Reporting train-pool and unseen-pool numbers side by side is the point: a large
gap indicates memorisation of training layouts rather than a transferable
navigation policy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from drone_rl.agents.base import Agent
from drone_rl.environment.drone_env import DroneCityEnv
from drone_rl.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from drone_rl.utils.logging import get_logger

__all__ = ["EvaluationResult", "evaluate_agent", "evaluate_pools", "rollout_episode"]

_LOG = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Outcome of evaluating one policy on one seed pool."""

    pool: str
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    trajectories: list[np.ndarray] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregated statistics, tagged with the pool name."""
        stats = aggregate_metrics(self.episodes)
        stats["pool"] = self.pool
        return stats

    def rows(self) -> list[dict[str, Any]]:
        """Per-episode rows for CSV export."""
        return [m.as_row() for m in self.episodes]


def rollout_episode(
    env: DroneCityEnv,
    agent: Agent,
    seed: int | None = None,
    options: Mapping[str, Any] | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
    record_trajectory: bool = False,
    step_callback=None,
) -> tuple[EpisodeMetrics, np.ndarray | None]:
    """Run one full episode and return its metrics.

    Parameters
    ----------
    deterministic:
        When ``True`` the agent acts greedily (``evaluation=True``).
    step_callback:
        Optional ``fn(step, action, obs, reward, info)`` invoked after each step,
        used by the GUI demo to update its HUD.
    """
    observation, info = env.reset(seed=seed, options=dict(options or {}))
    if deterministic:
        agent.eval_mode()
    else:
        agent.train_mode()

    step_limit = max_steps if max_steps is not None else env.env_config.max_steps
    step_index = 0
    while step_index < step_limit:
        action = agent.act(observation, evaluation=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        step_index += 1
        if step_callback is not None:
            step_callback(step_index, action, observation, reward, info)
        if terminated or truncated:
            break

    summary = env.episode_summary()
    metrics = EpisodeMetrics.from_summary(summary)
    metrics.status = summary["status"]
    trajectory = env.trajectory_array() if record_trajectory else None
    return metrics, trajectory


def evaluate_agent(
    agent: Agent,
    config: Mapping[str, Any],
    pool: str,
    episodes: int = 100,
    seed: int = 12345,
    deterministic: bool = True,
    record_trajectories: bool = False,
    max_trajectories: int = 20,
    progress: bool = False,
) -> EvaluationResult:
    """Evaluate ``agent`` for ``episodes`` episodes on the ``pool`` seed range.

    Episode seeds are ``seed + i``, so the scenario sequence is a deterministic
    function of ``seed`` alone and is reproducible across algorithms and runs.
    """
    env = DroneCityEnv(config=config, seed_pool=pool)
    result = EvaluationResult(pool=pool)

    iterator = range(int(episodes))
    # Only draw a progress bar on an interactive terminal; when output is piped
    # to a file tqdm's carriage returns turn the log into an unreadable smear.
    if progress and sys.stderr.isatty():
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"eval[{pool}]", leave=False)
        except ImportError:  # pragma: no cover - tqdm is a declared dependency
            pass

    try:
        for index in iterator:
            episode_seed = int(seed) + index
            keep_trajectory = record_trajectories and len(result.trajectories) < max_trajectories
            metrics, trajectory = rollout_episode(
                env,
                agent,
                seed=episode_seed,
                deterministic=deterministic,
                record_trajectory=keep_trajectory,
            )
            metrics.episode = index
            result.episodes.append(metrics)
            if keep_trajectory and trajectory is not None:
                result.trajectories.append(trajectory)
                result.scenarios.append(
                    {
                        "episode": index,
                        "city_seed": metrics.city_seed,
                        "start": env.start_position.tolist(),
                        "target": env.target_position.tolist(),
                        "status": metrics.status,
                        "success": bool(metrics.success),
                    }
                )
    finally:
        env.close()

    return result


def evaluate_pools(
    agent: Agent,
    config: Mapping[str, Any],
    pools: Sequence[str] = ("train", "validation", "test"),
    episodes: int = 100,
    seed: int = 12345,
    deterministic: bool = True,
    record_trajectories: bool = False,
    max_trajectories: int = 20,
    progress: bool = False,
) -> dict[str, EvaluationResult]:
    """Evaluate the same policy on several seed pools with identical scenarios.

    The same ``seed`` is used for every pool, so pool-to-pool differences come
    from the city layouts rather than from a different random draw.
    """
    results: dict[str, EvaluationResult] = {}
    for pool in pools:
        _LOG.info("Evaluating %d episodes on '%s' city seeds...", episodes, pool)
        results[pool] = evaluate_agent(
            agent=agent,
            config=config,
            pool=pool,
            episodes=episodes,
            seed=seed,
            deterministic=deterministic,
            record_trajectories=record_trajectories,
            max_trajectories=max_trajectories,
            progress=progress,
        )
    return results


#: ``(summary key, column title, value format)`` for the console report.
_TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("success_rate", "Success", "{:.1%}"),
    ("collision_rate", "Collision", "{:.1%}"),
    ("timeout_rate", "Timeout", "{:.1%}"),
    ("battery_failure_rate", "Battery", "{:.1%}"),
    ("out_of_bounds_rate", "OOB", "{:.1%}"),
    ("mean_reward", "Reward", "{:.2f}"),
    ("mean_steps", "Steps", "{:.1f}"),
    ("mean_energy", "Energy", "{:.2f}"),
    ("mean_path_length", "Path(m)", "{:.2f}"),
    ("mean_flight_time", "Time(s)", "{:.2f}"),
)

_POOL_WIDTH = 12
_CELL_WIDTH = 10


def format_summary_table(results: Mapping[str, EvaluationResult]) -> str:
    """Render pool summaries as a fixed-width console table."""
    if not results:
        return "(no results)"

    header = f"{'Pool':<{_POOL_WIDTH}}" + "".join(
        f"{title:>{_CELL_WIDTH}}" for _key, title, _fmt in _TABLE_COLUMNS
    )
    lines = [header, "-" * len(header)]
    for pool, result in results.items():
        summary = result.summary
        cells = []
        for key, _title, value_format in _TABLE_COLUMNS:
            value = summary.get(key, float("nan"))
            text = "n/a" if value is None or np.isnan(value) else value_format.format(value)
            cells.append(f"{text:>{_CELL_WIDTH}}")
        lines.append(f"{pool:<{_POOL_WIDTH}}" + "".join(cells))
    return "\n".join(lines)
