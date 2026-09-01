"""Episode metrics and their aggregation (Phase 12).

One :class:`EpisodeMetrics` per episode; :func:`aggregate_metrics` reduces a
list of them to the report table. Rates are computed over *all* episodes, while
"time to reach" style averages (flight time, path length, path efficiency) are
reported both overall and restricted to successful episodes - averaging flight
time over failures is misleading, since a crash 5 steps in looks like a fast
flight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["EpisodeMetrics", "aggregate_metrics", "METRIC_FIELDS", "summarise_pool"]

#: Column order used by CSV logs and result tables.
METRIC_FIELDS: tuple[str, ...] = (
    "episode",
    "reward",
    "success",
    "collision",
    "battery_empty",
    "out_of_bounds",
    "timeout",
    "steps",
    "energy",
    "battery_remaining",
    "path_length",
    "flight_time",
    "final_distance",
    "straight_line_distance",
    "path_efficiency",
    "city_seed",
    "seed_pool",
    "epsilon",
    "loss",
    "q_mean",
)


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode: int = 0
    reward: float = 0.0
    success: bool = False
    collision: bool = False
    battery_empty: bool = False
    out_of_bounds: bool = False
    timeout: bool = False
    steps: int = 0
    energy: float = 0.0
    battery_remaining: float = 0.0
    path_length: float = 0.0
    flight_time: float = 0.0
    final_distance: float = 0.0
    straight_line_distance: float = 0.0
    path_efficiency: float = 0.0
    city_seed: int = -1
    seed_pool: str = ""
    status: str = ""
    epsilon: float = float("nan")
    loss: float = float("nan")
    q_mean: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any], episode: int = 0) -> "EpisodeMetrics":
        """Build from :meth:`DroneCityEnv.episode_summary`."""
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        values = {k: v for k, v in summary.items() if k in known}
        values["episode"] = episode
        return cls(**values)

    def as_row(self) -> dict[str, Any]:
        """Flat dict suitable for a CSV row."""
        row = asdict(self)
        extra = row.pop("extra", {}) or {}
        row.update(extra)
        return row


def _mean(values: Sequence[float]) -> float:
    """Mean of ``values``, or NaN when empty."""
    return float(np.mean(values)) if len(values) else float("nan")


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) else float("nan")


def aggregate_metrics(episodes: Iterable[EpisodeMetrics]) -> dict[str, float]:
    """Reduce per-episode metrics to the summary statistics of Phase 12.

    Returns rates in ``[0, 1]`` plus means/standard deviations. Success-only
    variants carry the ``_success`` suffix.
    """
    records = list(episodes)
    count = len(records)
    if count == 0:
        return {"episodes": 0}

    successes = [m for m in records if m.success]
    rewards = [m.reward for m in records]
    steps = [float(m.steps) for m in records]
    energies = [m.energy for m in records]
    path_lengths = [m.path_length for m in records]
    flight_times = [m.flight_time for m in records]
    final_distances = [m.final_distance for m in records]

    summary: dict[str, float] = {
        "episodes": float(count),
        "success_rate": float(sum(m.success for m in records) / count),
        "collision_rate": float(sum(m.collision for m in records) / count),
        "battery_failure_rate": float(sum(m.battery_empty for m in records) / count),
        "out_of_bounds_rate": float(sum(m.out_of_bounds for m in records) / count),
        "timeout_rate": float(sum(m.timeout for m in records) / count),
        "mean_reward": _mean(rewards),
        "std_reward": _std(rewards),
        "mean_steps": _mean(steps),
        "mean_energy": _mean(energies),
        "mean_battery_used_pct": _mean(
            [
                100.0 * m.energy / max(m.energy + m.battery_remaining, 1e-6)
                for m in records
            ]
        ),
        "mean_path_length": _mean(path_lengths),
        "mean_flight_time": _mean(flight_times),
        "mean_final_distance": _mean(final_distances),
        "mean_path_efficiency": _mean([m.path_efficiency for m in records]),
    }

    # Success-conditioned averages: what a *completed* delivery costs.
    summary.update(
        {
            "mean_steps_success": _mean([float(m.steps) for m in successes]),
            "mean_energy_success": _mean([m.energy for m in successes]),
            "mean_flight_time_success": _mean([m.flight_time for m in successes]),
            "mean_path_length_success": _mean([m.path_length for m in successes]),
            "mean_path_efficiency_success": _mean([m.path_efficiency for m in successes]),
            "mean_reward_success": _mean([m.reward for m in successes]),
        }
    )
    return summary


def summarise_pool(
    pool: str,
    episodes: Iterable[EpisodeMetrics],
) -> dict[str, Any]:
    """Aggregate one seed pool's episodes and tag the result with its name."""
    summary = aggregate_metrics(episodes)
    summary["pool"] = pool
    return summary
