"""Evaluation pipeline, metrics and result plotting."""

from drone_rl.evaluation.evaluate import (
    EvaluationResult,
    evaluate_agent,
    evaluate_pools,
    format_summary_table,
    rollout_episode,
)
from drone_rl.evaluation.metrics import (
    METRIC_FIELDS,
    EpisodeMetrics,
    aggregate_metrics,
    summarise_pool,
)

__all__ = [
    "EvaluationResult",
    "evaluate_agent",
    "evaluate_pools",
    "format_summary_table",
    "rollout_episode",
    "METRIC_FIELDS",
    "EpisodeMetrics",
    "aggregate_metrics",
    "summarise_pool",
]
