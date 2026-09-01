"""Result plotting (Phase 13).

Produces the training-curve set, the train-vs-unseen generalisation comparison
and 3D/2D trajectory figures. Matplotlib is used with the ``Agg`` backend so
these run headless on a training machine.

Training curves are plotted as a rolling mean over a configurable window with
the raw per-episode series faded behind them; RL episode returns are noisy
enough that a raw-only plot is unreadable, and a smoothed-only plot hides the
variance that matters when judging stability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from drone_rl.utils.logging import get_logger  # noqa: E402

__all__ = [
    "plot_training_curves",
    "plot_generalisation_comparison",
    "plot_trajectories_3d",
    "plot_trajectory_topdown",
    "plot_learning_stages",
    "rolling_mean",
]

_LOG = get_logger(__name__)

# Consistent styling across every figure.
_PRIMARY = "#2A6FDB"
_SECONDARY = "#E8833A"
_SUCCESS = "#2BA84A"
_DANGER = "#D64550"
_MUTED = "#9AA3AE"
_FIGSIZE = (8.0, 4.6)
_DPI = 130


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    """Centred-trailing rolling mean that keeps the series length.

    Uses a minimum period of 1 so early episodes are still plotted rather than
    leaving a gap at the start of every curve.
    """
    series = pd.Series(np.asarray(values, dtype=np.float64))
    return series.rolling(window=max(1, int(window)), min_periods=1).mean().to_numpy()


def _finish(fig, ax, title: str, xlabel: str, ylabel: str, path: Path) -> Path:
    """Apply shared styling and save a figure."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def _curve(
    metrics: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    window: int,
    color: str = _PRIMARY,
    as_percentage: bool = False,
) -> Path | None:
    """Plot one metric against episode with a rolling mean overlay."""
    if column not in metrics.columns:
        _LOG.warning("Column %r missing from metrics; skipping %s", column, output_path.name)
        return None
    values = pd.to_numeric(metrics[column], errors="coerce").astype(float)
    if as_percentage:
        values = values * 100.0
    episodes = metrics["episode"].to_numpy()

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(episodes, values, color=_MUTED, alpha=0.30, linewidth=0.8, label="per episode")
    ax.plot(
        episodes,
        rolling_mean(values.to_numpy(), window),
        color=color,
        linewidth=2.0,
        label=f"rolling mean ({window})",
    )
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, title, "Episode", ylabel, output_path)


def plot_training_curves(
    metrics_csv: str | Path,
    output_dir: str | Path,
    window: int = 50,
) -> list[Path]:
    """Generate the Phase-13 training curve set from ``metrics.csv``.

    Produces reward, success rate, collision rate, energy, flight time, path
    length, steps, epsilon and loss curves.
    """
    metrics = pd.read_csv(metrics_csv)
    if metrics.empty:
        _LOG.warning("Metrics file %s is empty; no curves produced.", metrics_csv)
        return []
    output_dir = Path(output_dir)

    specs = [
        ("reward", "Episode reward", "Reward", "reward_vs_episode.png", _PRIMARY, False),
        ("success", "Success rate", "Success rate (%)", "success_rate_vs_episode.png", _SUCCESS, True),
        (
            "collision",
            "Collision rate",
            "Collision rate (%)",
            "collision_rate_vs_episode.png",
            _DANGER,
            True,
        ),
        ("energy", "Energy consumed", "Battery units", "energy_vs_episode.png", _SECONDARY, False),
        (
            "flight_time",
            "Flight time",
            "Simulated seconds",
            "flight_time_vs_episode.png",
            _PRIMARY,
            False,
        ),
        (
            "path_length",
            "Path length",
            "Metres flown",
            "path_length_vs_episode.png",
            _SECONDARY,
            False,
        ),
        ("steps", "Episode length", "Steps", "steps_vs_episode.png", _PRIMARY, False),
        ("epsilon", "Exploration rate", "Epsilon", "epsilon_vs_episode.png", _MUTED, False),
        ("loss", "TD loss", "Smooth L1 loss", "loss_vs_episode.png", _DANGER, False),
        ("q_mean", "Mean predicted Q", "Q value", "q_value_vs_episode.png", _PRIMARY, False),
    ]

    written: list[Path] = []
    for column, title, ylabel, filename, color, as_pct in specs:
        path = _curve(
            metrics, column, title, ylabel, output_dir / filename, window, color, as_pct
        )
        if path is not None:
            written.append(path)

    written.append(_plot_outcome_composition(metrics, output_dir / "outcome_composition.png", window))
    return written


def _plot_outcome_composition(metrics: pd.DataFrame, output_path: Path, window: int) -> Path:
    """Stacked view of how episodes ended, over training.

    Makes failure-mode shifts visible: e.g. collisions falling while timeouts
    rise means the agent learned to avoid buildings before it learned to arrive.
    """
    episodes = metrics["episode"].to_numpy()
    outcomes = [
        ("success", "Reached target", _SUCCESS),
        ("collision", "Collision", _DANGER),
        ("timeout", "Timeout", _MUTED),
        ("battery_empty", "Battery empty", _SECONDARY),
        ("out_of_bounds", "Out of bounds", "#7B5EA7"),
    ]
    series = []
    labels = []
    colors = []
    for column, label, color in outcomes:
        if column not in metrics.columns:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").fillna(0).astype(float)
        series.append(rolling_mean(values.to_numpy(), window) * 100.0)
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.stackplot(episodes, *series, labels=labels, colors=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    return _finish(
        fig,
        ax,
        f"Episode outcomes (rolling {window})",
        "Episode",
        "Share of episodes (%)",
        output_path,
    )


def plot_generalisation_comparison(
    summaries: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    metrics: Sequence[str] = ("success_rate", "collision_rate", "timeout_rate"),
    title: str = "Training cities vs unseen cities",
) -> Path:
    """Grouped bar chart of pool summaries (Phase 12/13).

    ``summaries`` maps pool name -> aggregated stats from
    :func:`~drone_rl.evaluation.metrics.aggregate_metrics`.
    """
    pools = list(summaries)
    pretty = {
        "success_rate": "Success",
        "collision_rate": "Collision",
        "timeout_rate": "Timeout",
        "battery_failure_rate": "Battery empty",
        "out_of_bounds_rate": "Out of bounds",
    }
    palette = {
        "success_rate": _SUCCESS,
        "collision_rate": _DANGER,
        "timeout_rate": _MUTED,
        "battery_failure_rate": _SECONDARY,
        "out_of_bounds_rate": "#7B5EA7",
    }

    x_positions = np.arange(len(pools), dtype=float)
    bar_width = 0.8 / max(1, len(metrics))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for index, metric in enumerate(metrics):
        values = [100.0 * float(summaries[p].get(metric, 0.0)) for p in pools]
        offset = (index - (len(metrics) - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=pretty.get(metric, metric),
            color=palette.get(metric, _PRIMARY),
            alpha=0.9,
        )
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p}\ncities" for p in pools])
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, title, "", "Share of episodes (%)", Path(output_path))


def plot_trajectories_3d(
    trajectories: Sequence[np.ndarray],
    buildings: Sequence[Any],
    start: np.ndarray | None = None,
    target: np.ndarray | None = None,
    output_path: str | Path = "trajectory_3d.png",
    title: str = "Flown trajectory",
) -> Path:
    """3D plot of flown paths over the city footprint."""
    fig = plt.figure(figsize=(8.5, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    for building in buildings:
        _draw_box_3d(ax, building)

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(trajectories))))
    for index, path in enumerate(trajectories):
        path = np.asarray(path)
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color=colors[index], linewidth=1.8, alpha=0.95,
        )

    if start is not None:
        ax.scatter(*np.asarray(start), color=_SECONDARY, s=70, marker="o", label="Start")
    if target is not None:
        ax.scatter(*np.asarray(target), color=_SUCCESS, s=110, marker="*", label="Target")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    if start is not None or target is not None:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=_DPI)
    plt.close(fig)
    return path_out


def _draw_box_3d(ax, building) -> None:
    """Render one building as a translucent extruded footprint."""
    min_corner = np.asarray(building.min_corner)
    max_corner = np.asarray(building.max_corner)
    xs = [min_corner[0], max_corner[0]]
    ys = [min_corner[1], max_corner[1]]
    top = max_corner[2]

    # Four vertical edges plus the roof outline is enough to read the shape
    # without the cost of full polygon rendering for hundreds of buildings.
    for x in xs:
        for y in ys:
            ax.plot([x, x], [y, y], [0, top], color=_MUTED, linewidth=0.6, alpha=0.55)
    roof_x = [xs[0], xs[1], xs[1], xs[0], xs[0]]
    roof_y = [ys[0], ys[0], ys[1], ys[1], ys[0]]
    ax.plot(roof_x, roof_y, [top] * 5, color=_MUTED, linewidth=0.8, alpha=0.75)


def plot_trajectory_topdown(
    trajectories: Sequence[np.ndarray],
    buildings: Sequence[Any],
    start: np.ndarray | None = None,
    target: np.ndarray | None = None,
    output_path: str | Path = "trajectory_topdown.png",
    title: str = "Trajectory (top-down)",
    labels: Sequence[str] | None = None,
) -> Path:
    """Top-down plot of flown paths over building footprints."""
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    for building in buildings:
        min_corner = building.min_corner
        ax.add_patch(
            plt.Rectangle(
                (min_corner[0], min_corner[1]),
                2 * building.half_extent_x,
                2 * building.half_extent_y,
                facecolor="#C9CFD6",
                edgecolor="#8B939C",
                linewidth=0.5,
            )
        )

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(trajectories))))
    for index, path in enumerate(trajectories):
        path = np.asarray(path)
        label = labels[index] if labels is not None and index < len(labels) else None
        ax.plot(path[:, 0], path[:, 1], color=colors[index], linewidth=2.0, label=label)
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[index], s=28, zorder=5)

    if start is not None:
        ax.scatter(start[0], start[1], color=_SECONDARY, s=90, marker="o",
                   edgecolor="white", zorder=6, label="Start")
    if target is not None:
        ax.scatter(target[0], target[1], color=_SUCCESS, s=180, marker="*",
                   edgecolor="white", zorder=6, label="Target")

    ax.set_aspect("equal")
    if labels is not None or start is not None:
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    return _finish(fig, ax, title, "x (m)", "y (m)", Path(output_path))


def plot_learning_stages(
    stage_summaries: Mapping[str, Mapping[str, float]],
    output_path: str | Path,
    title: str = "Learning progression",
) -> Path:
    """Bar chart of success/collision rate across training checkpoints.

    This is the quantitative counterpart to the Phase-11 visual comparison.
    """
    stages = list(stage_summaries)
    x_positions = np.arange(len(stages), dtype=float)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(stages)), 4.6))
    success = [100.0 * float(stage_summaries[s].get("success_rate", 0.0)) for s in stages]
    collision = [100.0 * float(stage_summaries[s].get("collision_rate", 0.0)) for s in stages]

    bars_a = ax.bar(x_positions - 0.19, success, width=0.38, color=_SUCCESS, label="Success")
    bars_b = ax.bar(x_positions + 0.19, collision, width=0.38, color=_DANGER, label="Collision")
    ax.bar_label(bars_a, fmt="%.0f%%", fontsize=8, padding=2)
    ax.bar_label(bars_b, fmt="%.0f%%", fontsize=8, padding=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, title, "", "Share of episodes (%)", Path(output_path))
