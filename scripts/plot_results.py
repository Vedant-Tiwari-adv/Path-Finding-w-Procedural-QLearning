"""Generate every results figure (Phase 13).

Reads the artefacts produced by ``train.py`` and ``evaluate.py`` and writes
PNGs to ``outputs/figures/``:

* training curves - reward, success, collision, energy, flight time, path
  length, steps, epsilon, loss, mean Q, and an outcome-composition stackplot
* the train-vs-unseen generalisation comparison
* trajectory plots (top-down and 3D) from recorded evaluation episodes

Examples
--------
    python scripts/plot_results.py
    python scripts/plot_results.py --metrics outputs/logs/dqn_baseline/metrics.csv --window 100
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from drone_rl.environment.city import CityConfig, generate_city
from drone_rl.utils.config import load_config
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("plot_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", type=Path, default=None,
                        help="metrics.csv (default: newest under outputs/logs)")
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--trajectory-dir", type=Path, default=Path("outputs/trajectories"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling-mean window for the training curves")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def find_latest_metrics(logs_dir: Path = Path("outputs/logs")) -> Path | None:
    """Return the most recently modified ``metrics.csv`` under ``logs_dir``."""
    candidates = sorted(
        logs_dir.glob("*/metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def plot_training(metrics_path: Path, output_dir: Path, window: int) -> list[Path]:
    """Training curve set."""
    from drone_rl.evaluation.visualization import plot_training_curves

    _LOG.info("Training curves from %s (window=%d)", metrics_path, window)
    written = plot_training_curves(metrics_path, output_dir, window=window)
    for path in written:
        _LOG.info("  %s", path.name)
    return written


def plot_generalisation(evaluation_dir: Path, output_dir: Path) -> list[Path]:
    """Train-vs-unseen bar charts, one per evaluated policy."""
    from drone_rl.evaluation.visualization import plot_generalisation_comparison

    written: list[Path] = []
    for summary_path in sorted(evaluation_dir.glob("*_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for policy, per_pool in payload.get("summaries", {}).items():
            if not per_pool:
                continue
            path = plot_generalisation_comparison(
                summaries=per_pool,
                output_path=output_dir / f"generalisation_{policy}.png",
                title=f"{policy}: training cities vs unseen cities",
            )
            _LOG.info("  %s", path.name)
            written.append(path)
    if not written:
        _LOG.warning(
            "No *_summary.json in %s - run scripts/evaluate.py first.", evaluation_dir
        )
    return written


def plot_trajectories(trajectory_dir: Path, output_dir: Path,
                      city_config: CityConfig) -> list[Path]:
    """Top-down and 3D plots from recorded evaluation trajectories."""
    from drone_rl.evaluation.visualization import plot_trajectories_3d, plot_trajectory_topdown

    written: list[Path] = []
    for archive_path in sorted(trajectory_dir.glob("*.npz")):
        with np.load(archive_path, allow_pickle=True) as archive:
            scenarios = json.loads(str(archive["scenarios"])) if "scenarios" in archive else []
            trajectories = [archive[k] for k in sorted(archive) if k.startswith("traj_")]
        if not trajectories or not scenarios:
            continue

        # Each recorded trajectory is from a different city, so plot the first
        # few individually rather than overlaying incompatible layouts.
        for index, (trajectory, scenario) in enumerate(zip(trajectories[:3], scenarios[:3])):
            layout = generate_city(int(scenario["city_seed"]), city_config)
            stem = f"{archive_path.stem}_ep{scenario.get('episode', index)}"
            status = scenario.get("status", "")

            path = plot_trajectory_topdown(
                trajectories=[trajectory],
                buildings=layout.buildings,
                start=np.asarray(scenario["start"]),
                target=np.asarray(scenario["target"]),
                output_path=output_dir / f"trajectory_{stem}_topdown.png",
                title=f"{archive_path.stem} - city {scenario['city_seed']} ({status})",
            )
            written.append(path)

            path = plot_trajectories_3d(
                trajectories=[trajectory],
                buildings=layout.buildings,
                start=np.asarray(scenario["start"]),
                target=np.asarray(scenario["target"]),
                output_path=output_dir / f"trajectory_{stem}_3d.png",
                title=f"{archive_path.stem} - city {scenario['city_seed']} ({status})",
            )
            written.append(path)

    for path in written:
        _LOG.info("  %s", path.name)
    if not written:
        _LOG.warning("No trajectories in %s - run scripts/evaluate.py first.", trajectory_dir)
    return written


def main() -> int:
    args = parse_args()
    configure_logging("INFO")

    config = load_config(args.config, overrides=args.overrides)
    city_config = CityConfig.from_mapping(config.get("city", {}))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    metrics_path = args.metrics or find_latest_metrics()
    if metrics_path and Path(metrics_path).is_file():
        written += plot_training(Path(metrics_path), args.output_dir, args.window)
    else:
        _LOG.warning("No metrics.csv found - run scripts/train.py first.")

    if args.evaluation_dir.is_dir():
        written += plot_generalisation(args.evaluation_dir, args.output_dir)
    if args.trajectory_dir.is_dir():
        written += plot_trajectories(args.trajectory_dir, args.output_dir, city_config)

    _LOG.info("=" * 60)
    _LOG.info("Wrote %d figures to %s", len(written), args.output_dir)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
