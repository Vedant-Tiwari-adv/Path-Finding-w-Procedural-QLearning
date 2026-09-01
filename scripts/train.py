"""Train an agent (Phase 9, Step 13).

Headless by design - no GUI is opened, so thousands of episodes run at full
speed. Use ``scripts/demo.py`` to watch a trained checkpoint fly.

Examples
--------
Full baseline run::

    python scripts/train.py --config configs/dqn.yaml

Quick sanity check (a few minutes)::

    python scripts/train.py --config configs/dqn.yaml --episodes 50 \\
        --set training.learning_starts=500 dqn.min_replay_size=500

Follow the run in TensorBoard::

    tensorboard --logdir outputs/logs
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from drone_rl.training.trainer import Trainer
from drone_rl.utils.config import load_config, save_config
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/dqn.yaml"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training.total_episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed")
    parser.add_argument("--name", default=None, help="Override experiment.name")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="Config overrides, e.g. dqn.learning_rate=1e-4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config, overrides=args.overrides)
    experiment = config.setdefault("experiment", {})
    if args.seed is not None:
        experiment["seed"] = args.seed
    if args.name is not None:
        experiment["name"] = args.name
    if args.episodes is not None:
        config.setdefault("training", {})["total_episodes"] = args.episodes

    output_dir = Path(args.output_dir or experiment.get("output_dir", "outputs"))
    log_dir = output_dir / "logs" / str(experiment.get("name", "experiment"))
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level, log_file=log_dir / "train.log")

    # Snapshot the fully resolved config next to the run for reproducibility.
    save_config(config, log_dir / "config.resolved.yaml")

    trainer = Trainer(config=config, output_dir=output_dir, device=args.device)
    metrics = trainer.train()

    if metrics:
        from drone_rl.evaluation.metrics import aggregate_metrics

        window = min(100, len(metrics))
        final = aggregate_metrics(metrics[-window:])
        first = aggregate_metrics(metrics[:window])
        _LOG.info("=" * 66)
        _LOG.info("TRAINING COMPLETE - %d episodes, %d env steps",
                  len(metrics), trainer.global_step)
        _LOG.info("%-22s %12s %12s", "metric", f"first {window}", f"last {window}")
        _LOG.info("-" * 66)
        for key, label, scale in (
            ("success_rate", "success rate (%)", 100.0),
            ("collision_rate", "collision rate (%)", 100.0),
            ("timeout_rate", "timeout rate (%)", 100.0),
            ("mean_reward", "mean reward", 1.0),
            ("mean_steps", "mean steps", 1.0),
            ("mean_energy", "mean energy", 1.0),
        ):
            _LOG.info("%-22s %12.2f %12.2f", label, scale * first[key], scale * final[key])
        _LOG.info("=" * 66)

    _LOG.info("Checkpoints : %s", trainer.checkpoint_dir)
    _LOG.info("Metrics CSV : %s", trainer.metrics_logger.csv_path)
    _LOG.info("TensorBoard : tensorboard --logdir %s", output_dir / "logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
