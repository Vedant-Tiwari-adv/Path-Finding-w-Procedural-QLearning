"""Evaluate a checkpoint on training and unseen cities (Phase 12).

Reports the full metric table per city-seed pool. The point of running several
pools is generalisation: ``train`` cities were seen during training, ``test``
cities never were, and the gap between them is the research-relevant number.

Every pool is evaluated with the same evaluation seed, so all pools and all
algorithms see an identically generated sequence of scenarios.

Examples
--------
Standard protocol::

    python scripts/evaluate.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --episodes 100 --seed 1000

One pool only::

    python scripts/evaluate.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --pools test --episodes 200

Compare a trained policy against the random baseline::

    python scripts/evaluate.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --with-random
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.environment.drone_env import DroneCityEnv
from drone_rl.evaluation.evaluate import evaluate_pools, format_summary_table
from drone_rl.utils.config import load_config
from drone_rl.utils.device import describe_device, resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--random", action="store_true", help="Evaluate a random policy")
    parser.add_argument("--with-random", action="store_true",
                        help="Also evaluate the random baseline for reference")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluation seed (default: fairness.evaluation_seed)")
    parser.add_argument("--pools", nargs="*", default=None,
                        choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--tag", default=None, help="Filename prefix for the results")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def load_agent(args: argparse.Namespace, config: dict, num_actions: int):
    """Return ``(agent, label, config)``, preferring the checkpoint's own config."""
    if args.random or args.checkpoint is None:
        return RandomAgent(num_actions=num_actions, seed=0), "random", config

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = agent_from_checkpoint(payload, device=device)
    agent.eval_mode()

    stored = payload.get("config")
    if stored:
        # Evaluating under different dynamics than training would invalidate
        # the comparison, so the checkpoint's environment config wins.
        merged = dict(stored)
        merged["evaluation"] = config.get("evaluation", {})
        merged["fairness"] = config.get("fairness", {})
        config = merged
    label = f"{payload.get('algorithm', 'agent')}_ep{payload.get('episode', 0)}"
    _LOG.info("Loaded %s (episode %s) from %s",
              payload.get("algorithm"), payload.get("episode"), args.checkpoint.name)
    return agent, label, config


def run_evaluation(agent, label: str, config: dict, args: argparse.Namespace,
                   pools: list[str], seed: int) -> dict:
    """Evaluate one policy across pools and persist per-episode rows."""
    results = evaluate_pools(
        agent=agent,
        config=config,
        pools=pools,
        episodes=args.episodes,
        seed=seed,
        deterministic=True,
        record_trajectories=not args.no_trajectories,
        max_trajectories=int(config.get("evaluation", {}).get("max_trajectories", 20)),
        progress=True,
    )

    print()
    print(f"  Policy: {label}   ({args.episodes} episodes per pool, eval seed {seed})")
    print(format_summary_table(results))
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pool, result in results.items():
        for row in result.rows():
            row["pool"] = pool
            row["policy"] = label
            rows.append(row)
    csv_path = args.output_dir / f"{label}_episodes.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _LOG.info("Per-episode rows -> %s", csv_path)

    # Trajectories are stored as .npz so plot_results.py can draw real paths.
    if not args.no_trajectories:
        trajectory_dir = Path("outputs/trajectories")
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for pool, result in results.items():
            if not result.trajectories:
                continue
            np.savez_compressed(
                trajectory_dir / f"{label}_{pool}.npz",
                **{f"traj_{i}": t for i, t in enumerate(result.trajectories)},
                scenarios=json.dumps(result.scenarios),
            )

    return {pool: result.summary for pool, result in results.items()}


def main() -> int:
    args = parse_args()
    configure_logging("INFO")

    config = load_config(args.config, overrides=args.overrides)
    evaluation_block = config.get("evaluation", {})
    fairness = config.get("fairness", {})

    pools = args.pools or list(evaluation_block.get("pools", ["train", "validation", "test"]))
    seed = args.seed if args.seed is not None else int(fairness.get("evaluation_seed", 12345))

    probe = DroneCityEnv(config=config, seed_pool=pools[0])
    num_actions = int(probe.action_space.n)
    probe.close()

    agent, label, config = load_agent(args, config, num_actions)
    if args.tag:
        label = args.tag

    _LOG.info("=" * 72)
    _LOG.info("EVALUATION | pools=%s | episodes=%d | seed=%d", pools, args.episodes, seed)
    _LOG.info("=" * 72)

    summaries = {label: run_evaluation(agent, label, config, args, pools, seed)}

    if args.with_random and not args.random:
        _LOG.info("Evaluating the random baseline for reference...")
        summaries["random"] = run_evaluation(
            RandomAgent(num_actions=num_actions, seed=0), "random", config, args, pools, seed
        )

    # ---- generalisation gap ---------------------------------------------
    if "train" in pools and "test" in pools:
        _LOG.info("-" * 72)
        _LOG.info("GENERALISATION (train cities vs unseen test cities)")
        for policy, per_pool in summaries.items():
            train_rate = per_pool["train"]["success_rate"]
            test_rate = per_pool["test"]["success_rate"]
            _LOG.info(
                "  %-18s train %5.1f%%  ->  unseen %5.1f%%   (gap %+.1f pp)",
                policy, 100 * train_rate, 100 * test_rate,
                100 * (test_rate - train_rate),
            )
        _LOG.info("-" * 72)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "episodes_per_pool": args.episodes,
        "evaluation_seed": seed,
        "pools": pools,
        "device": describe_device(resolve_device(args.device)).as_dict(),
        "fairness": fairness,
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{label}_summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    _LOG.info("Saved summary to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
