"""Random-action baseline (Phase 7, Steps 10-11).

Runs a uniform-random policy for a number of episodes and reports the metric
table. This validates the environment end to end before any learning code is
trusted: if collisions, battery depletion, timeouts and the (rare) accidental
success all appear here, the termination logic and reward wiring are working.

Its numbers are also the honest floor that DQN must beat.

Examples
--------
Headless, 200 episodes::

    python scripts/random_agent.py --episodes 200

Watch it fly in the 3D GUI::

    python scripts/random_agent.py --episodes 3 --gui --debug
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path


from drone_rl.agents.random_agent import RandomAgent
from drone_rl.environment.drone_env import DroneCityEnv
from drone_rl.evaluation.evaluate import rollout_episode
from drone_rl.evaluation.metrics import aggregate_metrics
from drone_rl.utils.config import load_config
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("random_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024, help="Base episode seed")
    parser.add_argument("--pool", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--gui", action="store_true", help="Render in a PyBullet window")
    parser.add_argument("--show-rays", action="store_true", help="Draw sensor rays (GUI only)")
    parser.add_argument("--debug", action="store_true", help="Print per-step reward components")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="GUI pacing multiplier; higher is slower")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/random_agent.json"))
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging("INFO")

    config = load_config(args.config, overrides=args.overrides)
    if args.debug:
        config.setdefault("environment", {})["debug"] = True

    render_config = dict(config.get("render", {}))
    render_config["show_rays"] = args.show_rays
    render_config.pop("mode", None)

    env = DroneCityEnv(
        config=config,
        seed_pool=args.pool,
        render_mode="human" if args.gui else None,
        render_config=render_config if args.gui else None,
    )
    agent = RandomAgent(num_actions=int(env.action_space.n), seed=args.seed)

    _LOG.info(
        "Random baseline: %d episodes on '%s' cities (obs=%d, actions=%d)",
        args.episodes, args.pool, env.observation_space.shape[0], env.action_space.n,
    )

    step_delay = 0.0
    if args.gui:
        step_delay = env.drone_config.dt / max(args.realtime, 1e-3)

    def pace(*_ignored) -> None:
        """Slow the GUI to roughly real time so the flight is watchable."""
        if step_delay > 0:
            time.sleep(step_delay)

    episodes = []
    started = time.perf_counter()
    try:
        for index in range(args.episodes):
            metrics, _ = rollout_episode(
                env,
                agent,
                seed=args.seed + index,
                deterministic=False,
                step_callback=pace if args.gui else None,
            )
            metrics.episode = index
            episodes.append(metrics)
            if args.gui or args.episodes <= 20:
                _LOG.info(
                    "episode %3d | city %4d | %-14s | reward %8.2f | steps %3d | "
                    "energy %5.1f | final distance %5.1f m",
                    index, metrics.city_seed, metrics.status, metrics.reward,
                    metrics.steps, metrics.energy, metrics.final_distance,
                )
            elif (index + 1) % 25 == 0:
                _LOG.info("... %d/%d episodes", index + 1, args.episodes)
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    summary = aggregate_metrics(episodes)
    total_steps = int(sum(m.steps for m in episodes))

    _LOG.info("=" * 62)
    _LOG.info("RANDOM AGENT BASELINE  (%d episodes on '%s' cities)", args.episodes, args.pool)
    _LOG.info("=" * 62)
    _LOG.info("Success rate        : %6.2f %%", 100 * summary["success_rate"])
    _LOG.info("Collision rate      : %6.2f %%", 100 * summary["collision_rate"])
    _LOG.info("Battery failure rate: %6.2f %%", 100 * summary["battery_failure_rate"])
    _LOG.info("Out-of-bounds rate  : %6.2f %%", 100 * summary["out_of_bounds_rate"])
    _LOG.info("Timeout rate        : %6.2f %%", 100 * summary["timeout_rate"])
    _LOG.info("Average reward      : %8.2f (sd %.2f)", summary["mean_reward"], summary["std_reward"])
    _LOG.info("Average steps       : %8.2f", summary["mean_steps"])
    _LOG.info("Average energy      : %8.2f battery units", summary["mean_energy"])
    _LOG.info("Average path length : %8.2f m", summary["mean_path_length"])
    _LOG.info("Average flight time : %8.2f simulated s", summary["mean_flight_time"])
    _LOG.info("Average final dist. : %8.2f m", summary["mean_final_distance"])
    _LOG.info("-" * 62)
    _LOG.info("Wall clock: %.1f s for %d env steps (%.0f steps/s)",
              elapsed, total_steps, total_steps / max(elapsed, 1e-6))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": "random",
        "episodes": args.episodes,
        "pool": args.pool,
        "seed": args.seed,
        "summary": summary,
        "per_episode": [m.as_row() for m in episodes],
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    _LOG.info("Saved results to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
