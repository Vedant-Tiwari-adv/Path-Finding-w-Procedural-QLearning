"""Visualise the learning progression (Phase 11).

Replays several training stages - random, early, mid, final - through the
*identical* scenario: same city seed, same start, same target, same initial
heading. Anything you see differ between stages is therefore attributable to
the policy and nothing else.

This is the classroom demonstration: early policies crash and wander, later
policies route around buildings and arrive.

Examples
--------
Auto-discover checkpoints and show them in a window::

    python scripts/compare_learning.py

Record each stage to its own clip, plus a side-by-side trajectory figure::

    python scripts/compare_learning.py --record-dir outputs/videos --headless

Pick the stages explicitly::

    python scripts/compare_learning.py \\
        --checkpoints outputs/checkpoints/dqn_episode_250.pt \\
                      outputs/checkpoints/dqn_episode_2000.pt \\
                      outputs/checkpoints/dqn_best.pt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.environment.drone_env import DroneCityEnv
from drone_rl.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from drone_rl.utils.config import load_config
from drone_rl.utils.device import resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("compare_learning")

_EPISODE_PATTERN = re.compile(r"_episode_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None,
                        help="Explicit stage checkpoints, in order")
    parser.add_argument("--stages", type=int, default=3,
                        help="Number of training checkpoints to auto-select")
    parser.add_argument("--no-random", action="store_true",
                        help="Omit the untrained random-policy stage")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Scenarios to replay through every stage")
    parser.add_argument("--pool", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=4242,
                        help="Fixes the scenario; identical for every stage")
    parser.add_argument("--city-seed", type=int, default=None)
    parser.add_argument("--realtime", type=float, default=1.0)
    parser.add_argument("--show-rays", action="store_true")
    parser.add_argument("--camera-mode", default="orbit",
                        choices=["free", "follow", "orbit", "cinematic"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=None,
                        help="Write one clip per stage into this directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/evaluation/learning_progression.json"))
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def discover_checkpoints(directory: Path, count: int) -> list[Path]:
    """Pick ``count`` evenly spaced ``*_episode_N.pt`` files, plus the best one.

    Spacing them across the run is what makes the progression legible: three
    consecutive late checkpoints would all look the same.
    """
    if not directory.is_dir():
        return []
    numbered: list[tuple[int, Path]] = []
    for path in directory.glob("*_episode_*.pt"):
        match = _EPISODE_PATTERN.search(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])

    selected: list[Path] = []
    if numbered and count > 0:
        # Evenly spaced indices across the sorted checkpoints, excluding the
        # very last (the "best"/"final" model represents that end already).
        indices = np.linspace(0, len(numbered) - 1, num=min(count, len(numbered)))
        for index in sorted({int(round(i)) for i in indices}):
            selected.append(numbered[index][1])

    for final_name in ("dqn_best.pt", "dqn_final.pt"):
        candidate = directory / final_name
        if candidate.is_file() and candidate not in selected:
            selected.append(candidate)
            break
    return selected


def stage_label(path: Path | None) -> str:
    """Human-readable name for a stage."""
    if path is None:
        return "Random policy (untrained)"
    match = _EPISODE_PATTERN.search(path.name)
    if match:
        return f"DQN @ episode {int(match.group(1))}"
    if "best" in path.stem:
        return "DQN (best validation)"
    if "final" in path.stem:
        return "DQN (final)"
    return path.stem


def load_stage_agent(path: Path | None, num_actions: int, device_name: str, seed: int):
    """Build the agent for one stage."""
    if path is None:
        return RandomAgent(num_actions=num_actions, seed=seed)
    device = resolve_device(device_name)
    payload = torch.load(path, map_location=device, weights_only=False)
    agent = agent_from_checkpoint(payload, device=device)
    agent.eval_mode()
    return agent


def build_scenarios(config: dict, pool: str, seed: int, count: int,
                    city_seed: int | None) -> list[dict]:
    """Sample the scenarios once, so every stage replays exactly these.

    Freezing city seed, start, target and initial yaw is what makes the
    comparison fair - the stages differ only in their policy.
    """
    probe = DroneCityEnv(config=config, seed_pool=pool)
    scenarios = []
    try:
        for index in range(count):
            options = {"city_seed": city_seed} if city_seed is not None else {}
            probe.reset(seed=seed + index, options=options)
            scenarios.append(
                {
                    "city_seed": int(probe.city.seed),
                    "start": probe.start_position.copy(),
                    "target": probe.target_position.copy(),
                    "yaw": float(probe.drone.state.yaw),
                }
            )
    finally:
        probe.close()
    return scenarios


def main() -> int:
    args = parse_args()
    configure_logging("INFO")
    config = load_config(args.config, overrides=args.overrides)

    checkpoints: list[Path | None]
    if args.checkpoints:
        checkpoints = list(args.checkpoints)
    else:
        checkpoints = list(discover_checkpoints(args.checkpoint_dir, args.stages))
        if not checkpoints:
            _LOG.warning(
                "No checkpoints found in %s - showing the random policy only. "
                "Train first: python scripts/train.py --config configs/dqn.yaml",
                args.checkpoint_dir,
            )
    if not args.no_random:
        checkpoints = [None, *checkpoints]

    render_config = dict(config.get("render", {}))
    render_config.pop("mode", None)
    render_config["show_rays"] = args.show_rays
    render_config["camera_mode"] = args.camera_mode

    probe = DroneCityEnv(config=config, seed_pool=args.pool)
    num_actions = int(probe.action_space.n)
    probe.close()

    scenarios = build_scenarios(config, args.pool, args.seed, args.episodes, args.city_seed)
    _LOG.info("=" * 74)
    _LOG.info("LEARNING PROGRESSION - %d stages x %d identical scenario(s)",
              len(checkpoints), len(scenarios))
    for scenario in scenarios:
        _LOG.info("  city seed %d | start %s -> target %s",
                  scenario["city_seed"],
                  np.round(scenario["start"], 1), np.round(scenario["target"], 1))
    _LOG.info("=" * 74)

    env = DroneCityEnv(
        config=config,
        seed_pool=args.pool,
        render_mode="rgb_array" if args.headless else "human",
        render_config=render_config,
    )
    step_delay = env.drone_config.dt / max(args.realtime, 1e-3) if not args.headless else 0.0

    results: dict[str, dict] = {}
    trajectories: dict[str, np.ndarray] = {}
    try:
        for checkpoint in checkpoints:
            label = stage_label(checkpoint)
            agent = load_stage_agent(checkpoint, num_actions, args.device, args.seed)

            recorder = None
            if args.record_dir is not None:
                from drone_rl.evaluation.video import VideoRecorder

                env.reset(seed=args.seed)  # ensure the renderer exists
                recorder = VideoRecorder(renderer=env.renderer)

            stage_metrics: list[EpisodeMetrics] = []
            for index, scenario in enumerate(scenarios):
                observation, info = env.reset(
                    seed=args.seed + index,
                    options={
                        "city_seed": scenario["city_seed"],
                        "start": scenario["start"],
                        "target": scenario["target"],
                        "yaw": scenario["yaw"],
                    },
                )
                if env.renderer is not None:
                    env.renderer.clear_dynamic()
                    env.renderer.set_markers(
                        env.start_position, env.target_position, env.env_config.target_radius
                    )

                terminated = truncated = False
                while not (terminated or truncated):
                    action = agent.act(observation, evaluation=True)
                    observation, _r, terminated, truncated, info = env.step(action)
                    if env.renderer is not None:
                        env.renderer.set_hud(
                            [
                                f"STAGE: {label}",
                                f"Scenario {index + 1}/{len(scenarios)}  "
                                f"city {info['city_seed']}",
                                f"Step     {info['step']}/{env.env_config.max_steps}",
                                f"Reward   {info['episode_reward']:+8.2f}",
                                f"Battery  {info['battery_fraction'] * 100:5.1f}%",
                                f"Distance {info['distance_to_target']:6.2f} m",
                                f"Status   {info['status']}",
                            ],
                            position=env.drone.state.position + np.array([0.0, 0.0, 6.0]),
                        )
                    if recorder is not None:
                        recorder.capture()
                    if step_delay > 0:
                        time.sleep(step_delay)

                summary = env.episode_summary()
                metrics = EpisodeMetrics.from_summary(summary, episode=index)
                metrics.status = summary["status"]
                stage_metrics.append(metrics)
                if index == 0:
                    trajectories[label] = env.trajectory_array()
                if recorder is not None:
                    recorder.hold(int(args.fps * 0.8))

                _LOG.info("  %-28s scenario %d -> %-14s reward %8.2f  steps %3d",
                          label, index + 1, summary["status"], summary["reward"],
                          summary["steps"])

            results[label] = aggregate_metrics(stage_metrics)
            if recorder is not None:
                args.record_dir.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
                recorder.save(args.record_dir / f"stage_{safe}.mp4", fps=args.fps)
            if not args.headless:
                time.sleep(1.2)
    except KeyboardInterrupt:
        _LOG.info("Interrupted by user.")
    finally:
        env.close()

    # ---- report ----------------------------------------------------------
    if results:
        _LOG.info("=" * 74)
        _LOG.info("%-30s %9s %10s %9s %10s", "stage", "success", "collision", "reward", "steps")
        _LOG.info("-" * 74)
        for label, stats in results.items():
            _LOG.info("%-30s %8.1f%% %9.1f%% %9.2f %10.1f",
                      label[:30], 100 * stats["success_rate"], 100 * stats["collision_rate"],
                      stats["mean_reward"], stats["mean_steps"])
        _LOG.info("=" * 74)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(
                {"scenarios": [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in s.items()} for s in scenarios],
                 "stages": results},
                handle, indent=2, default=str,
            )
        _LOG.info("Saved stage metrics to %s", args.output)

    # ---- figures ---------------------------------------------------------
    if trajectories:
        from drone_rl.evaluation.visualization import (
            plot_learning_stages,
            plot_trajectory_topdown,
        )

        args.figure_dir.mkdir(parents=True, exist_ok=True)
        layout = env.city
        path = plot_trajectory_topdown(
            trajectories=list(trajectories.values()),
            buildings=layout.buildings if layout else [],
            start=scenarios[0]["start"],
            target=scenarios[0]["target"],
            output_path=args.figure_dir / "learning_progression_paths.png",
            title=f"Same scenario, different training stages (city {scenarios[0]['city_seed']})",
            labels=list(trajectories.keys()),
        )
        _LOG.info("Saved %s", path)
        path = plot_learning_stages(
            results, args.figure_dir / "learning_progression_rates.png"
        )
        _LOG.info("Saved %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
