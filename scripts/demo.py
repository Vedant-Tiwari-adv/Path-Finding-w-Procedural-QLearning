"""Watch a trained checkpoint fly (Phase 10).

Opens a PyBullet window, loads a policy, and flies episodes in the procedural
city with a live HUD showing episode, step, reward, battery, distance and
status. Optionally records the run to MP4.

Camera
------
Arrows pan, Shift/Ctrl raise/lower, PgUp/PgDn zoom, Alt+arrows change the
viewing angle, F1-F4 pick free/follow/orbit/cinematic tracking, F5 resets.

Examples
--------
Fly the best checkpoint on unseen cities::

    python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt --pool test

Chase camera, sensor rays visible, half speed::

    python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --camera-mode orbit --show-rays --realtime 0.5

Record without opening a window::

    python scripts/demo.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --episodes 3 --record outputs/videos/final.mp4 --headless

An untrained random policy (no checkpoint)::

    python scripts/demo.py --random --episodes 3
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.environment.drone_env import DroneCityEnv, EpisodeOutcome
from drone_rl.evaluation.metrics import EpisodeMetrics, aggregate_metrics
from drone_rl.utils.config import load_config
from drone_rl.utils.device import resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("demo")

#: Marker colours for terminal events, matching the renderer palette.
_STATUS_COLORS = {
    EpisodeOutcome.TARGET_REACHED: (0.15, 0.95, 0.35),
    EpisodeOutcome.COLLISION: (1.0, 0.2, 0.15),
    EpisodeOutcome.BATTERY_EMPTY: (1.0, 0.65, 0.1),
    EpisodeOutcome.OUT_OF_BOUNDS: (0.6, 0.4, 0.9),
    EpisodeOutcome.TIMEOUT: (0.7, 0.7, 0.7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint to load (omit with --random)")
    parser.add_argument("--random", action="store_true",
                        help="Fly an untrained random policy instead")
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"),
                        help="Fallback config when a checkpoint carries none")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--pool", default="test", choices=["train", "validation", "test"],
                        help="City-seed pool; 'test' shows generalisation to unseen cities")
    parser.add_argument("--seed", type=int, default=None, help="Base episode seed")
    parser.add_argument("--city-seed", type=int, default=None, help="Pin one city layout")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="Playback pacing; <1 is slower, >1 faster")
    parser.add_argument("--show-rays", action="store_true", help="Draw the sensor rays")
    parser.add_argument("--camera-mode", default=None,
                        choices=["free", "follow", "orbit", "cinematic"])
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a window (useful with --record)")
    parser.add_argument("--record", type=Path, default=None, help="Write an MP4/GIF here")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def load_policy(args: argparse.Namespace, config: dict, num_actions: int):
    """Return ``(agent, label, config)``; the checkpoint's config wins if present.

    Using the config stored in the checkpoint matters: replaying a policy under
    different dynamics than it trained on would be misleading.
    """
    if args.random or args.checkpoint is None:
        return RandomAgent(num_actions=num_actions, seed=args.seed or 0), "Random policy", config

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = agent_from_checkpoint(payload, device=device)
    agent.eval_mode()

    stored_config = payload.get("config")
    if stored_config:
        config = stored_config
    label = f"{payload.get('algorithm', 'agent')} @ episode {payload.get('episode', '?')}"
    _LOG.info("Loaded %s from %s", label, args.checkpoint.name)
    return agent, label, config


def build_hud(label: str, episode: int, total: int, info: dict, env: DroneCityEnv) -> list[str]:
    """Compose the on-screen overlay for the current step."""
    return [
        f"{label}",
        f"Episode  {episode}/{total}   city seed {info['city_seed']} ({env.seed_pool})",
        f"Step     {info['step']}/{env.env_config.max_steps}",
        f"Reward   {info['episode_reward']:+8.2f}",
        f"Battery  {info['battery_fraction'] * 100:5.1f}%",
        f"Distance {info['distance_to_target']:6.2f} m",
        f"Status   {info['status']}",
    ]


def main() -> int:
    args = parse_args()
    configure_logging("INFO")

    config = load_config(args.config, overrides=args.overrides)

    render_config = dict(config.get("render", {}))
    render_config.pop("mode", None)
    render_config["show_rays"] = args.show_rays
    if args.camera_mode:
        render_config["camera_mode"] = args.camera_mode

    probe = DroneCityEnv(config=config, seed_pool=args.pool)
    num_actions = int(probe.action_space.n)
    probe.close()

    agent, label, config = load_policy(args, config, num_actions)

    render_mode = "rgb_array" if args.headless else "human"
    env = DroneCityEnv(
        config=config,
        seed_pool=args.pool,
        render_mode=render_mode,
        render_config=render_config,
    )

    recorder = None
    if args.record is not None:
        from drone_rl.evaluation.video import VideoRecorder

        env.reset(seed=0)  # force the renderer to exist before wrapping it
        recorder = VideoRecorder(renderer=env.renderer)

    base_seed = args.seed if args.seed is not None else 20_250
    step_delay = env.drone_config.dt / max(args.realtime, 1e-3) if not args.headless else 0.0

    _LOG.info("=" * 68)
    _LOG.info("DEMO: %s", label)
    _LOG.info("Cities: '%s' pool | %d episodes", args.pool, args.episodes)
    if not args.headless:
        _LOG.info("Camera: arrows pan | Shift/Ctrl up-down | PgUp/PgDn zoom")
        _LOG.info("        Alt+arrows angle | F1 free F2 follow F3 orbit F4 cinematic F5 reset")
    _LOG.info("=" * 68)

    episodes: list[EpisodeMetrics] = []
    try:
        for index in range(args.episodes):
            options = {"city_seed": args.city_seed} if args.city_seed is not None else {}
            observation, info = env.reset(seed=base_seed + index, options=options)
            if env.renderer is not None:
                env.renderer.clear_dynamic()
                env.renderer.set_markers(
                    env.start_position, env.target_position, env.env_config.target_radius
                )

            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(observation, evaluation=True)
                observation, _reward, terminated, truncated, info = env.step(action)

                if env.renderer is not None:
                    env.renderer.set_hud(
                        build_hud(label, index + 1, args.episodes, info, env),
                        position=env.drone.state.position + np.array([0.0, 0.0, 6.0]),
                    )
                if recorder is not None:
                    recorder.capture()
                if step_delay > 0:
                    time.sleep(step_delay)

            summary = env.episode_summary()
            metrics = EpisodeMetrics.from_summary(summary, episode=index)
            metrics.status = summary["status"]
            episodes.append(metrics)

            if env.renderer is not None:
                colour = _STATUS_COLORS.get(summary["status"], (1.0, 1.0, 1.0))
                env.renderer.flash_event(env.drone.state.position, colour)
            if recorder is not None:
                recorder.hold(int(args.fps * 0.8))  # linger on the outcome

            _LOG.info(
                "episode %2d | city %4d | %-14s | reward %8.2f | steps %3d | "
                "energy %5.1f | final distance %5.2f m",
                index + 1, summary["city_seed"], summary["status"], summary["reward"],
                summary["steps"], summary["energy"], summary["final_distance"],
            )
            if not args.headless:
                time.sleep(1.0)  # let the viewer read the outcome
    except KeyboardInterrupt:
        _LOG.info("Interrupted by user.")
    finally:
        if recorder is not None and args.record is not None:
            recorder.save(args.record, fps=args.fps)
        env.close()

    if episodes:
        stats = aggregate_metrics(episodes)
        _LOG.info("-" * 68)
        _LOG.info(
            "%d episodes | success %.1f%% | collision %.1f%% | timeout %.1f%% | mean reward %.2f",
            len(episodes), 100 * stats["success_rate"], 100 * stats["collision_rate"],
            100 * stats["timeout_rate"], stats["mean_reward"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
