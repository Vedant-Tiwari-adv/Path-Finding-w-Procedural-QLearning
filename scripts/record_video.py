"""Record evaluation episodes to MP4/GIF (optional video capture).

Runs headless by default: frames are rendered offscreen with PyBullet's
software renderer, so recording never opens a window or steals focus, and can
run alongside a training job. Pass ``--gui`` to render through a window
instead, which uses hardware OpenGL and is roughly 4x faster per frame - the
better choice for final, higher-resolution clips.

Frame capture dominates runtime, and its cost depends on how PyBullet was
built. If ``pybullet.isNumpyEnabled()`` returns 0, every frame is marshalled
through a Python tuple of ~1M integers; see the README for the rebuild command.

Because the analytic physics core is the same one used for training, a recorded
clip is a faithful replay of what the policy actually does - not a re-simulation
under different dynamics.

Examples
--------
Record the best checkpoint on unseen cities::

    python scripts/record_video.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --episodes 3 --output outputs/videos/dqn_best.mp4

Record every training stage, each to its own clip::

    python scripts/record_video.py --all-stages --output-dir outputs/videos

Only record successful deliveries (nice for a highlight reel)::

    python scripts/record_video.py --checkpoint outputs/checkpoints/dqn_best.pt \\
        --episodes 20 --only-success --max-clips 3
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.environment.drone_env import DroneCityEnv, EpisodeOutcome
from drone_rl.evaluation.overlay import FrameOverlay
from drone_rl.evaluation.video import VideoRecorder
from drone_rl.utils.config import load_config
from drone_rl.utils.device import resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("record_video")

_EPISODE_PATTERN = re.compile(r"_episode_(\d+)\.pt$")

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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--random", action="store_true", help="Record a random policy")
    parser.add_argument("--all-stages", action="store_true",
                        help="Record random + every *_episode_N.pt + best, each separately")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--max-stages", type=int, default=4,
                        help="Cap on auto-selected training checkpoints")
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--pool", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--city-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Single output file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--every-n-steps", type=int, default=1,
                        help="Capture one frame per N simulation steps. Frame capture is "
                             "the dominant cost, so 2 halves recording time; halve --fps "
                             "alongside it to keep the same playback speed.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--camera-mode", default="orbit",
                        choices=["free", "follow", "orbit", "cinematic"])
    parser.add_argument("--show-rays", action="store_true")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Do not composite the HUD/trajectory onto frames. PyBullet's own "
                             "debug overlay is never captured by getCameraImage, so this "
                             "yields clean scene-only footage.")
    parser.add_argument("--camera-distance", type=float, default=None,
                        help="Override camera distance; smaller framing makes the drone "
                             "read clearly in a clip (try 16-22)")
    parser.add_argument("--camera-pitch", type=float, default=None)
    parser.add_argument("--gui", action="store_true",
                        help="Render through a visible window. Counter-intuitively this is "
                             "also the FAST path: it uses hardware OpenGL (~84 ms/frame at "
                             "960x600) versus the headless software rasteriser (~358 ms). "
                             "Use it for final clips; use headless to record unattended.")
    parser.add_argument("--only-success", action="store_true",
                        help="Keep only episodes that reached the target")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Stop after this many kept episodes")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def stage_label(path: Path | None) -> str:
    """Readable name for a checkpoint."""
    if path is None:
        return "random"
    match = _EPISODE_PATTERN.search(path.name)
    if match:
        return f"episode_{int(match.group(1)):05d}"
    return path.stem


def discover_stages(directory: Path, max_stages: int) -> list[Path | None]:
    """Random policy plus evenly spaced checkpoints plus the best model."""
    stages: list[Path | None] = [None]
    if not directory.is_dir():
        return stages

    numbered = []
    for path in directory.glob("*_episode_*.pt"):
        match = _EPISODE_PATTERN.search(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])

    if numbered and max_stages > 0:
        indices = np.linspace(0, len(numbered) - 1, num=min(max_stages, len(numbered)))
        for index in sorted({int(round(i)) for i in indices}):
            stages.append(numbered[index][1])

    best = directory / "dqn_best.pt"
    if best.is_file():
        stages.append(best)
    return stages


def load_agent(path: Path | None, num_actions: int, device_name: str, seed: int):
    """Build the policy for one stage."""
    if path is None:
        return RandomAgent(num_actions=num_actions, seed=seed)
    device = resolve_device(device_name)
    payload = torch.load(path, map_location=device, weights_only=False)
    agent = agent_from_checkpoint(payload, device=device)
    agent.eval_mode()
    return agent


def record_stage(
    env: DroneCityEnv,
    agent,
    label: str,
    args: argparse.Namespace,
    output_path: Path,
) -> Path | None:
    """Record ``args.episodes`` episodes of one policy into a single clip."""
    recorder = VideoRecorder(
        renderer=env.renderer,
        width=args.width,
        height=args.height,
        every_n_steps=args.every_n_steps,
        max_frames=6_000,
        overlay=None if args.no_overlay else FrameOverlay(args.width, args.height),
    )
    kept = 0
    outcomes: list[str] = []

    for index in range(args.episodes):
        if args.max_clips is not None and kept >= args.max_clips:
            break
        options = {"city_seed": args.city_seed} if args.city_seed is not None else {}
        observation, info = env.reset(seed=args.seed + index, options=options)
        env.renderer.clear_dynamic()
        env.renderer.set_markers(
            env.start_position, env.target_position, env.env_config.target_radius
        )

        mark = len(recorder.frames)  # rollback point if we discard this episode
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(observation, evaluation=True)
            observation, _r, terminated, truncated, info = env.step(action)
            hud = [
                f"{label}",
                f"Episode  {index + 1}   city {info['city_seed']} ({env.seed_pool})",
                f"Step     {info['step']}/{env.env_config.max_steps}",
                f"Reward   {info['episode_reward']:+8.2f}",
                f"Battery  {info['battery_fraction'] * 100:5.1f}%",
                f"Distance {info['distance_to_target']:6.2f} m",
                f"Status   {info['status']}",
            ]
            # The live window still gets PyBullet's own overlay; the recorder
            # composites its own copy, because debug items are not captured.
            env.renderer.set_hud(
                hud, position=env.drone.state.position + np.array([0.0, 0.0, 6.0])
            )
            recorder.capture(
                hud_lines=hud,
                trajectory=env.trajectory_array(),
                drone_position=env.drone.state.position,
                target_position=env.target_position,
                start_position=env.start_position,
            )

        summary = env.episode_summary()
        colour = _STATUS_COLORS.get(summary["status"], (1.0, 1.0, 1.0))
        env.renderer.flash_event(env.drone.state.position, colour)
        final_hud = [
            f"{label}",
            f"Episode  {index + 1}   city {summary['city_seed']} ({env.seed_pool})",
            f"Steps    {summary['steps']}",
            f"Reward   {summary['reward']:+8.2f}",
            f"Energy   {summary['energy']:6.1f}",
            f"Distance {summary['final_distance']:6.2f} m",
            f"Status   {summary['status']}",
        ]
        recorder.capture(
            force=True,
            hud_lines=final_hud,
            trajectory=env.trajectory_array(),
            drone_position=env.drone.state.position,
            target_position=env.target_position,
            start_position=env.start_position,
            status_colour=tuple(int(255 * c) for c in colour),
        )
        recorder.hold(int(args.fps * 1.2))

        if args.only_success and not summary["success"]:
            del recorder.frames[mark:]  # discard this episode's frames
            continue

        kept += 1
        outcomes.append(summary["status"])
        _LOG.info("  ep %2d | city %4d | %-14s | reward %8.2f | steps %3d",
                  index + 1, summary["city_seed"], summary["status"],
                  summary["reward"], summary["steps"])

    if not recorder.frames:
        _LOG.warning("  no frames kept for %s", label)
        return None
    _LOG.info("  outcomes: %s", ", ".join(outcomes) or "(none)")
    return recorder.save(output_path, fps=args.fps)


def main() -> int:
    args = parse_args()
    configure_logging("INFO")
    config = load_config(args.config, overrides=args.overrides)

    render_config = dict(config.get("render", {}))
    render_config.pop("mode", None)
    render_config["show_rays"] = args.show_rays
    render_config["camera_mode"] = args.camera_mode
    render_config["width"] = args.width
    render_config["height"] = args.height
    if args.camera_distance is not None:
        render_config["camera_distance"] = args.camera_distance
    if args.camera_pitch is not None:
        render_config["camera_pitch"] = args.camera_pitch

    stages: list[Path | None]
    if args.all_stages:
        stages = discover_stages(args.checkpoint_dir, args.max_stages)
    elif args.random or args.checkpoint is None:
        stages = [None]
    else:
        stages = [args.checkpoint]

    env = DroneCityEnv(
        config=config,
        seed_pool=args.pool,
        render_mode="human" if args.gui else "rgb_array",
        render_config=render_config,
    )
    num_actions = int(env.action_space.n)

    _LOG.info("=" * 66)
    _LOG.info("RECORDING %d stage(s), %d episode(s) each, %s",
              len(stages), args.episodes, "GUI" if args.gui else "headless")
    _LOG.info("=" * 66)

    written: list[Path] = []
    try:
        env.reset(seed=args.seed)  # instantiate the renderer
        for stage in stages:
            label = stage_label(stage)
            agent = load_agent(stage, num_actions, args.device, args.seed)
            _LOG.info("Stage: %s", label)

            if args.output is not None and len(stages) == 1:
                output_path = args.output
            else:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / f"{label}.mp4"

            path = record_stage(env, agent, label, args, output_path)
            if path is not None:
                written.append(path)
    finally:
        env.close()

    _LOG.info("=" * 66)
    for path in written:
        _LOG.info("Wrote %s", path)
    _LOG.info("%d clip(s) written", len(written))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
