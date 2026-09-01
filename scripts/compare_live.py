"""Fly two trained policies through the same city, side by side, live in 3D.

Both agents get an *identical* scenario - same procedural city, same start pose,
same target - so any divergence in the two trails is the policies disagreeing,
not the environment. Each drone draws its own coloured ribbon, so the comparison
is legible as a picture rather than as a table of numbers.

    cyan   = agent A  (default: the proposed HER-DQN)
    orange = agent B  (default: the DQN baseline)

The two environments are stepped independently and headlessly - range sensing is
analytic, so neither needs PyBullet - while a single renderer owns the city, the
camera and the overlay. An agent that finishes early holds its final pose and
keeps its trail on screen while the other one flies on.

Controls
--------
======================  ====================================================
N / P                   next / previous city
R                       replay the current city
SPACE                   pause and resume
TAB                     cycle camera focus: agent A, agent B, midpoint
Arrows, PgUp/PgDn, F1-F5  camera rig (see the demo script)
Q / Esc                 quit
======================  ====================================================

Examples
--------
Proposed against the baseline on unseen cities::

    python scripts/compare_live.py

Pin a specific set of maps and fly them slowly::

    python scripts/compare_live.py --city-seeds 901 907 913 --realtime 0.5

Compare a mid-training checkpoint against the finished policy::

    python scripts/compare_live.py \\
        --checkpoint-a outputs/checkpoints/her_dqn_best.pt \\
        --checkpoint-b outputs/checkpoints/her_dqn_episode_500.pt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.agents.random_agent import RandomAgent
from drone_rl.environment.drone_env import DroneCityEnv, EpisodeOutcome
from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig
from drone_rl.utils.config import load_config
from drone_rl.utils.device import resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("compare_live")

#: Per-agent identity colour, used for the airframe and its trail alike.
_COLOR_A = (0.16, 0.62, 0.96)
_COLOR_B = (0.96, 0.45, 0.13)

#: Simulation steps between overlay refreshes. See the call site for why the
#: overlay is not redrawn every frame.
_HUD_EVERY_N_STEPS = 4

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
    parser.add_argument("--checkpoint-a", type=Path,
                        default=Path("outputs/checkpoints/her_dqn_best.pt"))
    parser.add_argument("--checkpoint-b", type=Path,
                        default=Path("outputs/checkpoints/dqn_best.pt"))
    parser.add_argument("--label-a", default=None, help="Override the overlay name for A")
    parser.add_argument("--label-b", default=None, help="Override the overlay name for B")
    parser.add_argument("--random-b", action="store_true",
                        help="Fly an untrained random policy as agent B")
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--pool", default="test", choices=["train", "validation", "test"],
                        help="Seed pool the cities are drawn from")
    parser.add_argument("--city-seeds", type=int, nargs="*", default=None,
                        help="Explicit list of city layouts to cycle through")
    parser.add_argument("--cities", type=int, default=8,
                        help="How many cities to draw from the pool when none are named")
    parser.add_argument("--seed", type=int, default=20_250, help="Base episode seed")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="Playback pacing; <1 is slower, >1 faster")
    parser.add_argument("--laps", type=int, default=0,
                        help="Stop after this many passes over the city list (0 = run until Q)")
    parser.add_argument("--trail-length", type=int, default=600,
                        help="Segments kept per trail; lower this if rendering stutters")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def load_agent(checkpoint: Path, device, num_actions: int, fallback_random: bool = False):
    """Return ``(agent, label)`` for a checkpoint, or a random policy."""
    if fallback_random or checkpoint is None:
        return RandomAgent(num_actions=num_actions, seed=0), "Random policy"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = agent_from_checkpoint(payload, device=device)
    agent.eval_mode()
    label = f"{payload.get('algorithm', 'agent')} @ ep {payload.get('episode', '?')}"
    return agent, label


class DroneVisual:
    """One coloured airframe plus its trail, drawn on an existing PyBullet client.

    The renderer owns a single drone of its own; a two-agent comparison needs two
    independently coloured ones, so these are built here rather than by extending
    the renderer's single-drone contract.
    """

    #: Minimum distance between trail vertices, in metres. Every segment is a
    #: separate debug item and therefore a GUI round-trip, so emitting one per
    #: simulation step is what makes long flights crawl. At 3 m/s and dt=0.15
    #: this merges roughly every third step and the ribbon still reads smooth.
    MIN_SEGMENT = 0.12

    def __init__(self, pybullet, client_id: int, color, trail_length: int) -> None:
        self._p = pybullet
        self.client_id = client_id
        self.color = color
        self.trail_length = int(trail_length)
        self._trail_ids: list[int] = []
        self._last_point: np.ndarray | None = None
        self.body_id = self._build_body()

    def _build_body(self) -> int:
        p = self._p
        rgba = (*self.color, 1.0)
        body = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.20, 0.20, 0.075], rgbaColor=rgba,
            physicsClientId=self.client_id,
        )
        visuals, positions, orientations = [], [], []
        for angle_deg in (45.0, 135.0, 225.0, 315.0):
            angle = math.radians(angle_deg)
            dx, dy = 0.30 * math.cos(angle), 0.30 * math.sin(angle)
            visuals.append(
                p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.22, 0.03, 0.02],
                    rgbaColor=(*self.color, 1.0), physicsClientId=self.client_id,
                )
            )
            positions.append([dx * 0.6, dy * 0.6, 0.0])
            orientations.append(list(p.getQuaternionFromEuler([0.0, 0.0, angle])))
            # Rotor discs stay near-white so the airframe colour still reads as
            # the identity channel at a distance.
            visuals.append(
                p.createVisualShape(
                    p.GEOM_CYLINDER, radius=0.16, length=0.025,
                    rgbaColor=(0.93, 0.93, 0.93, 1.0), physicsClientId=self.client_id,
                )
            )
            positions.append([dx, dy, 0.05])
            orientations.append([0.0, 0.0, 0.0, 1.0])

        count = len(visuals)
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=body,
            basePosition=[0.0, 0.0, 2.0],
            linkMasses=[0.0] * count,
            linkCollisionShapeIndices=[-1] * count,
            linkVisualShapeIndices=visuals,
            linkPositions=positions,
            linkOrientations=orientations,
            linkInertialFramePositions=[[0, 0, 0]] * count,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * count,
            linkParentIndices=[0] * count,
            linkJointTypes=[self._p.JOINT_FIXED] * count,
            linkJointAxis=[[0, 0, 1]] * count,
            physicsClientId=self.client_id,
        )

    def move(self, position: np.ndarray, yaw: float) -> None:
        """Reposition the airframe and extend the trail to match."""
        p = self._p
        position = np.asarray(position, dtype=np.float64).reshape(3)
        half = float(yaw) * 0.5
        p.resetBasePositionAndOrientation(
            self.body_id,
            [float(v) for v in position],
            [0.0, 0.0, math.sin(half), math.cos(half)],
            physicsClientId=self.client_id,
        )
        if self._last_point is None:
            self._last_point = position.copy()
            return
        if float(np.linalg.norm(position - self._last_point)) < self.MIN_SEGMENT:
            return
        self._trail_ids.append(
            p.addUserDebugLine(
                [float(v) for v in self._last_point],
                [float(v) for v in position],
                lineColorRGB=list(self.color),
                lineWidth=2.5,
                physicsClientId=self.client_id,
            )
        )
        self._last_point = position.copy()
        while len(self._trail_ids) > self.trail_length:
            p.removeUserDebugItem(self._trail_ids.pop(0), physicsClientId=self.client_id)

    def clear_trail(self) -> None:
        for item_id in self._trail_ids:
            self._p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
        self._trail_ids.clear()
        self._last_point = None


def _key_pressed(pybullet, events, char: str) -> bool:
    """True on the frame ``char`` is first pressed."""
    code = ord(char)
    return bool(events.get(code, 0) & pybullet.KEY_WAS_TRIGGERED)


def main() -> int:  # noqa: C901 - a viewer loop is naturally branchy
    args = parse_args()
    configure_logging("INFO")

    config = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device)

    env_a = DroneCityEnv(config=config, seed_pool=args.pool)
    env_b = DroneCityEnv(config=config, seed_pool=args.pool)
    num_actions = int(env_a.action_space.n)

    agent_a, auto_label_a = load_agent(args.checkpoint_a, device, num_actions)
    agent_b, auto_label_b = load_agent(
        args.checkpoint_b, device, num_actions, fallback_random=args.random_b
    )
    label_a = args.label_a or auto_label_a
    label_b = args.label_b or auto_label_b

    if args.city_seeds:
        city_seeds = list(args.city_seeds)
    else:
        low, high = env_a._seed_ranges[args.pool]
        rng = np.random.default_rng(args.seed)
        city_seeds = sorted(
            int(s) for s in rng.choice(np.arange(low, high + 1), size=min(args.cities, high - low + 1), replace=False)
        )

    render_config = dict(config.get("render", {}))
    render_config.pop("mode", None)
    render_config.update(
        {
            "show_rays": False,
            "show_trail": False,  # trails are drawn per agent below
            "width": args.width,
            "height": args.height,
            "camera_mode": render_config.get("camera_mode") or "free",
        }
    )
    renderer = PyBulletRenderer(RenderConfig(mode="gui", **render_config))
    p = renderer._p

    visual_a = DroneVisual(p, renderer.client_id, _COLOR_A, args.trail_length)
    visual_b = DroneVisual(p, renderer.client_id, _COLOR_B, args.trail_length)

    step_delay = env_a.drone_config.dt / max(args.realtime, 1e-3)
    focus_modes = ("A", "B", "MID")

    _LOG.info("=" * 72)
    _LOG.info("LIVE COMPARISON  (identical city, start and target for both agents)")
    _LOG.info("  cyan   A : %s", label_a)
    _LOG.info("  orange B : %s", label_b)
    _LOG.info("Cities: %s pool %s", len(city_seeds), args.pool)
    _LOG.info("N/P city  R replay  SPACE pause  TAB focus  F1-F5 camera  Q quit")
    _LOG.info("=" * 72)

    index = 0
    focus = 0
    paused = False
    quit_requested = False
    flown = 0
    wins = {"A": 0, "B": 0}

    try:
        while not quit_requested:
            city_seed = city_seeds[index % len(city_seeds)]
            episode_seed = args.seed + index

            # Agent A defines the scenario; agent B is pinned to the same one, so
            # the only difference on screen is the policy.
            obs_a, _ = env_a.reset(seed=episode_seed, options={"city_seed": city_seed})
            obs_b, _ = env_b.reset(
                seed=episode_seed,
                options={
                    "city_seed": city_seed,
                    "start": env_a.start_position.copy(),
                    "target": env_a.target_position.copy(),
                    # Pinned rather than left to the RNG: both envs would draw the
                    # same yaw from the same seed today, but only because they
                    # happen to consume identical draws beforehand. Stating it
                    # keeps the guarantee from resting on that coincidence.
                    "yaw": float(env_a.drone.state.yaw),
                },
            )

            if not np.allclose(env_a.start_position, env_b.start_position) or not np.allclose(
                env_a.target_position, env_b.target_position
            ):
                _LOG.warning("Scenarios diverged; the comparison would not be like-for-like.")

            renderer.build_city(env_a.city)
            # The renderer builds a drone of its own; this viewer supplies two.
            if renderer._drone_id is not None:
                p.removeBody(renderer._drone_id, physicsClientId=renderer.client_id)
                renderer._drone_id = None
            renderer.set_markers(
                env_a.start_position, env_a.target_position, env_a.env_config.target_radius
            )
            visual_a.clear_trail()
            visual_b.clear_trail()

            done_a = done_b = False
            status_a = status_b = EpisodeOutcome.RUNNING
            steps_a = steps_b = 0
            reward_a = reward_b = 0.0
            frame = 0
            advance = 0  # -1 previous, +1 next, 0 stay
            replay = False

            while True:
                events = p.getKeyboardEvents(physicsClientId=renderer.client_id)

                if _key_pressed(p, events, "n"):
                    advance = 1
                    break
                if _key_pressed(p, events, "p"):
                    advance = -1
                    break
                if _key_pressed(p, events, "r"):
                    replay = True
                    break
                if _key_pressed(p, events, "q") or events.get(27, 0) & p.KEY_WAS_TRIGGERED:
                    quit_requested = True
                    break
                if _key_pressed(p, events, " "):
                    paused = not paused
                if events.get(9, 0) & p.KEY_WAS_TRIGGERED:  # TAB
                    focus = (focus + 1) % len(focus_modes)

                if not paused:
                    if not done_a:
                        action = agent_a.act(obs_a, evaluation=True)
                        obs_a, r_a, term_a, trunc_a, info_a = env_a.step(action)
                        reward_a += float(r_a)
                        steps_a += 1
                        if term_a or trunc_a:
                            done_a = True
                            status_a = env_a.status
                            renderer.flash_event(
                                env_a.drone.state.position,
                                _STATUS_COLORS.get(status_a, (1.0, 1.0, 1.0)),
                            )
                    if not done_b:
                        action = agent_b.act(obs_b, evaluation=True)
                        obs_b, r_b, term_b, trunc_b, info_b = env_b.step(action)
                        reward_b += float(r_b)
                        steps_b += 1
                        if term_b or trunc_b:
                            done_b = True
                            status_b = env_b.status
                            renderer.flash_event(
                                env_b.drone.state.position,
                                _STATUS_COLORS.get(status_b, (1.0, 1.0, 1.0)),
                            )

                position_a = env_a.drone.state.position
                position_b = env_b.drone.state.position
                visual_a.move(position_a, env_a.drone.state.yaw)
                visual_b.move(position_b, env_b.drone.state.yaw)

                distance_a = float(np.linalg.norm(env_a.target_position - position_a))
                distance_b = float(np.linalg.norm(env_b.target_position - position_b))
                frame += 1
                # The overlay is a dozen debug-text round-trips; refreshing it on
                # every simulation step costs more than the flight itself. At
                # dt=0.15 s this still updates ~2x a second, which reads as live.
                if frame % _HUD_EVERY_N_STEPS == 0 or (done_a and done_b):
                    renderer.set_hud(
                        [
                            f"city seed {city_seed}  ({args.pool} pool)"
                            f"   [{index % len(city_seeds) + 1}/{len(city_seeds)}]"
                            + ("   PAUSED" if paused else ""),
                            "",
                            f"CYAN   A  {label_a}",
                            f"          {status_a:<14} step {steps_a:>3}  "
                            f"R {reward_a:+7.1f}  d {distance_a:5.1f} m",
                            f"ORANGE B  {label_b}",
                            f"          {status_b:<14} step {steps_b:>3}  "
                            f"R {reward_b:+7.1f}  d {distance_b:5.1f} m",
                            "",
                            f"score  A {wins['A']} - {wins['B']} B"
                            f"    focus {focus_modes[focus]}",
                            "N/P city   R replay   SPACE pause   TAB focus   Q quit",
                        ],
                        position=np.array(
                            [0.0, 0.0, env_a.city_config.max_altitude + 4.0]
                        ),
                    )

                target = {
                    "A": position_a,
                    "B": position_b,
                    "MID": (position_a + position_b) * 0.5,
                }[focus_modes[focus]]
                renderer.camera.update(target, events=events)

                if done_a and done_b:
                    break
                if step_delay > 0 and not paused:
                    time.sleep(step_delay)

            if quit_requested:
                break

            if done_a and done_b and not replay and advance == 0:
                flown += 1
                if status_a == EpisodeOutcome.TARGET_REACHED:
                    wins["A"] += 1
                if status_b == EpisodeOutcome.TARGET_REACHED:
                    wins["B"] += 1
                _LOG.info(
                    "city %4d | A %-14s steps %3d R %+7.1f | B %-14s steps %3d R %+7.1f",
                    city_seed, status_a, steps_a, reward_a, status_b, steps_b, reward_b,
                )
                if args.laps > 0 and flown >= args.laps * len(city_seeds):
                    break
                # Hold on the finished picture so both trails can be read.
                hold_until = time.perf_counter() + 2.5
                while time.perf_counter() < hold_until:
                    events = p.getKeyboardEvents(physicsClientId=renderer.client_id)
                    if _key_pressed(p, events, "n"):
                        break
                    if _key_pressed(p, events, "q"):
                        quit_requested = True
                        break
                    renderer.camera.update(
                        (env_a.drone.state.position + env_b.drone.state.position) * 0.5,
                        events=events,
                    )
                    time.sleep(1.0 / 60.0)
                advance = 1

            if replay:
                continue
            # advance is +-1 by here: either the viewer asked for a different
            # city, or the run finished and the hold block set it to +1.
            index = (index + advance) % len(city_seeds)

    except KeyboardInterrupt:
        _LOG.info("Interrupted by user.")
    finally:
        renderer.close()
        env_a.close()
        env_b.close()

    _LOG.info("-" * 72)
    _LOG.info("Targets reached | A (%s): %d | B (%s): %d", label_a, wins["A"], label_b, wins["B"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
