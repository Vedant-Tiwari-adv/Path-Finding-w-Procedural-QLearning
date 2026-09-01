"""Grid arena: many cities at once, every policy racing in every one.

A single world is tiled with an R x C grid of independent procedural cities.
Every cell runs its own scenario, and every policy under test flies that
scenario simultaneously, each in its own colour. From the top-down overview the
whole grid reads as a scoreboard - you can see at a glance which colour arrives,
which stalls, and which flies into a wall, across dozens of layouts at once
rather than one at a time. Drop into any cell for a close 3D view and the others
keep running behind you.

Cells are independent: when everyone in a cell has finished it holds the result
briefly, then rolls to a fresh city and starts again, so the arena never stops.

Why it stays smooth
-------------------
The simulation ticks at ``dt`` (0.15 s, about 6.7 Hz) but the viewer renders at
60 fps on a fixed-timestep accumulator, interpolating each drone between its
previous and current pose. Rendering is therefore decoupled from the physics
rate: what you see is a continuous glide, not the 6.7 Hz staircase you get from
drawing one frame per simulation step.

Policy evaluation is batched across the whole grid - one forward pass per policy
per tick covering every cell - so an 81-cell arena costs about as much network
time as a single cell.

Controls
--------
========================  ==================================================
O                         top-down overview of the whole grid
TAB / N                   focus the next cell (3D close-up)
B                         focus the previous cell
R                         restart every cell on fresh cities
SPACE                     pause / resume
Arrows                    pan, relative to where the camera is looking
Shift / Ctrl              raise / lower
PgUp / PgDn               zoom
Alt + Arrows              swing the viewing angle
Q / Esc                   quit
========================  ==================================================

Examples
--------
The default 3x3 arena, proposed vs baseline::

    python scripts/arena.py

A big board, sixteen cities at once::

    python scripts/arena.py --grid 4x4

Three-way, including an untrained policy for reference::

    python scripts/arena.py --agents outputs/checkpoints/her_dqn_best.pt \\
        outputs/checkpoints/dqn_best.pt random
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from drone_rl.agents.random_agent import RandomAgent
from drone_rl.agents.registry import agent_from_checkpoint
from drone_rl.environment.camera import CameraMode
from drone_rl.environment.drone_env import DroneCityEnv, EpisodeOutcome
from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig
from drone_rl.utils.config import load_config
from drone_rl.utils.device import resolve_device
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("arena")

#: Identity colours, assigned to policies in order.
_AGENT_COLORS = (
    (0.16, 0.62, 0.96),  # cyan-blue
    (0.96, 0.45, 0.13),  # orange
    (0.30, 0.85, 0.40),  # green
    (0.85, 0.35, 0.80),  # magenta
)

_COLOR_GROUND = (0.20, 0.22, 0.26, 1.0)
_COLOR_BUILDING = (0.62, 0.64, 0.70, 1.0)
_COLOR_BUILDING_TALL = (0.45, 0.48, 0.58, 1.0)
_COLOR_TARGET = (0.10, 0.85, 0.35, 0.5)

_RENDER_FPS = 60.0
#: Seconds a finished cell holds its result before rolling to a new city.
_HOLD_SECONDS = 2.0
#: Simulation ticks between overlay refreshes (debug text is expensive).
_HUD_EVERY = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", nargs="*", default=[
        "outputs/checkpoints/her_dqn_best.pt",
        "outputs/checkpoints/dqn_best.pt",
    ], help="Checkpoints to race; the literal word 'random' adds a random policy")
    parser.add_argument("--grid", default="3x3", help="Arena size, e.g. 3x3, 4x4, 9x9")
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"))
    parser.add_argument("--pool", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=20_250)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed multiplier; the render stays at 60 fps")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Exit after this long (0 = run until Q)")
    parser.add_argument("--trail-length", type=int, default=70,
                        help="Trail segments kept per drone; lower it on very large grids")
    parser.add_argument("--spacing", type=float, default=1.35,
                        help="Cell pitch as a multiple of city width")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def parse_grid(text: str) -> tuple[int, int]:
    """Parse ``"3x3"`` into ``(rows, cols)``."""
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"Bad --grid {text!r}; expected something like 3x3")
    rows, cols = (max(1, int(p)) for p in parts)
    return rows, cols


class Policy:
    """A named policy plus its identity colour and batched inference path."""

    def __init__(self, agent, label: str, color, device) -> None:
        self.agent = agent
        self.label = label
        self.color = color
        self.device = device
        # Value-based agents expose a Q-network, which lets the whole grid be
        # evaluated in a single forward pass instead of one call per cell.
        self._network = getattr(agent, "q_network", None)

    def act_batch(self, observations: np.ndarray) -> np.ndarray:
        """Greedy actions for a stack of observations, ``(N, obs) -> (N,)``."""
        if len(observations) == 0:
            return np.zeros(0, dtype=np.int64)
        if self._network is None:
            return np.array(
                [self.agent.act(o, evaluation=True) for o in observations], dtype=np.int64
            )
        with torch.no_grad():
            tensor = torch.as_tensor(
                np.asarray(observations, dtype=np.float32), device=self.device
            )
            return self._network(tensor).argmax(dim=1).cpu().numpy().astype(np.int64)


class DroneVisual:
    """A coloured airframe and its trail, drawn on an existing PyBullet client."""

    def __init__(self, pybullet, client_id, color, trail_length, min_segment) -> None:
        self._p = pybullet
        self.client_id = client_id
        self.color = color
        self.trail_length = int(trail_length)
        self.min_segment = float(min_segment)
        self._trail_ids: list[int] = []
        self._last_point: np.ndarray | None = None
        self.body_id = self._build()

    def _build(self) -> int:
        p = self._p
        rgba = (*self.color, 1.0)
        base = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.30, 0.30, 0.10], rgbaColor=rgba,
            physicsClientId=self.client_id,
        )
        visuals, positions, orientations = [], [], []
        for angle_deg in (45.0, 135.0, 225.0, 315.0):
            angle = math.radians(angle_deg)
            dx, dy = 0.34 * math.cos(angle), 0.34 * math.sin(angle)
            visuals.append(
                p.createVisualShape(
                    p.GEOM_CYLINDER, radius=0.20, length=0.03,
                    rgbaColor=(0.95, 0.95, 0.95, 1.0), physicsClientId=self.client_id,
                )
            )
            positions.append([dx, dy, 0.06])
            orientations.append([0.0, 0.0, 0.0, 1.0])
        count = len(visuals)
        return p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=base,
            basePosition=[0.0, 0.0, 2.0],
            linkMasses=[0.0] * count,
            linkCollisionShapeIndices=[-1] * count,
            linkVisualShapeIndices=visuals,
            linkPositions=positions,
            linkOrientations=orientations,
            linkInertialFramePositions=[[0, 0, 0]] * count,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * count,
            linkParentIndices=[0] * count,
            linkJointTypes=[p.JOINT_FIXED] * count,
            linkJointAxis=[[0, 0, 1]] * count,
            physicsClientId=self.client_id,
        )

    def draw(self, position: np.ndarray, yaw: float) -> None:
        """Place the airframe at an already-interpolated pose."""
        p = self._p
        half = float(yaw) * 0.5
        p.resetBasePositionAndOrientation(
            self.body_id,
            [float(v) for v in position],
            [0.0, 0.0, math.sin(half), math.cos(half)],
            physicsClientId=self.client_id,
        )

    def extend_trail(self, position: np.ndarray) -> None:
        """Add a trail vertex if the drone has moved far enough to warrant one."""
        p = self._p
        position = np.asarray(position, dtype=np.float64)
        if self._last_point is None:
            self._last_point = position.copy()
            return
        if float(np.linalg.norm(position - self._last_point)) < self.min_segment:
            return
        self._trail_ids.append(
            p.addUserDebugLine(
                [float(v) for v in self._last_point],
                [float(v) for v in position],
                lineColorRGB=list(self.color), lineWidth=2.0,
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


class Cell:
    """One city in the grid, with one environment per competing policy.

    All policies in a cell fly the *same* scenario - identical layout, start and
    target - so the only thing that differs between the coloured ribbons is the
    policy that drew them.
    """

    def __init__(self, pybullet, client_id, index, offset, config, pool, policies,
                 trail_length, min_segment) -> None:
        self._p = pybullet
        self.client_id = client_id
        self.index = index
        self.offset = np.asarray(offset, dtype=np.float64)
        self.envs = [DroneCityEnv(config=config, seed_pool=pool) for _ in policies]
        self.visuals = [
            DroneVisual(pybullet, client_id, policy.color, trail_length, min_segment)
            for policy in policies
        ]
        self.observations = [None] * len(policies)
        self.done = [False] * len(policies)
        self.status = [EpisodeOutcome.RUNNING] * len(policies)
        self.previous: list[tuple[np.ndarray, float]] = [
            (np.zeros(3), 0.0) for _ in policies
        ]
        self.current: list[tuple[np.ndarray, float]] = [
            (np.zeros(3), 0.0) for _ in policies
        ]
        self.city_seed = -1
        self.hold_until: float | None = None
        self._building_ids: list[int] = []
        self._ground_id: int | None = None
        self._target_id: int | None = None

    # ---- scene ---------------------------------------------------------------

    def _clear_scene(self) -> None:
        p = self._p
        for body_id in self._building_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self._building_ids.clear()
        for attribute in ("_ground_id", "_target_id"):
            body_id = getattr(self, attribute)
            if body_id is not None:
                p.removeBody(body_id, physicsClientId=self.client_id)
                setattr(self, attribute, None)

    def _build_scene(self, layout) -> None:
        """Build this cell's city, translated to the cell's slot in the grid."""
        p = self._p
        cfg = layout.config
        ground_half = [cfg.width * 0.5, cfg.depth * 0.5, 0.1]
        ground_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=ground_half, rgbaColor=_COLOR_GROUND,
            physicsClientId=self.client_id,
        )
        self._ground_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=ground_visual,
            basePosition=[self.offset[0], self.offset[1], -0.1],
            physicsClientId=self.client_id,
        )

        tall = cfg.building_min_height + 0.6 * (cfg.building_max_height - cfg.building_min_height)
        for building in layout.buildings:
            color = _COLOR_BUILDING_TALL if building.height >= tall else _COLOR_BUILDING
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=list(building.half_extents), rgbaColor=color,
                physicsClientId=self.client_id,
            )
            centre = np.asarray(building.center, dtype=np.float64) + self.offset
            self._building_ids.append(
                p.createMultiBody(
                    baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
                    basePosition=[float(v) for v in centre],
                    physicsClientId=self.client_id,
                )
            )

    def _place_target(self, target: np.ndarray, radius: float) -> None:
        p = self._p
        visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=float(radius), rgbaColor=_COLOR_TARGET,
            physicsClientId=self.client_id,
        )
        self._target_id = p.createMultiBody(
            baseMass=0.0, baseVisualShapeIndex=visual,
            basePosition=[float(v) for v in (np.asarray(target) + self.offset)],
            physicsClientId=self.client_id,
        )

    # ---- episode lifecycle ---------------------------------------------------

    def start(self, city_seed: int, episode_seed: int) -> None:
        """Reset every policy in this cell onto one shared fresh scenario."""
        lead = self.envs[0]
        observation, _ = lead.reset(seed=episode_seed, options={"city_seed": city_seed})
        self.observations[0] = observation
        shared = {
            "city_seed": city_seed,
            "start": lead.start_position.copy(),
            "target": lead.target_position.copy(),
            "yaw": float(lead.drone.state.yaw),
        }
        for slot in range(1, len(self.envs)):
            self.observations[slot], _ = self.envs[slot].reset(
                seed=episode_seed, options=shared
            )

        self.city_seed = int(city_seed)
        self.done = [False] * len(self.envs)
        self.status = [EpisodeOutcome.RUNNING] * len(self.envs)
        self.hold_until = None

        self._clear_scene()
        self._build_scene(lead.city)
        self._place_target(lead.target_position, lead.env_config.target_radius)
        for slot, visual in enumerate(self.visuals):
            visual.clear_trail()
            pose = (self.envs[slot].drone.state.position + self.offset,
                    float(self.envs[slot].drone.state.yaw))
            self.previous[slot] = pose
            self.current[slot] = pose

    @property
    def finished(self) -> bool:
        return all(self.done)

    def commit_pose(self, slot: int) -> None:
        """Roll current pose into previous, then read the fresh one."""
        self.previous[slot] = self.current[slot]
        state = self.envs[slot].drone.state
        self.current[slot] = (state.position + self.offset, float(state.yaw))

    def render(self, alpha: float) -> None:
        """Draw every drone at its interpolated pose."""
        for slot, visual in enumerate(self.visuals):
            (prev_position, prev_yaw) = self.previous[slot]
            (position, yaw) = self.current[slot]
            blended = prev_position + (position - prev_position) * alpha
            # Interpolate the shorter way round the circle, so a wrap from +pi to
            # -pi does not spin the airframe the long way through a whole turn.
            delta = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
            visual.draw(blended, prev_yaw + delta * alpha)
            visual.extend_trail(blended)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def load_policies(specs, device, num_actions) -> list[Policy]:
    """Build a :class:`Policy` per checkpoint path (or ``random``)."""
    policies: list[Policy] = []
    for position, spec in enumerate(specs):
        color = _AGENT_COLORS[position % len(_AGENT_COLORS)]
        if str(spec).strip().lower() == "random":
            policies.append(
                Policy(RandomAgent(num_actions=num_actions, seed=0), "random", color, device)
            )
            continue
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)
        agent = agent_from_checkpoint(payload, device=device)
        agent.eval_mode()
        label = f"{payload.get('algorithm', 'agent')} @ ep {payload.get('episode', '?')}"
        policies.append(Policy(agent, label, color, device))
    return policies


def main() -> int:  # noqa: C901 - a viewer loop is naturally branchy
    args = parse_args()
    configure_logging("INFO")

    rows, cols = parse_grid(args.grid)
    config = load_config(args.config, overrides=args.overrides)
    device = resolve_device(args.device)

    probe = DroneCityEnv(config=config, seed_pool=args.pool)
    num_actions = int(probe.action_space.n)
    city_width = float(probe.city_config.width)
    city_depth = float(probe.city_config.depth)
    max_altitude = float(probe.city_config.max_altitude)
    seed_low, seed_high = probe._seed_ranges[args.pool]
    probe.close()

    policies = load_policies(args.agents, device, num_actions)

    stride_x = city_width * args.spacing
    stride_y = city_depth * args.spacing
    render_config = dict(config.get("render", {}))
    render_config.pop("mode", None)
    render_config.update({
        "show_rays": False, "show_trail": False,
        "width": args.width, "height": args.height,
        "camera_mode": "free", "show_camera_help": False,
    })
    renderer = PyBulletRenderer(RenderConfig(mode="gui", **render_config))
    p = renderer._p
    camera = renderer.camera

    # Larger grids are viewed from further away, so trail vertices can be
    # coarser without any visible loss of detail.
    min_segment = 0.15 if max(rows, cols) <= 3 else 0.30

    _LOG.info("Building a %dx%d arena (%d cells, %d policies)...",
              rows, cols, rows * cols, len(policies))
    cells: list[Cell] = []
    for row in range(rows):
        for col in range(cols):
            offset = (
                (col - (cols - 1) / 2.0) * stride_x,
                ((rows - 1) / 2.0 - row) * stride_y,
                0.0,
            )
            cells.append(
                Cell(p, renderer.client_id, len(cells), offset, config, args.pool,
                     policies, args.trail_length, min_segment)
            )

    rng = np.random.default_rng(args.seed)

    def next_city_seed() -> int:
        return int(rng.integers(seed_low, seed_high + 1))

    for index, cell in enumerate(cells):
        cell.start(next_city_seed(), args.seed + index)

    grid_span = max(cols * stride_x, rows * stride_y)
    overview = {
        "target": np.array([0.0, 0.0, 0.0]),
        "distance": grid_span * 0.62,
        "yaw": 0.0,
        "pitch": -89.0,
    }

    def focus_view(cell: Cell) -> dict:
        return {
            "target": cell.offset + np.array([0.0, 0.0, 6.0]),
            "distance": max(city_width, city_depth) * 0.75,
            "yaw": 45.0,
            "pitch": -32.0,
        }

    camera.set_mode(CameraMode.FREE)
    camera.target = overview["target"].astype(np.float64).copy()
    camera.distance = overview["distance"]
    camera.yaw = overview["yaw"]
    camera.pitch = overview["pitch"]
    camera.apply()

    glide_goal: dict | None = None
    focused = -1  # -1 is the overview
    paused = False
    tick = 0
    wins = [0] * len(policies)
    finishes = 0

    _LOG.info("=" * 74)
    _LOG.info("ARENA  %dx%d  |  %d cities live  |  pool '%s'", rows, cols, len(cells), args.pool)
    for policy, colour_name in zip(policies, ("cyan", "orange", "green", "magenta")):
        _LOG.info("   %-8s %s", colour_name, policy.label)
    _LOG.info("O overview | TAB next cell | B previous | R restart | SPACE pause | Q quit")
    _LOG.info("=" * 74)

    simulation_dt = float(config.get("drone", {}).get("dt", 0.15))
    frame_budget = 1.0 / _RENDER_FPS
    accumulator = 0.0
    previous_time = time.perf_counter()
    started_at = previous_time
    running = True

    try:
        while running:
            frame_start = time.perf_counter()
            frame_dt = min(frame_start - previous_time, 0.25)
            previous_time = frame_start

            events = p.getKeyboardEvents(physicsClientId=renderer.client_id)

            def pressed(char: str) -> bool:
                return bool(events.get(ord(char), 0) & p.KEY_WAS_TRIGGERED)

            if pressed("q") or events.get(27, 0) & p.KEY_WAS_TRIGGERED:
                running = False
                break
            if args.seconds > 0 and frame_start - started_at >= args.seconds:
                running = False
                break
            if pressed(" "):
                paused = not paused
            if pressed("o"):
                focused = -1
                glide_goal = overview
            if events.get(9, 0) & p.KEY_WAS_TRIGGERED or pressed("n"):  # TAB
                focused = (focused + 1) % len(cells)
                glide_goal = focus_view(cells[focused])
            if pressed("b"):
                focused = (focused - 1) % len(cells)
                glide_goal = focus_view(cells[focused])
            if pressed("r"):
                for index, cell in enumerate(cells):
                    cell.start(next_city_seed(), args.seed + tick + index)

            # --- fixed-timestep simulation, decoupled from the frame rate ---
            if not paused:
                accumulator += frame_dt * max(args.speed, 0.0)
            steps_this_frame = 0
            while accumulator >= simulation_dt and steps_this_frame < 4:
                accumulator -= simulation_dt
                steps_this_frame += 1
                tick += 1

                for slot, policy in enumerate(policies):
                    active = [
                        cell for cell in cells
                        if not cell.done[slot] and cell.hold_until is None
                    ]
                    if not active:
                        continue
                    batch = np.stack([cell.observations[slot] for cell in active])
                    actions = policy.act_batch(batch)
                    for cell, action in zip(active, actions):
                        observation, _reward, terminated, truncated, _info = cell.envs[
                            slot
                        ].step(int(action))
                        cell.observations[slot] = observation
                        cell.commit_pose(slot)
                        if terminated or truncated:
                            cell.done[slot] = True
                            cell.status[slot] = cell.envs[slot].status
                            if cell.status[slot] == EpisodeOutcome.TARGET_REACHED:
                                wins[slot] += 1

                now = time.perf_counter()
                for index, cell in enumerate(cells):
                    if cell.finished and cell.hold_until is None:
                        cell.hold_until = now + _HOLD_SECONDS
                        finishes += 1
                    elif cell.hold_until is not None and now >= cell.hold_until:
                        cell.start(next_city_seed(), args.seed + tick + index)

            # --- render ---
            alpha = float(np.clip(accumulator / simulation_dt, 0.0, 1.0)) if not paused else 1.0
            for cell in cells:
                cell.render(alpha)

            # A programmatic view change eases in rather than cutting, and any
            # manual camera input cancels it so the rig never fights the viewer.
            if glide_goal is not None:
                if np.any(camera._pan_velocity) or abs(camera._zoom_velocity) > 1e-3:
                    glide_goal = None
                else:
                    blend = 1.0 - math.exp(-6.0 * frame_dt)
                    camera.target += (glide_goal["target"] - camera.target) * blend
                    camera.distance += (glide_goal["distance"] - camera.distance) * blend
                    camera.pitch += (glide_goal["pitch"] - camera.pitch) * blend
                    delta_yaw = (glide_goal["yaw"] - camera.yaw + 180.0) % 360.0 - 180.0
                    camera.yaw += delta_yaw * blend
                    if (
                        float(np.linalg.norm(glide_goal["target"] - camera.target)) < 0.4
                        and abs(glide_goal["distance"] - camera.distance) < 0.4
                    ):
                        glide_goal = None

            camera.update(events=events)

            if tick % _HUD_EVERY == 0 or paused:
                scoreboard = "   ".join(
                    f"{name}:{count}"
                    for name, count in zip(("cyan", "orange", "green", "magenta"), wins)
                )
                view = "OVERVIEW" if focused < 0 else f"cell {focused + 1}/{len(cells)}"
                renderer.set_hud(
                    [
                        f"ARENA {rows}x{cols}   {view}" + ("   PAUSED" if paused else ""),
                        f"targets reached   {scoreboard}",
                        f"episodes finished {finishes}",
                        "O overview  TAB next  B prev  R restart  SPACE pause  Q quit",
                    ],
                    position=camera.target + np.array([0.0, 0.0, max_altitude * 1.6]),
                )

            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_budget:
                time.sleep(frame_budget - elapsed)

    except KeyboardInterrupt:
        _LOG.info("Interrupted by user.")
    finally:
        for cell in cells:
            cell.close()
        renderer.close()

    _LOG.info("-" * 74)
    _LOG.info("%d episodes flown across %d cities", finishes, len(cells))
    for policy, count in zip(policies, wins):
        share = 100.0 * count / max(finishes, 1)
        _LOG.info("  %-28s targets reached %4d  (%.1f%% of finished cells)",
                  policy.label, count, share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
