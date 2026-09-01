"""Generate and inspect procedural cities (Phase 1, Steps 3-4).

Prints layout statistics, optionally saves a top-down figure, and optionally
opens a PyBullet GUI window showing the generated city so you can confirm by
eye that a seed produces a sensible skyline.

Examples
--------
Show statistics for three seeds and prove determinism::

    python scripts/generate_city.py --seeds 0 1 2 --check-determinism

Open the 3D viewer on one city::

    python scripts/generate_city.py --seed 7 --gui

Save top-down figures for a range of seeds::

    python scripts/generate_city.py --seeds 0 1 2 3 --save-figures
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup; must precede drone_rl imports)

import argparse
import time
from pathlib import Path

import numpy as np

from drone_rl.environment.city import CityConfig, generate_city
from drone_rl.utils.config import load_config
from drone_rl.utils.logging import configure_logging, get_logger

_LOG = get_logger("generate_city")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/environment.yaml"),
                        help="YAML config providing the city: block")
    parser.add_argument("--seed", type=int, default=None, help="Single city seed to inspect")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Several city seeds")
    parser.add_argument("--gui", action="store_true", help="Open a PyBullet GUI window")
    parser.add_argument("--hold", type=float, default=20.0,
                        help="Seconds to keep the GUI window open (0 = until closed)")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save top-down PNGs to outputs/figures/cities")
    parser.add_argument("--check-determinism", action="store_true",
                        help="Regenerate each seed and assert the layout is identical")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="Config overrides, e.g. city.build_probability=0.5")
    return parser.parse_args()


def describe(seed: int, config: CityConfig) -> None:
    """Print a one-line summary of the city generated for ``seed``."""
    started = time.perf_counter()
    layout = generate_city(seed, config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    heights = [b.height for b in layout.buildings]
    _LOG.info(
        "seed %4d | %3d buildings | height %.1f-%.1f m (mean %.1f) | "
        "%5d free cells | %.1f ms",
        seed,
        layout.num_buildings,
        min(heights) if heights else 0.0,
        max(heights) if heights else 0.0,
        float(np.mean(heights)) if heights else 0.0,
        layout.free_positions.shape[0],
        elapsed_ms,
    )


def check_determinism(seeds: list[int], config: CityConfig) -> bool:
    """Assert that regenerating each seed reproduces an identical layout."""
    all_match = True
    for seed in seeds:
        first = generate_city(seed, config)
        second = generate_city(seed, config)
        same = first.buildings == second.buildings and np.array_equal(
            first.free_positions, second.free_positions
        )
        _LOG.info("seed %4d | determinism: %s", seed, "OK" if same else "MISMATCH")
        all_match &= same

    if len(seeds) >= 2:
        layouts = [generate_city(s, config) for s in seeds[:2]]
        distinct = layouts[0].buildings != layouts[1].buildings
        _LOG.info(
            "seeds %d vs %d produce different cities: %s",
            seeds[0], seeds[1], "OK" if distinct else "IDENTICAL (unexpected)",
        )
        all_match &= distinct
    return all_match


def save_figure(seed: int, config: CityConfig, output_dir: Path) -> Path:
    """Save a top-down PNG of one city."""
    from drone_rl.evaluation.visualization import plot_trajectory_topdown

    layout = generate_city(seed, config)
    return plot_trajectory_topdown(
        trajectories=[],
        buildings=layout.buildings,
        output_path=output_dir / f"city_seed_{seed}.png",
        title=f"Procedural city - seed {seed} ({layout.num_buildings} buildings)",
    )


def show_gui(
    seed: int,
    config: CityConfig,
    hold_seconds: float,
    render_settings: dict | None = None,
) -> None:
    """Open a PyBullet window displaying the generated city.

    Camera placement comes from the ``render:`` config block, so
    ``--set render.camera_pitch=-20`` works here exactly as it does for the
    other GUI scripts.
    """
    from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig

    settings = dict(render_settings or {})
    settings["mode"] = "gui"
    settings["show_trail"] = False  # nothing is flying, so there is no trail
    settings.pop("follow_drone", None)  # no drone to follow in this view

    layout = generate_city(seed, config)
    renderer = PyBulletRenderer(RenderConfig(**settings))
    try:
        renderer.build_city(layout)
        renderer.set_hud(
            [
                f"Procedural city - seed {seed}",
                f"{layout.num_buildings} buildings",
                f"{config.width:.0f} x {config.depth:.0f} m",
            ],
            position=np.array([0.0, 0.0, 22.0]),
        )
        _LOG.info("PyBullet GUI open for seed %d. Close the window or wait %.0fs.", seed, hold_seconds)
        deadline = time.time() + hold_seconds if hold_seconds > 0 else float("inf")
        while time.time() < deadline:
            time.sleep(0.05)
    except KeyboardInterrupt:
        _LOG.info("Interrupted.")
    finally:
        renderer.close()


def main() -> int:
    args = parse_args()
    configure_logging("INFO")

    config_dict = load_config(args.config, overrides=args.overrides)
    city_config = CityConfig.from_mapping(config_dict.get("city", {}))

    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None else 0]
    _LOG.info("City config: %s", city_config.as_dict())

    for seed in seeds:
        describe(seed, city_config)

    if args.check_determinism:
        _LOG.info("--- determinism check ---")
        if not check_determinism(seeds, city_config):
            _LOG.error("Determinism check FAILED")
            return 1

    if args.save_figures:
        output_dir = Path("outputs/figures/cities")
        for seed in seeds:
            path = save_figure(seed, city_config, output_dir)
            _LOG.info("Saved %s", path)

    if args.gui:
        show_gui(seeds[0], city_config, args.hold, config_dict.get("render"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
