"""Phase 1 - procedural 3D city generation.

The city is produced as pure geometry *data* (a list of axis-aligned boxes plus
free-space metadata). Nothing here imports PyBullet: the same
:class:`CityLayout` drives the fast analytic physics core used for training and
the PyBullet renderer used for demos, which guarantees that what you watch in
the GUI is the world the agent actually trained in.

Layout algorithm
----------------
1. The ground plane is divided into a grid of *city blocks* separated by roads
   of configurable width. Roads are the guaranteed-free corridors that make the
   navigation problem solvable.
2. Each block is subdivided into plots; a plot receives a building with
   probability ``build_probability``. Empty plots become open space (plazas).
3. Building footprints are inset from their plot by a random margin and given a
   height sampled from a configurable range, with taller buildings biased
   toward the city centre so the skyline is not uniform.

Every random draw comes from a single ``numpy.random.Generator`` seeded by the
city seed, so a seed maps deterministically to a layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = ["Building", "CityLayout", "CityConfig", "generate_city"]


@dataclass(frozen=True)
class Building:
    """An axis-aligned box obstacle resting on the ground plane.

    Attributes
    ----------
    center_x, center_y:
        Footprint centre in world coordinates (metres).
    half_extent_x, half_extent_y:
        Half-width and half-depth of the footprint.
    height:
        Height above ground. The box spans ``z in [0, height]``.
    """

    center_x: float
    center_y: float
    half_extent_x: float
    half_extent_y: float
    height: float

    @property
    def min_corner(self) -> tuple[float, float, float]:
        """Lower corner of the axis-aligned bounding box."""
        return (
            self.center_x - self.half_extent_x,
            self.center_y - self.half_extent_y,
            0.0,
        )

    @property
    def max_corner(self) -> tuple[float, float, float]:
        """Upper corner of the axis-aligned bounding box."""
        return (
            self.center_x + self.half_extent_x,
            self.center_y + self.half_extent_y,
            self.height,
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Centre of the 3D box (note: ``z`` is half the height)."""
        return (self.center_x, self.center_y, self.height * 0.5)

    @property
    def half_extents(self) -> tuple[float, float, float]:
        """Half extents of the 3D box, the form PyBullet's box shape expects."""
        return (self.half_extent_x, self.half_extent_y, self.height * 0.5)


@dataclass(frozen=True)
class CityConfig:
    """All parameters controlling procedural city generation.

    Distances are metres. ``width``/``depth`` describe the full extent of the
    city, which is centred on the origin, so coordinates run from ``-width/2``
    to ``+width/2``.
    """

    width: float = 60.0
    depth: float = 60.0
    max_altitude: float = 25.0
    block_size: float = 14.0
    road_width: float = 6.0
    plots_per_block: int = 2
    build_probability: float = 0.72
    building_min_height: float = 4.0
    building_max_height: float = 18.0
    building_margin: float = 0.8
    center_height_bias: float = 0.45
    min_buildings: int = 6
    boundary_wall_height: float = 30.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CityConfig":
        """Build a config from a YAML block, ignoring unrelated keys.

        Seed-range keys (``train_seed_start`` etc.) live in the same YAML block
        but describe the experiment rather than the geometry, so they are
        filtered out here.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class CityLayout:
    """A generated city: obstacle geometry plus the free space around it.

    ``buildings`` are the obstacles. ``free_positions`` is a coarse grid of
    ground-level points that lie outside every building footprint (with a safety
    margin), used to sample start and target locations that are not born inside
    a wall.
    """

    seed: int
    config: CityConfig
    buildings: list[Building] = field(default_factory=list)
    free_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))

    # ---- geometry accessors -------------------------------------------------

    @property
    def bounds_min(self) -> np.ndarray:
        """Lower world bound ``[x, y, z]`` of the navigable volume."""
        return np.array([-self.config.width / 2.0, -self.config.depth / 2.0, 0.0])

    @property
    def bounds_max(self) -> np.ndarray:
        """Upper world bound ``[x, y, z]`` of the navigable volume."""
        return np.array(
            [self.config.width / 2.0, self.config.depth / 2.0, self.config.max_altitude]
        )

    @property
    def num_buildings(self) -> int:
        return len(self.buildings)

    def box_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mins, maxs)`` arrays of shape ``(N, 3)`` for all buildings.

        This vectorised form is what the analytic collision and ray-casting code
        consumes; building it once per episode keeps the hot loop allocation-free.
        """
        if not self.buildings:
            empty = np.zeros((0, 3), dtype=np.float64)
            return empty, empty.copy()
        mins = np.array([b.min_corner for b in self.buildings], dtype=np.float64)
        maxs = np.array([b.max_corner for b in self.buildings], dtype=np.float64)
        return mins, maxs

    def max_building_height(self) -> float:
        """Tallest building in the city, or 0 for an empty city."""
        return max((b.height for b in self.buildings), default=0.0)

    def __iter__(self) -> Iterator[Building]:
        return iter(self.buildings)

    def __len__(self) -> int:
        return len(self.buildings)

    def summary(self) -> dict[str, Any]:
        """Compact description used by logging and ``scripts/generate_city.py``."""
        return {
            "seed": self.seed,
            "num_buildings": self.num_buildings,
            "max_building_height": round(self.max_building_height(), 2),
            "free_positions": int(self.free_positions.shape[0]),
            "width": self.config.width,
            "depth": self.config.depth,
            "max_altitude": self.config.max_altitude,
        }


def _block_origins(extent: float, block_size: float, road_width: float) -> list[float]:
    """Return the lower coordinates of each block along one axis.

    Blocks tile the axis with roads between them; the whole pattern is centred
    so the city is symmetric about the origin.
    """
    stride = block_size + road_width
    count = max(1, int((extent + road_width) // stride))
    used = count * stride - road_width
    start = -used / 2.0
    return [start + i * stride for i in range(count)]


def _sample_building_height(
    rng: np.random.Generator,
    config: CityConfig,
    center_x: float,
    center_y: float,
) -> float:
    """Sample a height, biasing buildings near the city centre to be taller.

    The bias produces a downtown-like skyline instead of uniform noise, which
    makes altitude changes a meaningful part of the navigation problem.
    """
    half_diagonal = float(np.hypot(config.width, config.depth) * 0.5) or 1.0
    radial = float(np.hypot(center_x, center_y)) / half_diagonal
    centrality = float(np.clip(1.0 - radial, 0.0, 1.0))

    base = rng.uniform(config.building_min_height, config.building_max_height)
    span = config.building_max_height - config.building_min_height
    boosted = base + config.center_height_bias * centrality * span
    return float(np.clip(boosted, config.building_min_height, config.building_max_height))


def _generate_buildings(rng: np.random.Generator, config: CityConfig) -> list[Building]:
    """Place buildings on a road-separated block grid."""
    buildings: list[Building] = []
    x_origins = _block_origins(config.width, config.block_size, config.road_width)
    y_origins = _block_origins(config.depth, config.block_size, config.road_width)
    plot_size = config.block_size / max(1, config.plots_per_block)

    for bx in x_origins:
        for by in y_origins:
            for ix in range(config.plots_per_block):
                for iy in range(config.plots_per_block):
                    if rng.random() > config.build_probability:
                        continue  # open plaza
                    plot_min_x = bx + ix * plot_size
                    plot_min_y = by + iy * plot_size

                    # Inset the footprint by a random margin on each side.
                    margin = config.building_margin
                    max_inset = max(0.0, plot_size * 0.5 - 0.8)
                    inset_x = float(rng.uniform(margin, max(margin, max_inset)))
                    inset_y = float(rng.uniform(margin, max(margin, max_inset)))

                    half_x = max(0.6, plot_size * 0.5 - inset_x)
                    half_y = max(0.6, plot_size * 0.5 - inset_y)
                    center_x = plot_min_x + plot_size * 0.5
                    center_y = plot_min_y + plot_size * 0.5

                    height = _sample_building_height(rng, config, center_x, center_y)
                    buildings.append(
                        Building(
                            center_x=float(center_x),
                            center_y=float(center_y),
                            half_extent_x=float(half_x),
                            half_extent_y=float(half_y),
                            height=float(height),
                        )
                    )
    return buildings


def _compute_free_positions(
    buildings: Sequence[Building],
    config: CityConfig,
    clearance: float,
    grid_step: float = 1.5,
) -> np.ndarray:
    """Grid-sample ground positions that clear every building by ``clearance``.

    Returned as an ``(M, 2)`` array of ``(x, y)`` points. Start and target
    sampling draws from this set, so neither can spawn inside a building.
    """
    half_w = config.width / 2.0
    half_d = config.depth / 2.0
    margin = clearance + 0.5
    xs = np.arange(-half_w + margin, half_w - margin + 1e-9, grid_step)
    ys = np.arange(-half_d + margin, half_d - margin + 1e-9, grid_step)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    if not buildings:
        return points

    mins = np.array([b.min_corner[:2] for b in buildings], dtype=np.float64)
    maxs = np.array([b.max_corner[:2] for b in buildings], dtype=np.float64)
    # A point is blocked if it falls inside any inflated footprint.
    inside = np.all(
        (points[:, None, :] >= mins[None, :, :] - clearance)
        & (points[:, None, :] <= maxs[None, :, :] + clearance),
        axis=2,
    )
    free_mask = ~inside.any(axis=1)
    return points[free_mask]


def generate_city(
    seed: int,
    config: CityConfig | None = None,
    clearance: float = 1.2,
) -> CityLayout:
    """Generate a deterministic city for ``seed``.

    The same ``seed`` and ``config`` always yield an identical layout; different
    seeds yield different layouts. If the random draw produces fewer than
    ``config.min_buildings`` buildings the city is regenerated with a derived
    seed, so degenerate empty cities never reach training.

    Parameters
    ----------
    seed:
        City seed. Determines the entire layout.
    config:
        Geometry parameters; defaults to :class:`CityConfig`.
    clearance:
        Safety radius (metres) enforced when computing free spawn positions.
    """
    cfg = config or CityConfig()
    attempt = 0
    while True:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), attempt]))
        buildings = _generate_buildings(rng, cfg)
        if len(buildings) >= cfg.min_buildings or attempt >= 8:
            break
        attempt += 1

    free_positions = _compute_free_positions(buildings, cfg, clearance=clearance)
    if free_positions.shape[0] == 0:
        # Degenerate: fall back to a coarse unfiltered grid so the episode can
        # still start. This is only reachable with pathological configs.
        free_positions = _compute_free_positions([], cfg, clearance=clearance)

    return CityLayout(
        seed=int(seed),
        config=cfg,
        buildings=buildings,
        free_positions=free_positions,
    )
