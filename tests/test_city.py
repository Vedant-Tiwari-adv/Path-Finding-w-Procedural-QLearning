"""Phase 1 tests: procedural city generation."""

from __future__ import annotations

import numpy as np
import pytest

from drone_rl.environment.city import CityConfig, generate_city


def test_same_seed_gives_identical_city():
    """The central determinism guarantee: seed -> layout is a function."""
    a = generate_city(7)
    b = generate_city(7)
    assert a.buildings == b.buildings
    assert np.array_equal(a.free_positions, b.free_positions)


def test_different_seeds_give_different_cities():
    layouts = [generate_city(seed) for seed in range(6)]
    signatures = {tuple(sorted((b.center_x, b.center_y, b.height) for b in c)) for c in layouts}
    assert len(signatures) == len(layouts), "Distinct seeds produced duplicate cities"


def test_city_respects_configured_bounds():
    config = CityConfig(width=50.0, depth=40.0, max_altitude=20.0)
    layout = generate_city(3, config)

    assert layout.num_buildings >= config.min_buildings
    for building in layout.buildings:
        assert -config.width / 2 - 1e-6 <= building.min_corner[0]
        assert building.max_corner[0] <= config.width / 2 + 1e-6
        assert -config.depth / 2 - 1e-6 <= building.min_corner[1]
        assert building.max_corner[1] <= config.depth / 2 + 1e-6
        assert config.building_min_height <= building.height <= config.building_max_height


def test_free_positions_are_outside_buildings():
    """Start/target sampling relies on this: no free cell may sit in a wall."""
    clearance = 1.0
    layout = generate_city(11, clearance=clearance)
    mins, maxs = layout.box_arrays()

    assert layout.free_positions.shape[0] > 0
    for point in layout.free_positions[::17]:  # subsample for speed
        inside = np.all(
            (point >= mins[:, :2] - clearance) & (point <= maxs[:, :2] + clearance), axis=1
        )
        assert not inside.any(), f"Free position {point} lies inside a building footprint"


def test_box_arrays_match_building_list():
    layout = generate_city(5)
    mins, maxs = layout.box_arrays()
    assert mins.shape == (layout.num_buildings, 3)
    assert maxs.shape == (layout.num_buildings, 3)
    assert np.all(maxs >= mins)
    # z always starts at ground level and ends at the building height.
    assert np.allclose(mins[:, 2], 0.0)
    assert np.allclose(maxs[:, 2], [b.height for b in layout.buildings])


def test_roads_leave_open_corridors():
    """Roads must remain free, otherwise many episodes would be unsolvable."""
    config = CityConfig(width=60.0, depth=60.0, block_size=14.0, road_width=6.0)
    layout = generate_city(2, config)
    mins, maxs = layout.box_arrays()

    # Sample the mid-line of the road between the first two block columns.
    stride = config.block_size + config.road_width
    road_x = -stride / 2.0 + config.block_size / 2.0 + config.road_width / 2.0
    for y in np.linspace(-config.depth / 2 + 4, config.depth / 2 - 4, 25):
        point = np.array([road_x, y])
        inside = np.all((point >= mins[:, :2]) & (point <= maxs[:, :2]), axis=1)
        assert not inside.any(), f"Road point {point} is blocked by a building"


@pytest.mark.parametrize("seed", [0, 42, 799, 800, 999])
def test_generation_is_fast_enough_for_training(seed):
    layout = generate_city(seed)
    assert layout.num_buildings > 0
    assert layout.free_positions.shape[0] > 50
