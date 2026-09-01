"""Analytic geometry tests, including a parity check against PyBullet.

The parity test is the guarantee that the fast analytic core used for training
describes the same world the PyBullet GUI shows during a demo. It is skipped
automatically when PyBullet is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from drone_rl.environment.city import generate_city
from drone_rl.environment.geometry import (
    closest_distance_to_boxes,
    min_sq_distance_to_boxes,
    norm3,
    raycast_boxes,
    segment_hits_boxes,
    sphere_hits_boxes,
)


@pytest.fixture
def unit_box():
    """A 2x2x2 box centred at the origin."""
    mins = np.array([[-1.0, -1.0, -1.0]])
    maxs = np.array([[1.0, 1.0, 1.0]])
    return mins, maxs


def test_norm3_matches_numpy():
    vector = np.array([3.0, -4.0, 12.0])
    assert norm3(vector) == pytest.approx(np.linalg.norm(vector))


def test_ray_hits_box_at_expected_distance(unit_box):
    mins, maxs = unit_box
    distance = raycast_boxes(
        origins=np.array([[-5.0, 0.0, 0.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=None,
    )
    assert distance[0] == pytest.approx(4.0)  # from x=-5 to the face at x=-1


def test_ray_missing_box_returns_max_range(unit_box):
    mins, maxs = unit_box
    distance = raycast_boxes(
        origins=np.array([[-5.0, 5.0, 0.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=None,
    )
    assert distance[0] == pytest.approx(20.0)


def test_ray_pointing_away_does_not_hit(unit_box):
    """A box behind the origin must not register: t < 0 is not a hit."""
    mins, maxs = unit_box
    distance = raycast_boxes(
        origins=np.array([[-5.0, 0.0, 0.0]]),
        directions=np.array([[-1.0, 0.0, 0.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=None,
    )
    assert distance[0] == pytest.approx(20.0)


def test_downward_ray_reports_altitude(unit_box):
    """With a ground plane at z=0 a downward ray measures height."""
    mins = np.zeros((0, 3))
    maxs = np.zeros((0, 3))
    distance = raycast_boxes(
        origins=np.array([[10.0, 10.0, 6.5]]),
        directions=np.array([[0.0, 0.0, -1.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=0.0,
    )
    assert distance[0] == pytest.approx(6.5)


def test_ceiling_plane_is_detected():
    distance = raycast_boxes(
        origins=np.array([[0.0, 0.0, 5.0]]),
        directions=np.array([[0.0, 0.0, 1.0]]),
        box_mins=np.zeros((0, 3)),
        box_maxs=np.zeros((0, 3)),
        max_range=50.0,
        ground_z=0.0,
        ceiling_z=25.0,
    )
    assert distance[0] == pytest.approx(20.0)


def test_ray_from_inside_box_returns_zero(unit_box):
    mins, maxs = unit_box
    distance = raycast_boxes(
        origins=np.array([[0.0, 0.0, 0.0]]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=None,
    )
    assert distance[0] == pytest.approx(0.0)


def test_non_unit_direction_is_normalised(unit_box):
    """Distances must be in metres regardless of the direction's magnitude."""
    mins, maxs = unit_box
    distance = raycast_boxes(
        origins=np.array([[-5.0, 0.0, 0.0]]),
        directions=np.array([[7.3, 0.0, 0.0]]),
        box_mins=mins,
        box_maxs=maxs,
        max_range=20.0,
        ground_z=None,
    )
    assert distance[0] == pytest.approx(4.0)


def test_closest_distance_and_sphere_overlap(unit_box):
    mins, maxs = unit_box
    assert closest_distance_to_boxes(np.array([3.0, 0.0, 0.0]), mins, maxs) == pytest.approx(2.0)
    assert closest_distance_to_boxes(np.array([0.0, 0.0, 0.0]), mins, maxs) == pytest.approx(0.0)
    assert sphere_hits_boxes(np.array([1.4, 0.0, 0.0]), 0.5, mins, maxs)
    assert not sphere_hits_boxes(np.array([1.8, 0.0, 0.0]), 0.5, mins, maxs)


def test_min_sq_distance_handles_multiple_points(unit_box):
    mins, maxs = unit_box
    points = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    result = min_sq_distance_to_boxes(points, mins, maxs)
    assert result == pytest.approx([4.0, 0.0, 16.0])


def test_min_sq_distance_with_no_boxes_is_infinite():
    result = min_sq_distance_to_boxes(
        np.array([[0.0, 0.0, 0.0]]), np.zeros((0, 3)), np.zeros((0, 3))
    )
    assert np.isinf(result).all()


def test_swept_test_catches_tunnelling(unit_box):
    """A move that jumps clean over a box must still register as a collision."""
    mins, maxs = unit_box
    start = np.array([-4.0, 0.0, 0.0])
    end = np.array([4.0, 0.0, 0.0])
    # The endpoints are both outside the box, so a naive endpoint-only test
    # would miss this.
    assert not sphere_hits_boxes(start, 0.3, mins, maxs)
    assert not sphere_hits_boxes(end, 0.3, mins, maxs)
    assert segment_hits_boxes(start, end, 0.3, mins, maxs, substeps=8)


def test_swept_test_allows_clear_move(unit_box):
    mins, maxs = unit_box
    assert not segment_hits_boxes(
        np.array([-4.0, 5.0, 0.0]), np.array([4.0, 5.0, 0.0]), 0.3, mins, maxs, substeps=8
    )


# ---------------------------------------------------------------------------
# PyBullet parity
#
# Only these two tests need PyBullet. The skip marker is applied per-test
# rather than at module level, because a module-level ``importorskip`` would
# silently skip every pure-geometry test above it as well.
# ---------------------------------------------------------------------------

requires_pybullet = pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None,
    reason="PyBullet not installed; analytic/PyBullet parity check skipped.",
)


@requires_pybullet
def test_analytic_rays_match_pybullet_raytest():
    """The analytic ray core must agree with PyBullet on the same geometry.

    This is what licenses using the fast core for training while showing the
    PyBullet scene in demos: both describe the same city.
    """
    from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig

    layout = generate_city(4)
    mins, maxs = layout.box_arrays()
    max_range = 15.0

    renderer = PyBulletRenderer(RenderConfig(mode="direct"))
    try:
        renderer.build_city(layout)

        rng = np.random.default_rng(0)
        origins, endpoints, directions = [], [], []
        for _ in range(60):
            origin = np.array(
                [rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(2.0, 18.0)]
            )
            # Skip origins inside a building. PyBullet's rayTest ignores the
            # body a ray starts inside and reports "no hit", whereas the
            # analytic core reports distance 0 (see
            # test_ray_from_inside_box_returns_zero). The analytic behaviour is
            # the one we want for a sensor, but the two are not comparable, and
            # the drone can never legitimately occupy that state anyway: a
            # collision terminates the episode first.
            if min_sq_distance_to_boxes(origin[None, :], mins, maxs)[0] <= 0.0:
                continue
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            origins.append(origin)
            directions.append(direction)
            endpoints.append(origin + direction * max_range)

        assert len(origins) >= 30, "Too few valid sample origins to be a meaningful check"
        origins_arr = np.array(origins)
        directions_arr = np.array(directions)

        analytic = np.array(
            [
                raycast_boxes(
                    origins=origins_arr[i : i + 1],
                    directions=directions_arr[i : i + 1],
                    box_mins=mins,
                    box_maxs=maxs,
                    max_range=max_range,
                    ground_z=None,  # PyBullet's ground slab is handled below
                    ceiling_z=None,
                )[0]
                for i in range(len(origins))
            ]
        )
        fractions = renderer.raytest_batch(origins_arr, np.array(endpoints))
        bullet = fractions * max_range

        # PyBullet's scene also contains the ground slab, which the analytic
        # call above deliberately excludes. Compare only rays that did not end
        # on the ground (i.e. those whose bullet hit is at z > 0.05).
        hit_points = origins_arr + directions_arr * bullet[:, None]
        above_ground = hit_points[:, 2] > 0.05

        np.testing.assert_allclose(
            analytic[above_ground], bullet[above_ground], atol=0.02, rtol=0.0
        )
    finally:
        renderer.close()


@requires_pybullet
def test_pybullet_scene_contains_expected_bodies():
    """The renderer must instantiate one body per building plus ground+drone."""
    import pybullet

    from drone_rl.environment.renderer import PyBulletRenderer, RenderConfig

    layout = generate_city(6)
    renderer = PyBulletRenderer(RenderConfig(mode="direct"))
    try:
        renderer.build_city(layout)
        num_bodies = pybullet.getNumBodies(physicsClientId=renderer.client_id)
        assert num_bodies == layout.num_buildings + 2  # buildings + ground + drone
    finally:
        renderer.close()
