"""Vectorised analytic geometry against axis-aligned boxes.

This module is the physics ground truth for the whole project. Because the
drone is kinematic (Phase 2 explicitly rules out aerodynamic simulation), the
only geometric queries the environment needs are:

* *Does a sphere of radius r at position p overlap any building?* -> collision.
* *How far along direction d from p until we hit something?* -> range sensor.

Both are solved in closed form here, which is exact, deterministic across
platforms, and roughly two orders of magnitude faster than driving a physics
engine for the same queries. The PyBullet renderer displays the identical
geometry, and ``tests/test_geometry.py`` asserts these routines agree with
PyBullet's own ``rayTest`` where PyBullet is installed.

All functions are pure and allocation-light; arrays are ``float64``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "raycast_boxes",
    "sphere_hits_boxes",
    "closest_distance_to_boxes",
    "segment_hits_boxes",
    "min_sq_distance_to_boxes",
    "norm3",
]

# Directions with a zero component would divide by zero in the slab test. We
# substitute a tiny value, which pushes the slab intersection to +/-inf and is
# the standard branch-free formulation of the algorithm.
_EPS = 1e-12


def norm3(vector: np.ndarray) -> float:
    """Euclidean length of a 3-vector.

    ``np.linalg.norm`` is a general-purpose dispatcher and costs roughly 10x
    this for a single small vector. The environment evaluates several norms per
    step across millions of steps, so the specialised version is worth having.
    """
    return float(np.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]))


def min_sq_distance_to_boxes(
    points: np.ndarray,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
) -> np.ndarray:
    """Squared distance from each point to the nearest box surface.

    Working in squared distances avoids a square root in the collision path,
    where the result is only ever compared against a radius. The closest point
    of an AABB to a query point is the per-axis clamp of that point, which makes
    this a single vectorised expression.

    Parameters
    ----------
    points:
        ``(P, 3)`` query points.
    box_mins, box_maxs:
        ``(B, 3)`` box corners.

    Returns
    -------
    np.ndarray
        ``(P,)`` squared distances; ``inf`` when there are no boxes.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    box_mins = np.asarray(box_mins, dtype=np.float64).reshape(-1, 3)
    box_maxs = np.asarray(box_maxs, dtype=np.float64).reshape(-1, 3)
    if box_mins.shape[0] == 0:
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    expanded = points[:, None, :]  # (P, 1, 3)
    clamped = np.clip(expanded, box_mins[None, :, :], box_maxs[None, :, :])
    delta = clamped - expanded
    return np.einsum("pbi,pbi->pb", delta, delta).min(axis=1)


def raycast_boxes(
    origins: np.ndarray,
    directions: np.ndarray,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
    max_range: float,
    ground_z: float | None = 0.0,
    ceiling_z: float | None = None,
    directions_are_unit: bool = False,
) -> np.ndarray:
    """Cast rays against a set of AABBs and return the hit distance per ray.

    Uses the branch-free *slab method*: for each axis the ray is clipped against
    the pair of parallel planes bounding the box, and the intersection is
    non-empty when the largest entry parameter does not exceed the smallest exit
    parameter.

    Parameters
    ----------
    origins:
        ``(R, 3)`` ray origins, or ``(3,)`` for a single shared origin.
    directions:
        ``(R, 3)`` ray directions. Need not be unit length; they are normalised
        internally so returned distances are in metres.
    box_mins, box_maxs:
        ``(B, 3)`` lower/upper corners of the boxes. May be empty.
    max_range:
        Sensor range. Rays that hit nothing return exactly ``max_range``.
    ground_z:
        When not ``None``, the ground plane at this height is treated as an
        infinite horizontal obstacle (so a downward ray reports altitude).
    ceiling_z:
        When not ``None``, an infinite horizontal plane at this height bounds
        the navigable volume from above.
    directions_are_unit:
        Set by callers that already supply unit directions (the sensor array
        rotates a fixed unit basis, so its rays always are) to skip the
        normalisation.

    Returns
    -------
    np.ndarray
        ``(R,)`` distances in ``[0, max_range]``.
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    box_mins = np.asarray(box_mins, dtype=np.float64).reshape(-1, 3)
    box_maxs = np.asarray(box_maxs, dtype=np.float64).reshape(-1, 3)

    if directions_are_unit:
        unit_dirs = directions
    else:
        norms = np.sqrt(np.einsum("ij,ij->i", directions, directions))[:, None]
        unit_dirs = directions / np.where(norms < _EPS, 1.0, norms)

    num_rays = unit_dirs.shape[0]
    max_range = float(max_range)
    best = np.full(num_rays, max_range, dtype=np.float64)

    # A single shared origin (all rays leaving the drone) broadcasts against the
    # per-ray directions, so no explicit tiling is needed.
    inv_dir = 1.0 / np.where(np.abs(unit_dirs) < _EPS, _EPS, unit_dirs)  # (R, 3)

    if box_mins.shape[0] > 0:
        # (R, B, 3) parameters where each ray crosses each slab plane.
        t_lower = (box_mins[None, :, :] - origins[:, None, :]) * inv_dir[:, None, :]
        t_upper = (box_maxs[None, :, :] - origins[:, None, :]) * inv_dir[:, None, :]

        t_near = np.minimum(t_lower, t_upper).max(axis=2)  # entry parameter
        t_far = np.maximum(t_lower, t_upper).min(axis=2)  # exit parameter

        # A hit requires a non-empty overlap lying ahead of the origin.
        # ``t_near < 0 <= t_far`` means the origin is inside the box: distance 0.
        entry = np.where(t_near < 0.0, 0.0, t_near)
        valid = (t_far >= np.maximum(t_near, 0.0)) & (entry <= max_range)
        best = np.minimum(best, np.where(valid, entry, np.inf).min(axis=1))

    # Ground and ceiling planes. inv_dir already guards the zero-dz case, so no
    # error-state handling is needed here.
    dz = unit_dirs[:, 2]
    inv_dz = inv_dir[:, 2]
    origin_z = origins[:, 2]
    for plane_z, sign in ((ground_z, -1.0), (ceiling_z, 1.0)):
        if plane_z is None:
            continue
        t_plane = (plane_z - origin_z) * inv_dz
        # Only rays travelling toward the plane can hit it.
        plane_hit = (dz * sign > _EPS) & (t_plane >= 0.0) & (t_plane <= max_range)
        best = np.where(plane_hit, np.minimum(best, t_plane), best)

    return np.clip(best, 0.0, max_range, out=best)


def closest_distance_to_boxes(
    point: np.ndarray,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
) -> float:
    """Return the Euclidean distance from ``point`` to the nearest box surface.

    Zero when the point lies inside a box. Used for the "clearance" diagnostic
    exposed in ``info``; the collision path uses the squared-distance form.
    """
    if len(box_mins) == 0:
        return float("inf")
    point = np.asarray(point, dtype=np.float64).reshape(1, 3)
    return float(np.sqrt(min_sq_distance_to_boxes(point, box_mins, box_maxs)[0]))


def sphere_hits_boxes(
    center: np.ndarray,
    radius: float,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
) -> bool:
    """Test whether a sphere overlaps any box (the drone's collision test)."""
    if len(box_mins) == 0:
        return False
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    return bool(min_sq_distance_to_boxes(center, box_mins, box_maxs)[0] <= radius * radius)


def segment_hits_boxes(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    box_mins: np.ndarray,
    box_maxs: np.ndarray,
    substeps: int = 4,
) -> bool:
    """Swept collision test for a sphere moving from ``start`` to ``end``.

    A single end-point test can tunnel through a thin wall when the per-step
    displacement exceeds the wall thickness. Sampling the segment at
    ``substeps`` intermediate points bounds the miss distance by
    ``|end - start| / substeps``, which for the default action displacement is
    well under the thinnest building dimension.

    All samples are tested in one vectorised call rather than in a Python loop.
    """
    if len(box_mins) == 0:
        return False
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)

    steps = max(1, int(substeps))
    # Skip fraction 0: the start pose was already collision-free last step.
    fractions = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
    samples = start[None, :] + (end - start)[None, :] * fractions  # (steps, 3)
    return bool(
        (min_sq_distance_to_boxes(samples, box_mins, box_maxs) <= radius * radius).any()
    )
