"""Tests for the frame overlay, especially the world-to-pixel projection.

A transposed or mis-ordered matrix here would not raise - it would silently
draw the trajectory in the wrong place in every recorded clip, which is exactly
the kind of bug that survives to the presentation.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from drone_rl.evaluation.overlay import FrameOverlay, project_points

requires_pybullet = pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None, reason="PyBullet not installed"
)

WIDTH, HEIGHT = 320, 200


@pytest.fixture
def matrices():
    """View/projection looking at the origin from +x, with the camera above."""
    import pybullet as p

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.0],
        distance=20.0,
        yaw=0.0,
        pitch=-30.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=400.0
    )
    return view, projection


@requires_pybullet
def test_camera_target_projects_to_frame_centre(matrices):
    """The point the camera looks at must land at the centre of the image."""
    view, projection = matrices
    pixels, visible = project_points(
        np.array([[0.0, 0.0, 0.0]]), view, projection, WIDTH, HEIGHT
    )
    assert visible[0]
    assert pixels[0, 0] == pytest.approx(WIDTH / 2, abs=1.0)
    assert pixels[0, 1] == pytest.approx(HEIGHT / 2, abs=1.0)


def camera_basis(view_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``(position, right, forward)`` in world space from a view matrix.

    Derived rather than assumed, because PyBullet's yaw convention is not
    obvious and hard-coding a guess makes the test assert the wrong thing.
    """
    view = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    rotation, translation = view[:3, :3], view[:3, 3]
    position = -rotation.T @ translation
    right = rotation[0, :]  # camera +x axis expressed in world coordinates
    forward = -rotation[2, :]  # camera looks down its own -z
    return position, right, forward


@requires_pybullet
def test_points_behind_camera_are_marked_invisible(matrices):
    """Behind-camera points must be dropped, not wrapped to a valid pixel."""
    view, projection = matrices
    position, _right, forward = camera_basis(view)

    behind = position - forward * 50.0
    _pixels, visible = project_points(
        behind.reshape(1, 3), view, projection, WIDTH, HEIGHT
    )
    assert not visible[0]

    in_front = position + forward * 10.0
    _pixels, visible = project_points(
        in_front.reshape(1, 3), view, projection, WIDTH, HEIGHT
    )
    assert visible[0]


@requires_pybullet
def test_higher_points_project_higher_in_the_image(matrices):
    """Image y grows downward, so greater altitude means a smaller y pixel."""
    view, projection = matrices
    pixels, visible = project_points(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]), view, projection, WIDTH, HEIGHT
    )
    assert visible.all()
    assert pixels[1, 1] < pixels[0, 1]


@requires_pybullet
def test_camera_right_direction_projects_right_of_centre(matrices):
    """Offsetting along the camera's own +x axis must move right in the image."""
    view, projection = matrices
    _position, right, _forward = camera_basis(view)

    pixels, visible = project_points(
        np.vstack([np.zeros(3), right * 4.0]), view, projection, WIDTH, HEIGHT
    )
    assert visible.all()
    assert pixels[0, 0] == pytest.approx(WIDTH / 2, abs=1.0)
    assert pixels[1, 0] > WIDTH / 2


@requires_pybullet
def test_projection_is_consistent_with_distance(matrices):
    """A point twice as far off-axis projects roughly twice as far off-centre."""
    view, projection = matrices
    _position, right, _forward = camera_basis(view)

    pixels, visible = project_points(
        np.vstack([right * 2.0, right * 4.0]), view, projection, WIDTH, HEIGHT
    )
    assert visible.all()
    near_offset = abs(pixels[0, 0] - WIDTH / 2)
    far_offset = abs(pixels[1, 0] - WIDTH / 2)
    assert far_offset == pytest.approx(2 * near_offset, rel=0.05)


def test_project_points_handles_empty_input():
    pixels, visible = project_points(np.zeros((0, 3)), [0.0] * 16, [0.0] * 16, 10, 10)
    assert pixels.shape == (0, 2)
    assert visible.shape == (0,)


@requires_pybullet
def test_overlay_draws_without_changing_frame_shape(matrices):
    view, projection = matrices
    overlay = FrameOverlay(WIDTH, HEIGHT)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    result = overlay.draw(
        frame,
        lines=["STAGE: test", "Status   TARGET REACHED"],
        trajectory=np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]]),
        drone_position=np.array([2.0, 2.0, 3.0]),
        target_position=np.array([0.0, 0.0, 2.0]),
        start_position=np.array([0.0, 0.0, 1.0]),
        view_matrix=view,
        projection_matrix=projection,
        status_colour=(0, 255, 0),
    )
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Something must actually have been drawn onto the black frame.
    assert result.sum() > 0


def test_overlay_text_only_on_black_frame():
    """The HUD must render even without projection matrices."""
    overlay = FrameOverlay(WIDTH, HEIGHT)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result = overlay.draw(frame, lines=["Episode 1", "Status   COLLISION"])
    assert result.sum() > 0
