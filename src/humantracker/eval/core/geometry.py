"""Geometry helpers shared by every tracker backend.

Quaternions use MuJoCo's scalar-first ``(w, x, y, z)`` layout throughout; scipy
expects scalar-last, so :func:`wxyz_to_xyzw` sits between the two. That reorder
is done explicitly rather than with scipy's ``scalar_first=`` keyword, which
requires scipy >= 1.14 while this project supports >= 1.13.

:func:`quat2mat` and :func:`quat_to_mat` are *not* interchangeable. scipy
normalises the quaternion it is given; the outer-product form does not. On unit
quaternions the two agree to ~7e-16, but on non-unit input they diverge without
bound. Both are kept because :func:`quat_to_mat` feeds the policy observation
and the reward model's trained feature layout, where the arithmetic is fixed by
those contracts and must not drift.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "wrap_to_pi",
    "wxyz_to_xyzw",
    "quat2mat",
    "quat2euler",
    "quat_to_mat",
    "quat_conj",
    "quat_mul",
    "quat_rotate",
    "batch_base2navi",
    "get_sensor_data",
    "render_frame",
]


def wrap_to_pi(ang):
    """Wrap angle(s) into ``(-pi, pi]``."""
    return np.arctan2(np.sin(ang), np.cos(ang))


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reorder scalar-first quaternion(s) to the scalar-last layout scipy wants."""
    return np.roll(np.asarray(quat_wxyz), -1, axis=-1)


def quat2mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` rotation matrices."""
    return R.from_quat(wxyz_to_xyzw(quat_wxyz)).as_matrix()


def quat2euler(quat_wxyz: np.ndarray, order: str = "xyz", degrees: bool = False) -> np.ndarray:
    """``(..., 4)`` wxyz quaternions to Euler angles in the given ``order``."""
    return R.from_quat(wxyz_to_xyzw(quat_wxyz)).as_euler(order, degrees=degrees)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from one wxyz quaternion, without normalising it.

    Distinct from :func:`quat2mat` on purpose -- see the module docstring.
    """
    q = np.outer(q, q)
    return np.array([
        [q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3], 2 * (q[1, 2] - q[0, 3]), 2 * (q[1, 3] + q[0, 2])],
        [2 * (q[1, 2] + q[0, 3]), q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3], 2 * (q[2, 3] - q[0, 1])],
        [2 * (q[1, 3] - q[0, 2]), 2 * (q[2, 3] + q[0, 1]), q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3]],
    ])


def quat_conj(q):
    """Conjugate of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    """Hamilton product ``q1 * q2``, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate(q, v):
    """Rotate 3-vector ``v`` by wxyz quaternion ``q``."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def batch_base2navi(base2world: np.ndarray) -> np.ndarray:
    """Gravity-aligned (navigation) frame rotations from base rotation matrices.

    Keeps the base x-axis heading, drops its tilt, and re-orthogonalises against
    world +z. Degenerate orientations -- the base's local x-axis pointing exactly
    along world +/-z, e.g. a handstand or a flip's inverted apex -- have no defined
    heading; those entries fall back to the identity, the same convention
    ``core/rm_feature_extractor.py``'s ``_base2navi_single`` already uses for this
    exact singularity.
    """
    batch_shape = base2world.shape[:-2]
    x_proj = base2world[..., :3, 0].reshape(-1, 3).copy()
    x_proj[:, 2] = 0.0
    norm = np.linalg.norm(x_proj, axis=-1)
    degenerate = norm < 1e-9
    x_proj[degenerate] = (1.0, 0.0, 0.0)
    x_proj[~degenerate] /= norm[~degenerate, None]

    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    x_axis = np.cross(y_axis, z_axis)

    result = np.stack((x_axis, y_axis, np.broadcast_to(z_axis, x_axis.shape)), axis=-1)
    result[degenerate] = np.eye(3)
    return result.reshape(*batch_shape, 3, 3)


def get_sensor_data(mj_model, mj_data, sensor_name: str) -> np.ndarray:
    """The named sensor's slice of ``mj_data.sensordata``, as a copy.

    The copy is deliberate: ``sensordata`` is overwritten by every ``mj_step``,
    so a view would change under any caller that holds it across a step.
    """
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim].copy()


def render_frame(renderer, mj_data, cam, cam_distance=3.0, cam_azimuth=90, cam_elevation=-20):
    """Render one offscreen frame with the camera tracking the robot root."""
    cam.lookat[:] = mj_data.qpos[:3].copy()
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation
    renderer.update_scene(mj_data, camera=cam)
    return renderer.render().copy()
