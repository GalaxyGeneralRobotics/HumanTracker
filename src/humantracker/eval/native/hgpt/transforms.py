# Derived from GalaxyGeneralRobotics/Humanoid-GPT, de8498f (Apache-2.0).
# Evaluation-only extraction; see ../SOURCES.md for modifications.
import numpy as np
from scipy.spatial.transform import Rotation as R

def WarpPi(ang):
    return np.arctan2(np.sin(ang), np.cos(ang))

def quat2yaw(q_wxyz: np.ndarray) -> np.ndarray:
    """Extract heading yaw via arctan2(R[1,0], R[0,0]).

    Matches the training-time ``transforms_jax.quat2yaw`` exactly.

    Args:
        q_wxyz: (..., 4) quaternions in (w, x, y, z) order.
    Returns:
        yaw: (...,) heading angles in radians, wrapped to [-pi, pi].
    """
    w = q_wxyz[..., 0]
    x = q_wxyz[..., 1]
    y = q_wxyz[..., 2]
    z = q_wxyz[..., 3]
    R00 = 1.0 - 2.0 * (y * y + z * z)
    R10 = 2.0 * (x * y + w * z)
    return WarpPi(np.arctan2(R10, R00))

def wxyz2xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """(…,4) wxyz -> xyzw"""
    return np.roll(q_wxyz, -1, axis=-1)

def xyzw2wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """(…,4) xyzw -> wxyz"""
    return np.roll(q_xyzw, 1, axis=-1)

def quat2mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert batch of quaternions (w, x, y, z) to rotation matrices (3x3).
    Args:
        quat_wxyz: (N, 4) array of quaternions (w, x, y, z)
    Returns:
        rot_mats: (N, 3, 3) array of rotation matrices

    Uses a pure-numpy formula for small batches (N <= 4): scipy's
    ``Rotation.from_quat`` has ~100 us of per-call object/validation overhead
    which dominates for the (1, 4) case used every control step on the robot.
    Falls back to scipy for larger batches where its vectorized code wins.
    """
    q = np.asarray(quat_wxyz)
    if q.ndim == 2 and q.shape[0] <= 4:
        w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
        # Assume already normalized; if not, the rotation matrix is wrong by
        # a uniform scale factor.  Caller (IMU quaternion) guarantees unit norm.
        xx = x * x; yy = y * y; zz = z * z
        xy = x * y; xz = x * z; yz = y * z
        wx = w * x; wy = w * y; wz = w * z
        n = q.shape[0]
        out = np.empty((n, 3, 3), dtype=q.dtype if q.dtype.kind == "f" else np.float32)
        out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
        out[:, 0, 1] = 2.0 * (xy - wz)
        out[:, 0, 2] = 2.0 * (xz + wy)
        out[:, 1, 0] = 2.0 * (xy + wz)
        out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
        out[:, 1, 2] = 2.0 * (yz - wx)
        out[:, 2, 0] = 2.0 * (xz - wy)
        out[:, 2, 1] = 2.0 * (yz + wx)
        out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
        return out
    r = R.from_quat(q, scalar_first=True)
    return r.as_matrix()

def batch_base2navi(base2world: np.ndarray) -> np.ndarray:
    """Batched ``base2navi``.  Equivalent to the scalar version (agrees to
    2e-16 on random rotations): ``z x x_proj`` already discards the vertical
    component, so normalising ``y_axis`` does the horizontal renormalisation.

    That also means the gimbal-lock singularity moves to ``y_axis``: it is
    exactly zero when the base x-axis is vertical.  Falling back to world +y
    reproduces the yaw = 0 frame the scalar version returns there.
    """
    x_proj = base2world[..., :3, 0]
    x_proj = x_proj / np.linalg.norm(x_proj, axis=-1, keepdims=True)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
    valid = ~(norm <= np.sqrt(np.finfo(norm.dtype).tiny))
    y_axis = np.where(
        valid, y_axis / np.where(valid, norm, 1.0), np.array([0.0, 1.0, 0.0])
    )
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, np.broadcast_to(z_axis, x_axis.shape)), axis=-1)
