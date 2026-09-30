# Derived from GalaxyGeneralRobotics/Humanoid-GPT, de8498f (Apache-2.0).
# Evaluation-only extraction; see ../SOURCES.md for modifications.
from collections.abc import Iterable
from typing import Literal
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R, Slerp
from .transforms import batch_base2navi, quat2mat, wxyz2xyzw, xyzw2wxyz
from .constants import KPT_NAMES, ACTION_JOINT_NAMES

def resample_state(
    qpos: np.ndarray,
    # qvel: np.ndarray,
    freq_src: float,
    freq_tgt: float,
) -> np.ndarray:
    """Resample (qpos, qvel) from freq_src -> freq_tgt with SLERP on quats (wxyz)."""
    n_src = qpos.shape[0]
    t_end = (n_src - 1) / float(freq_src)
    t_src = np.linspace(0.0, t_end, n_src, endpoint=True)
    n_tgt = int(np.floor(t_end * float(freq_tgt))) + 1
    t_tgt = np.linspace(0.0, t_end, n_tgt, endpoint=True)

    # xyz & joints: linear
    xyz = np.vstack([np.interp(t_tgt, t_src, qpos[:, i]) for i in range(3)]).T
    joints = np.vstack(
        [np.interp(t_tgt, t_src, qpos[:, 7 + i]) for i in range(qpos.shape[1] - 7)]
    ).T

    # quat (wxyz) -> SciPy (xyzw) -> SLERP -> back (wxyz)
    q_xyzw = wxyz2xyzw(qpos[:, 3:7])
    rot_src = R.from_quat(q_xyzw)
    slerp = Slerp(t_src, rot_src)
    q_xyzw_tgt = slerp(t_tgt).as_quat()
    q_wxyz_tgt = xyzw2wxyz(q_xyzw_tgt)

    # # velocities: linear
    # linv = np.vstack([np.interp(t_tgt, t_src, qvel[:, i]) for i in range(3)]).T
    # angv = np.vstack([np.interp(t_tgt, t_src, qvel[:, 3 + i]) for i in range(3)]).T
    # jvel = np.vstack(
    #     [np.interp(t_tgt, t_src, qvel[:, 6 + i]) for i in range(qvel.shape[1] - 6)]
    # ).T
    return np.hstack([xyz, q_wxyz_tgt, joints])

def recompute_qvel(
    qpos_wxyz: np.ndarray,
    frequency: float | Iterable[float],
    frame: Literal["world", "body"] = "world",
) -> np.ndarray:
    """
    Recompute generalized velocities from positions.

    Args:
        qpos_wxyz: (num_steps, 7+J) array.
            Free joint [x y z qw qx qy qz ...] followed by J joint positions.
        frequency: sampling rate(s) in Hz.
            - scalar: uniform rate
            - array (num_steps,): per-sample rate (first T-1 entries used)
            - array (T-1,): per-interval rate
        frame: 'body' (default, angular velocity in body frame)
               or 'world' (angular velocity expressed in world frame)

    Returns:
        qvel: (num_steps, 6+J) with qvel[0] estimated via forward difference
    """
    qpos_wxyz = np.asarray(qpos_wxyz)
    if qpos_wxyz.ndim != 2 or qpos_wxyz.shape[1] < 7:
        raise ValueError("qpos must be (num_steps, 7+J) with free joint at front (wxyz).")
    num_steps, N = qpos_wxyz.shape

    qvel = np.zeros((num_steps, N - 1), dtype=qpos_wxyz.dtype)
    if num_steps < 2:
        return qvel

    # ---- frequency handling
    freq = np.asarray(frequency, dtype=float)
    if freq.ndim == 0:
        rate = np.full(num_steps - 1, float(freq), dtype=float)
    elif freq.shape == (num_steps - 1,):
        rate = freq
    elif freq.shape == (num_steps,):
        rate = freq[:-1]
    else:
        raise ValueError(f"`frequency` must be scalar, shape (T-1,), or shape (num_steps,). Got shape {freq.shape}.")
    if not np.all(np.isfinite(rate)) or np.any(rate <= 0.0):
        raise ValueError("`frequency` must be positive and finite.")

    # ---- linear velocity (world frame)
    qvel[1:, 0:3] = np.diff(qpos_wxyz[:, 0:3], axis=0) * rate[:, None]

    # ---- angular velocity
    q_xyzw = wxyz2xyzw(qpos_wxyz[:, 3:7])  # convert to [x y z w]
    # normalize
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)

    r = R.from_quat(q_xyzw)
    rel = r[:-1].inv() * r[1:]
    omega_body = rel.as_rotvec() * rate[:, None]

    if frame == "body":
        qvel[1:, 3:6] = omega_body
    elif frame == "world":
        qvel[1:, 3:6] = r[:-1].apply(omega_body)
    else:
        raise ValueError("frame must be 'body' or 'world'.")

    # ---- joint velocities (continuous angles, no wrapping needed)
    if N > 7:
        qvel[1:, 6:] = np.diff(qpos_wxyz[:, 7:], axis=0) * rate[:, None]

    # Use forward difference for the first frame
    qvel[0] = qvel[1]

    return qvel

def mj_body_pose(mj_model, mj_data, name: str) -> np.ndarray:
    bid = mj_model.body(name).id
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = mj_data.xpos[bid]
    m[:3, :3] = mj_data.xmat[bid].reshape(3, 3)
    return m

def extract_kpt(
    mj_model: mujoco.MjModel,
    qpos_src: np.ndarray,
    qvel_src: np.ndarray,
    key_body_names: list[str],
    qpos_jnt_ids: np.ndarray = None,
    qvel_jnt_ids: np.ndarray = None,
    fps: float = 50,
) -> dict[str, np.ndarray]:
    """Roll the model with provided low-frequency states and export kpts/vels in a navigation frame."""
    dt = 1 / fps
    num_jnt = 29  # matches your data layout
    num_steps = int(qpos_src.shape[0])
    mj_data = mujoco.MjData(mj_model)

    kpt_body_ids = np.int32([mj_model.body(b).id for b in key_body_names])

    # fill robot qpos/qvel
    qpos_rb = np.zeros((num_steps, mj_model.nq), dtype=np.float32)
    qvel_rb = np.zeros((num_steps, mj_model.nv), dtype=np.float32)
    qpos_rb[:, :7] = qpos_src[:, :7]
    if qpos_jnt_ids is not None:
        qpos_rb[:, qpos_jnt_ids] = qpos_src[:, 7:]
    else:
        qpos_rb[:, 7:] = qpos_src[:, 7:]

    qvel_rb[:, :6] = qvel_src[:, :6]
    if qvel_jnt_ids is not None:
        qvel_rb[:, qvel_jnt_ids] = qvel_src[:, 6:]
    else:
        qvel_rb[:, 6:] = qvel_src[:, 6:]

    # navigation frame: yaw-aligned, origin on ground; rot from base quat
    base2wrd_rot = quat2mat(qpos_src[:, 3:7])
    gvi2wrd_rot = batch_base2navi(base2wrd_rot)
    gvi2wrd_pose = np.tile(np.eye(4, dtype=np.float32), (num_steps, 1, 1))
    gvi2wrd_pose[:, :2, 3] = qpos_src[:, :2]
    gvi2wrd_pose[:, :3, :3] = gvi2wrd_rot

    out = {
        "qpos": np.zeros((num_steps, 7 + num_jnt), dtype=np.float32),
        "qvel": np.zeros((num_steps, 6 + num_jnt), dtype=np.float32),
        "kpt2gv_pose": np.zeros((num_steps, len(key_body_names), 4, 4), dtype=np.float32),
        "kpt_cvel_in_gv": np.zeros((num_steps, len(key_body_names), 6), dtype=np.float32),
        "gv_vel": np.zeros((num_steps, 3), dtype=np.float32),
        "gv2wrd_pose": gvi2wrd_pose.copy(),
        "foot_contact": np.zeros((num_steps, 2), dtype=np.float32),
    }
    kpt2wrd_pos = np.zeros((num_steps, len(key_body_names), 3), dtype=np.float32)
    kpt_cvel_in_wrd = np.zeros((num_steps, len(key_body_names), 6), dtype=np.float32)

    for t in range(num_steps):
        mj_data.qpos[:] = qpos_rb[t]
        mj_data.qvel[:] = qvel_rb[t]
        mujoco.mj_forward(mj_model, mj_data)

        kpt_world = np.float32([mj_body_pose(mj_model, mj_data, n) for n in key_body_names])  # (K,4,4)
        kpt_gv = np.linalg.inv(gvi2wrd_pose[t]) @ kpt_world  # (K,4,4)
        kpt2wrd_pos[t] = np.float32(kpt_world[:, :3, 3]).copy()
        kpt_cvel_in_wrd[t] = np.float32(mj_data.cvel[kpt_body_ids]).copy()  # (K,6)

        out["kpt2gv_pose"][t] = kpt_gv
        out["qpos"][t] = mj_data.qpos.copy()
        out["qvel"][t] = mj_data.qvel.copy()

    for t in range(1, num_steps):
        nav2wrd_last = gvi2wrd_pose[t - 1]
        nav2wrd_curr = gvi2wrd_pose[t]
        curr2last = np.linalg.inv(nav2wrd_last) @ nav2wrd_curr
        vel_lin = curr2last[:2, 3] / dt
        vel_yaw = R.from_matrix(curr2last[:3, :3]).as_euler("xyz")[2] / dt
        out["gv_vel"][t] = np.hstack([vel_lin, [vel_yaw]])

    kpt_cvel_in_gv = np.zeros((num_steps, len(KPT_NAMES), 6))
    R_wrd2gv = np.swapaxes(gvi2wrd_pose[..., :3, :3], -1, -2)  # R_gv2wrd^T = R_wrd2gv
    kpt_cvel_in_gv[..., :3] = np.einsum("...ij,...kj->...ki", R_wrd2gv, kpt_cvel_in_wrd[..., :3])
    kpt_cvel_in_gv[..., 3:] = np.einsum("...ij,...kj->...ki", R_wrd2gv, kpt_cvel_in_wrd[..., 3:])
    out["kpt_cvel_in_gv"] = kpt_cvel_in_gv

    return out

def qpos2kpt(mj_model, qpos_src, freq_src, freq_tgt):
    """The evaluator's conversion: no smoothing, clipping or contact estimation."""
    qpos = resample_state(qpos_src, freq_src, freq_tgt)
    qvel = recompute_qvel(qpos, freq_tgt)
    mj_model.opt.timestep = 1 / float(freq_tgt)
    # Preserve the upstream reduced-joint layouts.
    excluded = {34: {13, 14}, 30: {13, 14, 20, 21, 27, 28},
                28: {13, 14, 19, 20, 21, 26, 27, 28}}.get(qpos.shape[1])
    if excluded:
        included = np.setdiff1d(np.arange(29), list(excluded))
        full_qpos = np.zeros((len(qpos), 36))
        full_qvel = np.zeros((len(qvel), 35))
        full_qpos[:, np.hstack([np.arange(7), 7 + included])] = qpos
        full_qvel[:, np.hstack([np.arange(6), 6 + included])] = qvel
        qpos, qvel = full_qpos, full_qvel
    qpos_ids = np.hstack([mj_model.joint(n).qposadr for n in ACTION_JOINT_NAMES])
    dof_ids = np.hstack([mj_model.joint(n).dofadr for n in ACTION_JOINT_NAMES])
    return extract_kpt(mj_model, qpos, qvel, KPT_NAMES, qpos_ids, dof_ids, freq_tgt)
