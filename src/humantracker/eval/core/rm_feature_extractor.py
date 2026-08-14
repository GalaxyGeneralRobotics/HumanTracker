"""Shared real-MuJoCo feature extraction for RM preference rollouts.

Computes every per-frame field required by
``humantracker/reward_model/features.py``
(``RETARGET_DATA_HEAD`` prompt fields + ``SIM_DATA_HEAD`` trajectory fields)
directly from genuine MuJoCo simulation state.

Design constraints (enforced):
  * No zero-padding, no silent fallback. If a required reference field is
    absent the extractor raises ``KeyError`` so the data gap is surfaced loudly
    instead of being masked by zeros.
  * Rollout-side ("traj") features come from the actual simulated robot, never
    from the reference trajectory.

Every backend shares this one extractor, so a preference pair collected from two
different trackers is guaranteed to carry identically-computed features.  ``foot_acc``
is intentionally *not* produced here: it is a temporal derivative of ``foot_vel`` and
is computed over the full sequence at save time (see
``eval/core/rollout_export.py``), again without padding.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import mujoco

from humantracker.eval.core.geometry import quat_to_mat
from humantracker.eval.core.keypoints import KPT_NAMES, NUM_KPT
from humantracker.reward_model.features import RETARGET_DATA_HEAD, SIM_DATA_HEAD


# 14 tracked keypoint bodies (G1), order must match training data layout.
FEET_SITES = ["left_foot", "right_foot"]

ACTION_DIM = 25

# Reference-field aliases (different retarget caches use different names).
_GV_VEL_KEYS = ("gv_vel", "navi_vel")
_KPT_POSE_KEYS = ("kpt2gv_pose", "kpt_npose")
_KPT_CVEL_KEYS = ("kpt_cvel_in_gv", "kpt_cvel")


def _require(ref: Mapping[str, np.ndarray], keys, ctx: str) -> np.ndarray:
    for key in keys:
        if key in ref:
            return np.asarray(ref[key])
    raise KeyError(
        f"reference frame missing required field (one of {keys}) for {ctx}; "
        f"available={sorted(ref.keys())}"
    )


def _base2navi_single(root_rot_mat: np.ndarray) -> np.ndarray:
    """Gravity-view (navigation) frame rotation from a root rotation matrix."""
    x_proj = root_rot_mat[:3, 0].copy()
    x_proj[2] = 0.0
    norm = np.linalg.norm(x_proj)
    if norm < 1e-9:
        return np.eye(3)
    x_proj /= norm
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis /= (np.linalg.norm(y_axis) + 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def _rot_pos_to_pose(xmat_flat: np.ndarray, xpos: np.ndarray) -> np.ndarray:
    """(N,9) flat rotations + (N,3) positions -> (N,4,4) SE3 poses."""
    n = xpos.shape[0]
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = xmat_flat.reshape(n, 3, 3)
    poses[:, :3, 3] = xpos
    return poses


def _se3_inv(pose: np.ndarray) -> np.ndarray:
    if pose.ndim == 2:
        inv = np.eye(4)
        rot = pose[:3, :3]
        inv[:3, :3] = rot.T
        inv[:3, 3] = -rot.T @ pose[:3, 3]
        return inv
    n = pose.shape[0]
    inv = np.tile(np.eye(4), (n, 1, 1))
    rt = pose[:, :3, :3].transpose(0, 2, 1)
    inv[:, :3, :3] = rt
    inv[:, :3, 3] = -np.einsum('nij,nj->ni', rt, pose[:, :3, 3])
    return inv


def _sensor(mj_model, mj_data, name: str) -> np.ndarray:
    sid = mj_model.sensor(name).id
    adr = mj_model.sensor_adr[sid]
    dim = mj_model.sensor_dim[sid]
    return mj_data.sensordata[adr:adr + dim].copy()


# Cache foot/floor geom id resolution per model (keyed by id(model)).
_FOOT_GEOM_CACHE: Dict[int, tuple] = {}


def _foot_geom_ids(mj_model):
    """Resolve (floor_id, left_foot_ids, right_foot_ids) by geom-name prefix.

    The G1 model has multiple collision geoms per foot
    (e.g. left_foot1_collision, left_foot_box_collision); all of them count as
    that foot for contact detection and contact-force accumulation.
    """
    key = id(mj_model)
    cached = _FOOT_GEOM_CACHE.get(key)
    if cached is not None:
        return cached
    floor_id = mj_model.geom("floor").id
    left_ids, right_ids = set(), set()
    for gid in range(mj_model.ngeom):
        name = mj_model.geom(gid).name or ""
        if name.startswith("left_foot"):
            left_ids.add(gid)
        elif name.startswith("right_foot"):
            right_ids.add(gid)
    if not left_ids or not right_ids:
        raise KeyError(
            "could not resolve foot collision geoms by prefix 'left_foot'/'right_foot'"
        )
    result = (int(floor_id), left_ids, right_ids)
    _FOOT_GEOM_CACHE[key] = result
    return result


def extract_frame_fields(
    mj_model,
    mj_data,
    ref_curr: Mapping[str, np.ndarray],
    ref_next: Mapping[str, np.ndarray],
    action: np.ndarray,
    motor_targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract one frame of RM features as a dict of named 1-D arrays.

    Returned keys (all real, all from ``mj_data`` / sensors / actual body state):
      prompt: ref_pose, ref_root_navi_vel, ref_joint_pos, ref_joint_vel, ref_foot_contact
      traj:   sensor_pose, imu_pose, action, motor_target, joint_pos, joint_vel,
              foot_contact, foot_force, foot_vel, linvel_pelvis, root_navi_vel,
              acu_root2gv_lin_vel, acu_root2gv_ang_vel, acu_kpt2gv_pose,
              acu_kpt_cvel_in_gv, next_ref2acu_gv_vel, next_ref2acu_kpt_pose,
              next_ref2acu_kpt_cvel

    ``foot_acc`` is deliberately omitted (temporal derivative computed at save
    time over the full sequence). Missing reference inputs raise ``KeyError``.
    """
    # Recompute all derived quantities (kinematics, sensors, contacts, cvel)
    # from the current qpos/qvel so the extracted features are real and
    # backend-independent (works whether mj_data came from a CPU step or was
    # synced from an MJX rollout). Idempotent for an already-stepped CPU state.
    mujoco.mj_forward(mj_model, mj_data)

    qpos = np.asarray(mj_data.qpos).copy()
    qvel = np.asarray(mj_data.qvel).copy()

    # ── prompt (reference, 70D) ──
    ref_qpos = np.asarray(_require(ref_curr, ("qpos",), "ref_pose"))[0]
    ref_qvel = np.asarray(_require(ref_curr, ("qvel",), "ref_joint_vel"))[0]
    ref_gv_vel = _require(ref_curr, _GV_VEL_KEYS, "ref_root_navi_vel")[0][:3]
    ref_fc = _require(ref_curr, ("foot_contact",), "ref_foot_contact")[0][:2]

    ref_pose = ref_qpos[:7].astype(np.float32)
    ref_joint_pos = ref_qpos[7:].astype(np.float32)
    ref_joint_vel = ref_qvel[6:].astype(np.float32)
    ref_root_navi_vel = np.asarray(ref_gv_vel, dtype=np.float32)
    ref_foot_contact = np.asarray(ref_fc, dtype=np.float32)

    # ── traj (rollout robot, 780D) ──
    # Training data defines both sensor_pose and imu_pose as the floating-base
    # root pose qpos[:7]; we match that convention exactly (no IMU-site offset).
    sensor_pose = qpos[:7].astype(np.float32)
    imu_pose = qpos[:7].astype(np.float32)

    action = np.asarray(action).reshape(-1)
    if action.shape[0] < ACTION_DIM:
        raise ValueError(f"action has {action.shape[0]} dims, need >= {ACTION_DIM}")
    action_25 = action[:ACTION_DIM].astype(np.float32)

    motor_target = np.asarray(motor_targets).reshape(-1).astype(np.float32)

    joint_pos = qpos[7:].astype(np.float32)
    joint_vel = qvel[6:].astype(np.float32)

    # foot_contact (2) + foot_force (6): real collision / contact forces.
    # The G1 model has no single "left_foot" geom; each foot is made of several
    # collision geoms (left_foot1/2/3_collision, left_foot_box_collision, ...).
    floor_geom_id, left_foot_geom_ids, right_foot_geom_ids = _foot_geom_ids(mj_model)
    left_contact, right_contact = -1.0, -1.0
    foot_force = np.zeros(6, dtype=np.float32)
    force_buf = np.zeros(6)
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == floor_geom_id:
            other = g2
        elif g2 == floor_geom_id:
            other = g1
        else:
            continue
        is_left = other in left_foot_geom_ids
        is_right = other in right_foot_geom_ids
        if not (is_left or is_right):
            continue
        mujoco.mj_contactForce(mj_model, mj_data, i, force_buf)
        if is_left:
            left_contact = 1.0
            foot_force[:3] += force_buf[:3]
        if is_right:
            right_contact = 1.0
            foot_force[3:] += force_buf[:3]
    foot_contact = np.array([left_contact, right_contact], dtype=np.float32)

    # foot_vel (6): real foot-site linear velocity (J @ qvel)
    foot_vel = np.zeros(6, dtype=np.float32)
    for idx, name in enumerate(FEET_SITES):
        sid = mj_model.site(name).id
        jacp = np.zeros((3, mj_model.nv))
        mujoco.mj_jacSite(mj_model, mj_data, jacp, None, sid)
        foot_vel[idx * 3:(idx + 1) * 3] = (jacp @ qvel).astype(np.float32)

    # linvel_pelvis (3): real pelvis velocimeter
    linvel_pelvis = _sensor(mj_model, mj_data, "local_linvel_pelvis").astype(np.float32)
    gyro_pelvis = _sensor(mj_model, mj_data, "gyro_pelvis").astype(np.float32)

    # gravity-view frame
    root2wrd_rot = quat_to_mat(qpos[3:7])
    gv2wrd_pose = np.eye(4)
    gv2wrd_pose[:3, :3] = _base2navi_single(root2wrd_rot)
    gv2wrd_pose[:2, 3] = qpos[:2]
    gv2wrd_pose[2, 3] = 0.0
    root2gv_rot = gv2wrd_pose[:3, :3].T @ root2wrd_rot

    acu_root2gv_lin_vel = (root2gv_rot @ linvel_pelvis).astype(np.float32)
    acu_root2gv_ang_vel = (root2gv_rot @ gyro_pelvis).astype(np.float32)
    root_navi_vel = np.concatenate([acu_root2gv_ang_vel, acu_root2gv_lin_vel]).astype(np.float32)

    # actual keypoint poses in GV frame, from real body xpos/xmat
    body_ids_kpt = np.array([mj_model.body(n).id for n in KPT_NAMES])
    acu_kpt2wrd_pose = _rot_pos_to_pose(
        np.asarray(mj_data.xmat)[body_ids_kpt], np.asarray(mj_data.xpos)[body_ids_kpt]
    )
    acu_kpt2gv_pose = _se3_inv(gv2wrd_pose) @ acu_kpt2wrd_pose  # (14,4,4)

    # actual keypoint spatial velocities in GV frame, from real mj_data.cvel
    r_wrd2gv = gv2wrd_pose[:3, :3].T
    acu_kpt_cvel_in_wrd = np.asarray(mj_data.cvel)[body_ids_kpt].astype(np.float64)  # (14,6)
    acu_kpt_cvel_in_gv = np.zeros_like(acu_kpt_cvel_in_wrd)
    acu_kpt_cvel_in_gv[:, :3] = (r_wrd2gv @ acu_kpt_cvel_in_wrd[:, :3].T).T
    acu_kpt_cvel_in_gv[:, 3:] = (r_wrd2gv @ acu_kpt_cvel_in_wrd[:, 3:].T).T

    # residuals: next reference relative to actual
    gv_next = _require(ref_next, _GV_VEL_KEYS, "next_ref2acu_gv_vel")[0]
    next_ref2acu_gv_vel = np.array([
        gv_next[0] - acu_root2gv_lin_vel[0],
        gv_next[1] - acu_root2gv_lin_vel[1],
        gv_next[2] - acu_root2gv_ang_vel[2],
    ], dtype=np.float32)

    acu_gv2kpt_pose = _se3_inv(acu_kpt2gv_pose)
    next_ref_kpt2gv = _require(ref_next, _KPT_POSE_KEYS, "next_ref2acu_kpt_pose")[0]
    next_ref2acu_kpt_pose = np.einsum('nij,njk->nik', acu_gv2kpt_pose, next_ref_kpt2gv)

    next_ref_kpt_cvel = _require(ref_next, _KPT_CVEL_KEYS, "next_ref2acu_kpt_cvel")[0]
    next_ref2acu_kpt_cvel = (next_ref_kpt_cvel - acu_kpt_cvel_in_gv).astype(np.float32)

    return {
        # prompt
        "ref_pose": ref_pose,
        "ref_root_navi_vel": ref_root_navi_vel,
        "ref_joint_pos": ref_joint_pos,
        "ref_joint_vel": ref_joint_vel,
        "ref_foot_contact": ref_foot_contact,
        # traj
        "sensor_pose": sensor_pose,
        "imu_pose": imu_pose,
        "action": action_25,
        "motor_target": motor_target,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "foot_contact": foot_contact,
        "foot_force": foot_force,
        "foot_vel": foot_vel,
        "linvel_pelvis": linvel_pelvis,
        "root_navi_vel": root_navi_vel,
        "acu_root2gv_lin_vel": acu_root2gv_lin_vel,
        "acu_root2gv_ang_vel": acu_root2gv_ang_vel,
        "acu_kpt2gv_pose": acu_kpt2gv_pose.flatten().astype(np.float32),
        "acu_kpt_cvel_in_gv": acu_kpt_cvel_in_gv.flatten().astype(np.float32),
        "next_ref2acu_gv_vel": next_ref2acu_gv_vel,
        "next_ref2acu_kpt_pose": next_ref2acu_kpt_pose.flatten().astype(np.float32),
        "next_ref2acu_kpt_cvel": next_ref2acu_kpt_cvel.flatten().astype(np.float32),
    }


def sequence_features_from_history(feature_history, fps: int = 50):
    """Convert canonical per-frame field dicts to RM prompt/traj sequences.

    This is the same feature source written to rollout parquet/npz by
    ``eval.rollout_export``.  Using it for eval-time RM scoring keeps training,
    cached rollouts, and benchmark scoring on one feature schema.
    """
    if not feature_history:
        return [], []
    fields = {}
    for key in (*RETARGET_DATA_HEAD, *(k for k in SIM_DATA_HEAD if k != "foot_acc")):
        fields[key] = np.stack([
            np.asarray(frame[key], dtype=np.float32).reshape(-1)
            for frame in feature_history
        ], axis=0)
    dt = 1.0 / float(fps)
    fields["foot_acc"] = np.gradient(fields["foot_vel"], dt, axis=0).astype(np.float32)
    prompt_seq = np.concatenate(
        [fields[k] for k in RETARGET_DATA_HEAD], axis=-1
    ).astype(np.float32)
    traj_seq = np.concatenate(
        [fields[k] for k in SIM_DATA_HEAD], axis=-1
    ).astype(np.float32)
    return list(prompt_seq), list(traj_seq)
