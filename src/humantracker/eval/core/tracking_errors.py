"""Per-frame tracking errors, shared by every tracker backend.

These are the benchmark's headline numbers -- MPJPE, MPJVE and the root errors -- so one
implementation has to produce all of them for a comparison between trackers to mean anything.
Each backend used to carry its own copy, and the Humanoid-GPT backend took its numbers from
that checkout's ``tracking.metrics`` instead: the same arithmetic apart from the
quaternion-to-matrix helper, which disagreed with this one by ~9e-16 on a unit quaternion.

Errors are measured in the navigation frame -- the pelvis frame with roll and pitch removed --
so a tracker is not charged for the reference motion's absolute heading.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from humantracker.eval.core.geometry import (
    batch_base2navi,
    quat2euler,
    quat2mat,
    wrap_to_pi,
)
from humantracker.eval.core.keypoints import KPT_NAMES

__all__ = [
    "calculate_kpt_mae_error",
    "calculate_joint_tracking_error",
    "calculate_root_tracking_error",
]


def calculate_kpt_mae_error(state, ref_curr, ref_next, mj_model):
    """Mean absolute keypoint position (m) and rotation (rad) error against the reference."""
    qpos = state.mj_data.qpos
    pelvis2world_rot = quat2mat(qpos[3:7][None])
    body_ids_kpt = np.array([mj_model.body(n).id for n in KPT_NAMES])
    kpt2wrd_rot = state.mj_data.xmat[body_ids_kpt].reshape(-1, 3, 3)
    kpt2wrd_pos = state.mj_data.xpos[body_ids_kpt][None]
    navi2world_rot = batch_base2navi(pelvis2world_rot)
    navi2world_pose = np.eye(4)
    navi2world_pose[:2, 3] = qpos[:2]
    navi2world_pose[:3, :3] = navi2world_rot[0]
    kpt2wrd_pose = np.full((len(KPT_NAMES), 4, 4), np.eye(4))
    kpt2wrd_pose[:, :3, :3] = kpt2wrd_rot
    kpt2wrd_pose[:, :3, 3] = kpt2wrd_pos[0]
    kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
    navi2kpt_pose = np.linalg.inv(kpt2navi_pose)
    curr_ref2kpt_pose = navi2kpt_pose @ ref_curr["kpt2gv_pose"][0]
    pos_error = np.abs(curr_ref2kpt_pose[:, :3, 3])
    pos_mae = np.mean(pos_error)
    rot_error = curr_ref2kpt_pose[:, :3, :3]
    rotvec = R.from_matrix(rot_error.reshape(-1, 3, 3)).as_rotvec().reshape(len(KPT_NAMES), 3)
    rot_mae = np.mean(np.linalg.norm(rotvec, axis=1))
    return pos_mae, rot_mae


def calculate_joint_tracking_error(state, ref_curr):
    """Mean absolute joint position (rad) and velocity (rad/s) error, excluding the root."""
    curr_qpos = state.mj_data.qpos[7:]
    curr_qvel = state.mj_data.qvel[6:]
    ref_qpos = ref_curr["qpos"][0, 7:]
    ref_qvel = ref_curr["qvel"][0, 6:]
    return float(np.mean(np.abs(curr_qpos - ref_qpos))), float(np.mean(np.abs(curr_qvel - ref_qvel)))


def calculate_root_tracking_error(state, ref_curr):
    """Root position (mm), velocity (mm/s) and yaw (rad) error against the reference."""
    curr_root_pos = state.mj_data.qpos[:3]
    curr_root_vel = state.mj_data.qvel[:3]
    cur_root_yaw = quat2euler(state.mj_data.qpos[3:7])[2]
    ref_root_pos = ref_curr["qpos"][0, :3]
    ref_root_vel = ref_curr["qvel"][0, :3]
    ref_root_yaw = quat2euler(ref_curr["qpos"][0, 3:7])[2]
    pos_error_mm = np.mean(np.abs(curr_root_pos - ref_root_pos)) * 1000.0
    vel_error_mms = np.mean(np.abs(curr_root_vel - ref_root_vel)) * 1000.0
    yaw_error = np.mean(np.abs(wrap_to_pi(cur_root_yaw - ref_root_yaw)))
    return pos_error_mm, vel_error_mms, yaw_error
