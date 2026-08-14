"""Feature layout of a reward-model sample, and how to mirror it left-to-right.

The reward model reads a fixed concatenation of named blocks: the retargeted reference the
tracker was asked to follow, then what the simulator actually produced. The two head dicts
below are the schema -- name to width, in order -- and everything downstream derives from
them rather than restating them, because a disagreement about the order or width of these
blocks does not raise, it feeds the model a differently-shaped world and returns a score.

Bilateral flip doubles the preference data by mirroring a trajectory about the sagittal
plane. It cannot be applied to a feature vector as a whole: each block mirrors differently
-- joint angles swap left for right and some invert, a quaternion negates two components,
a planar velocity negates one. So the flip is defined per block, and lives here next to the
layout it depends on.
"""
from __future__ import annotations

import numpy as np

RETARGET_DATA_HEAD = {
    "ref_pose": 7,
    "ref_root_navi_vel": 3,
    "ref_joint_pos": 29,
    "ref_joint_vel": 29,
    "ref_foot_contact": 2,
}


SIM_DATA_HEAD = {
    "sensor_pose": 7,
    "imu_pose": 7,
    "action": 25,
    "motor_target": 29,
    "joint_pos": 29,
    "joint_vel": 29,
    "foot_contact": 2,
    "foot_force": 6,
    "foot_vel": 6,
    "foot_acc": 6,
    "linvel_pelvis": 3,
    "root_navi_vel": 6,
    "acu_root2gv_lin_vel": 3,
    "acu_root2gv_ang_vel": 3,
    "acu_kpt2gv_pose": 14 * 4 * 4,
    "acu_kpt_cvel_in_gv": 14 * 6,
    "next_ref2acu_gv_vel": 3,
    "next_ref2acu_kpt_pose": 14 * 4 * 4,
    "next_ref2acu_kpt_cvel": 14 * 6,
}


DOF_FLIP_SOURCE = np.asarray([
    6, 7, 8, 9, 10, 11,
    0, 1, 2, 3, 4, 5,
    12, 13, 14, 22, 23, 24,
    25, 26, 27, 28, 15, 16,
    17, 18, 19, 20, 21,
], dtype=np.int64)


DOF_FLIP_SIGN = np.asarray([
    1, -1, -1, 1, 1, -1,
    1, -1, -1, 1, 1, -1,
    -1, -1, 1, 1, -1, -1,
    1, -1, 1, -1, 1, -1,
    -1, 1, -1, 1, -1,
], dtype=np.float32)


KPT_FLIP_SOURCE = np.asarray([
    0, 4, 5, 6, 1, 2, 3,
    7, 11, 12, 13, 8, 9, 10,
], dtype=np.int64)


POLAR_VEC3_SIGN = np.asarray([1, -1, 1], dtype=np.float32)


AXIAL_VEC3_SIGN = np.asarray([-1, 1, -1], dtype=np.float32)


PLANAR_VEL_SIGN = np.asarray([1, -1, -1], dtype=np.float32)


POSE7_SIGN = np.asarray([1, -1, 1, 1, -1, 1, -1], dtype=np.float32)


MATRIX4_SIGN = np.asarray([1, -1, 1, 1], dtype=np.float32)


def _flip_dof(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    n_dim = values.shape[-1]
    valid = (DOF_FLIP_SOURCE < n_dim) & (np.arange(len(DOF_FLIP_SOURCE)) < n_dim)
    target = np.arange(len(DOF_FLIP_SOURCE), dtype=np.int64)[valid]
    source = DOF_FLIP_SOURCE[valid]
    out[..., target] = values[..., source] * DOF_FLIP_SIGN[valid]
    return out


def _flip_feet_2(values: np.ndarray) -> np.ndarray:
    return values[..., [1, 0]].copy()


def _flip_feet_vec3(values: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    feet = values.reshape(*original_shape[:-1], 2, 3)
    out = feet[..., [1, 0], :].copy()
    out *= POLAR_VEC3_SIGN
    return out.reshape(original_shape)


def _flip_pose_matrices(values: np.ndarray, num_kpt: int = 14) -> np.ndarray:
    original_shape = values.shape
    mats = values.reshape(*original_shape[:-1], num_kpt, 4, 4)
    out = mats[..., KPT_FLIP_SOURCE, :, :].copy()
    out *= MATRIX4_SIGN.reshape(1, 1, 4, 1)
    out *= MATRIX4_SIGN.reshape(1, 1, 1, 4)
    return out.reshape(original_shape)


def _flip_kpt_cvel(values: np.ndarray, num_kpt: int = 14) -> np.ndarray:
    original_shape = values.shape
    cvel = values.reshape(*original_shape[:-1], num_kpt, 6)
    out = cvel[..., KPT_FLIP_SOURCE, :].copy()
    # MuJoCo cvel order is angular velocity first, linear velocity second.
    out[..., :3] *= AXIAL_VEC3_SIGN
    out[..., 3:] *= POLAR_VEC3_SIGN
    return out.reshape(original_shape)


def _flip_feature(name: str, values: np.ndarray) -> np.ndarray:
    if name in {"ref_pose", "sensor_pose", "imu_pose"}:
        return values * POSE7_SIGN
    if name in {"ref_root_navi_vel", "next_ref2acu_gv_vel"}:
        return values * PLANAR_VEL_SIGN
    if name in {"ref_joint_pos", "ref_joint_vel", "motor_target", "joint_pos", "joint_vel", "action"}:
        return _flip_dof(values)
    if name in {"ref_foot_contact", "foot_contact"}:
        return _flip_feet_2(values)
    if name in {"foot_force", "foot_vel", "foot_acc"}:
        return _flip_feet_vec3(values)
    if name in {"linvel_pelvis", "acu_root2gv_lin_vel"}:
        return values * POLAR_VEC3_SIGN
    if name == "acu_root2gv_ang_vel":
        return values * AXIAL_VEC3_SIGN
    if name == "root_navi_vel":
        out = values.copy()
        out[..., :3] *= AXIAL_VEC3_SIGN
        out[..., 3:] *= POLAR_VEC3_SIGN
        return out
    if name in {"acu_kpt2gv_pose", "next_ref2acu_kpt_pose"}:
        return _flip_pose_matrices(values)
    if name in {"acu_kpt_cvel_in_gv", "next_ref2acu_kpt_cvel"}:
        return _flip_kpt_cvel(values)
    return values.copy()


def flip_features(features: np.ndarray, keys: list[str], heads: dict[str, int]) -> np.ndarray:
    if not keys:
        return features.copy()

    chunks = []
    offset = 0
    for key in keys:
        dim = heads[key]
        chunks.append(_flip_feature(key, features[..., offset:offset + dim]))
        offset += dim
    return np.concatenate(chunks, axis=-1).astype(np.float32)


# The schema, as one mapping and as the ordered key tuple everything else validates against.
# Merged with ** rather than | so this module stays importable on Python 3.8, which the
# sonic, twist2 and gmt evaluation environments run.
FEATURE_HEADS = {**RETARGET_DATA_HEAD, **SIM_DATA_HEAD}
FEATURE_KEYS = tuple(FEATURE_HEADS)
FEATURE_DIM = sum(FEATURE_HEADS.values())
SEQ_LEN = 250

if len(FEATURE_HEADS) != len(RETARGET_DATA_HEAD) + len(SIM_DATA_HEAD):
    raise RuntimeError("a feature key appears in both the reference and simulation blocks")
