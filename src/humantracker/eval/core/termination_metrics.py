"""Shared termination metrics for tracker evaluation.

Both rules end a rollout on the same thresholds — a height error above 0.25 m,
a rotation error above 1 rad, or a non-finite state — and differ only in which
keypoints they inspect:

* ``trunk`` checks the pelvis and torso.
* ``whole_body`` checks the pelvis and all four limb ends: pelvis, both ankles
  and both wrists for height, with the pelvis as the sole rotation anchor.
  These are SONIC's published evaluation terms (https://arxiv.org/abs/2511.07820),
  which is what the paper reports, applied identically to every tracker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from humantracker.eval.core.geometry import batch_base2navi, quat2mat
from humantracker.eval.core.keypoints import KPT_NAMES


TERMINATION_METRICS = ("trunk", "whole_body")
WHOLE_BODY_HEIGHT_INDICES = np.array((0, 3, 6, 10, 13))


def _navigation_rotation(pelvis_rotation: np.ndarray) -> np.ndarray:
    """Gravity-aligned frame from the pelvis rotation, one call per frame (not batched
    over keypoints -- ``pelvis_rotation`` is already a single ``(3, 3)`` matrix indexed
    ``[:, 0]`` for its x-axis column).

    Degenerate when the pelvis local x-axis (forward) points exactly along world z --
    a handstand or a flip's inverted apex -- where projecting onto the horizontal plane
    gives a zero vector with no defined heading. Falls back to identity, matching
    ``core/geometry.py``'s ``batch_base2navi`` and
    ``core/rm_feature_extractor.py``'s ``_base2navi_single`` on this exact singularity.
    """
    x_axis = pelvis_rotation[:, 0].copy()
    x_axis[2] = 0.0
    norm = np.linalg.norm(x_axis)
    if norm < 1e-9:
        return np.eye(3)
    x_axis /= norm
    z_axis = np.array((0.0, 0.0, 1.0))
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def _error_poses(
    state: Mapping[str, np.ndarray],
    reference_poses: np.ndarray,
    mj_model,
) -> np.ndarray:
    qpos = state["qpos"]
    pelvis_quaternion_xyzw = np.r_[qpos[4:7], qpos[3]]
    pelvis_rotation = Rotation.from_quat(pelvis_quaternion_xyzw).as_matrix()
    navigation_pose = np.eye(4)
    navigation_pose[:3, :3] = _navigation_rotation(pelvis_rotation)
    navigation_pose[:2, 3] = qpos[:2]

    body_ids = np.array([mj_model.body(name).id for name in KPT_NAMES])
    body_poses = np.repeat(np.eye(4)[None], len(KPT_NAMES), axis=0)
    body_poses[:, :3, :3] = state["xmat"][body_ids].reshape(-1, 3, 3)
    body_poses[:, :3, 3] = state["xpos"][body_ids]
    body_to_navigation = np.linalg.inv(navigation_pose) @ body_poses
    return np.linalg.inv(body_to_navigation) @ reference_poses


def _whole_body_terminated(error_poses: np.ndarray, state: Mapping[str, np.ndarray]) -> bool:
    if not np.isfinite(state["qpos"]).all() or not np.isfinite(state["qvel"]).all():
        return True
    height_error = np.abs(error_poses[WHOLE_BODY_HEIGHT_INDICES, 2, 3])
    anchor_angle = Rotation.from_matrix(error_poses[0, :3, :3]).magnitude()
    return bool((height_error > 0.25).any() or anchor_angle**2 > 1.0)


def calculate_whole_body_trajectory_length(
    state_history: Sequence[Mapping[str, np.ndarray]],
    ref_traj: Mapping[str, np.ndarray],
    mj_model,
    *,
    ref_start_index: int = 0,
) -> tuple[float, int]:
    """Height error at pelvis, ankles and wrists; rotation error at the pelvis.

    These are SONIC's published evaluation terms (https://arxiv.org/abs/2511.07820).
    Unlike :func:`calculate_trunk_trajectory_length` this requires a
    ``state_history`` exactly as long as the reference.
    """
    if not state_history:
        return 0.0, 0
    reference_poses = ref_traj["kpt2gv_pose"]
    total_frames = len(reference_poses) - ref_start_index
    if len(state_history) != total_frames:
        raise ValueError(
            f"State/reference length mismatch: {len(state_history)} != {total_frames}"
        )

    alignment = _error_poses(
        state_history[0], reference_poses[ref_start_index], mj_model
    )
    inverse_alignment = np.linalg.inv(alignment)
    for index, state in enumerate(state_history):
        errors = (
            _error_poses(state, reference_poses[index + ref_start_index], mj_model)
            @ inverse_alignment
        )
        if _whole_body_terminated(errors, state):
            step = index + 1
            return step / total_frames, step
    return 1.0, total_frames
# ─── trunk rule ──────────────────────────────────────────────────────────────
# The two rules reach the navigation frame by different but algebraically
# equivalent routes: ``_navigation_rotation`` zeroes the pelvis x-axis
# z-component then normalises, while ``batch_base2navi`` normalises first and
# lets ``z x x_proj`` cancel that component. They agree to floating-point
# rounding. Both are kept so that neither metric's published numbers shift.


def _trunk_error_poses(
    state: Mapping[str, np.ndarray],
    reference_pose: np.ndarray,
    mj_model,
    alignment_transform: np.ndarray,
) -> np.ndarray:
    qpos = state["qpos"]
    pelvis2world_rot = quat2mat(qpos[3:7][None])
    body_ids_kpt = np.array([mj_model.body(name).id for name in KPT_NAMES])
    kpt2wrd_rot = state["xmat"][body_ids_kpt].reshape(-1, 3, 3)
    kpt2wrd_pos = state["xpos"][body_ids_kpt][None]
    navi2world_rot = batch_base2navi(pelvis2world_rot)
    navi2world_pose = np.full((1, 4, 4), np.eye(4))
    navi2world_pose[0, :3, :3] = navi2world_rot[0]
    navi2world_pose[0, :2, 3] = qpos[:2]
    kpt2wrd_pose = np.full((1, len(KPT_NAMES), 4, 4), np.eye(4))
    kpt2wrd_pose[0, :, :3, :3] = kpt2wrd_rot
    kpt2wrd_pose[0, :, :3, 3] = kpt2wrd_pos[0]
    kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
    navi2kpt_pose = np.linalg.inv(kpt2navi_pose)
    return navi2kpt_pose @ reference_pose @ np.linalg.inv(alignment_transform)


def _trunk_terminated(error_poses: np.ndarray, state: Mapping[str, np.ndarray]) -> bool:
    """Pelvis and torso only, against the shared 0.25 m / 1 rad thresholds."""
    kpt_id_peltor = [0, 7]
    err_rot = error_poses[0, kpt_id_peltor, :3, :3]
    rotvec = Rotation.from_matrix(err_rot.reshape(-1, 3, 3)).as_rotvec().reshape(
        len(kpt_id_peltor), 3)
    theta = np.linalg.norm(rotvec, axis=1)
    err_height = np.abs(error_poses[0, kpt_id_peltor, 2, 3])
    done_nan = np.isnan(state["qpos"]).any() or np.isnan(state["qvel"]).any()
    return bool((err_height > 0.25).any() or (theta > 1).any() or done_nan)


def calculate_trunk_trajectory_length(
    state_history: Sequence[Mapping[str, np.ndarray]],
    ref_traj: Mapping[str, np.ndarray],
    mj_model,
    *,
    ref_start_index: int = 0,
) -> tuple[float, int]:
    """Trunk termination terms: pelvis and torso height and rotation error.

    Unlike :func:`calculate_whole_body_trajectory_length` this tolerates a
    ``state_history`` shorter than the reference, which is what a rollout that
    stopped early produces.
    """
    if not state_history:
        return 0.0, 0
    reference_poses = ref_traj["kpt2gv_pose"]
    total_frames = len(ref_traj["qpos"]) - ref_start_index
    termination_step = len(state_history)

    first = state_history[0]
    qpos = first["qpos"]
    pelvis2world_rot = quat2mat(qpos[3:7][None])
    body_ids_kpt = np.array([mj_model.body(name).id for name in KPT_NAMES])
    kpt2wrd_rot = first["xmat"][body_ids_kpt].reshape(-1, 3, 3)
    kpt2wrd_pos = first["xpos"][body_ids_kpt][None]
    navi2world_rot = batch_base2navi(pelvis2world_rot)
    navi2world_pose = np.eye(4)
    navi2world_pose[:2, 3] = qpos[:2]
    navi2world_pose[:3, :3] = navi2world_rot[0]
    kpt2wrd_pose = np.full((len(KPT_NAMES), 4, 4), np.eye(4))
    kpt2wrd_pose[:, :3, :3] = kpt2wrd_rot
    kpt2wrd_pose[:, :3, 3] = kpt2wrd_pos[0]
    kpt2navi_pose = np.linalg.inv(navi2world_pose) @ kpt2wrd_pose
    navi2kpt_pose = np.linalg.inv(kpt2navi_pose)
    alignment_transform = navi2kpt_pose @ reference_poses[ref_start_index]

    for index, state in enumerate(state_history):
        reference_pose = reference_poses[index + ref_start_index][np.newaxis]
        errors = _trunk_error_poses(state, reference_pose, mj_model, alignment_transform)
        if _trunk_terminated(errors, state):
            termination_step = index + 1
            break
    return min(termination_step / total_frames, 1.0), termination_step


def calculate_trajectory_length(
    state_history: Sequence[Mapping[str, np.ndarray]],
    ref_traj: Mapping[str, np.ndarray],
    mj_model,
    metric: str = "trunk",
    *,
    ref_start_index: int = 0,
) -> tuple[float, int]:
    """Tracked-length ratio and termination step under the named rule."""
    if metric == "whole_body":
        return calculate_whole_body_trajectory_length(
            state_history, ref_traj, mj_model, ref_start_index=ref_start_index)
    if metric == "trunk":
        return calculate_trunk_trajectory_length(
            state_history, ref_traj, mj_model, ref_start_index=ref_start_index)
    raise ValueError(f"Unknown termination metric: {metric}")
