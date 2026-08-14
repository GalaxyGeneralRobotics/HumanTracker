"""Kinematic smoothness & contact consistency metrics for tracking evaluation.

These metrics complement per-frame pose tracking errors (MPJPE / MPJVE / KPT MAE)
by capturing how natural and hardware-friendly the produced motion is. A controller
can attain a low MPJPE while still producing very jittery or impulsive motion that
is unsafe to deploy on hardware; the metrics here surface that behaviour:

* ``joint_acc_mean``  — average joint acceleration (rad / s^2), 2nd-order central
  finite difference of joint angles.
* ``joint_jerk_mean`` — average joint jerk (rad / s^3), 3rd-order finite difference
  of joint angles.
* ``joint_acc_rms`` / ``joint_jerk_rms`` — same metrics in RMS form (more sensitive
  to spikes than the absolute mean).
* ``action_jerk_mean`` — temporal jerk of the policy's joint-target output (rad/s^3),
  measures how oscillatory the *commands* are (independent of physics).
* ``foot_contact_iou`` / ``foot_contact_acc`` — per-frame agreement between the
  simulated foot contact pattern and the reference foot contact pattern (0–1).

All metrics are computed via finite differences on the recorded ``qpos`` (joint
angles) and ``qvel`` (joint velocities) state history at the control rate
``ctrl_dt`` (e.g. 0.02 s for 50 Hz policies).

Error policy: malformed input raises. A metric is reported as ``inf`` only when the
trajectory is genuinely too short to define it — a tracker that terminates after two
steps has no measurable jerk. That is a legitimate benchmark outcome, not an error,
and callers aggregate over finite values only. Missing or inconsistent fields, by
contrast, mean the recorded rollout is wrong and are never papered over.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


_INF = float("inf")


def _stack_joint_qpos(state_history: Sequence[Dict]) -> np.ndarray:
    """Stack joint angles (qpos[7:]) from a state history list, shape (T, n_joints)."""
    return np.stack([np.asarray(s["qpos"][7:], dtype=np.float64) for s in state_history])


def _stack_joint_qvel(state_history: Sequence[Dict]) -> np.ndarray:
    """Stack joint velocities (qvel[6:]) from a state history list, shape (T, n_joints)."""
    return np.stack([np.asarray(s["qvel"][6:], dtype=np.float64) for s in state_history])


def compute_smoothness_metrics(
    state_history: Sequence[Dict],
    ctrl_dt: float = 0.02,
    action_history: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute kinematic smoothness metrics on a single trajectory.

    Smoothness is quantified by the **average joint jerk (rad/s^3)** and
    **acceleration (rad/s^2)** of the joint angles, computed using finite
    differences. Controllers that produce jittery or impulsive motion yield higher
    jerk/acceleration even when MPJPE is low, so these metrics emphasise natural,
    human-like movement and hardware-friendly actuation.

    Args:
        state_history: list of dicts with ``qpos`` and ``qvel`` per simulation step.
        ctrl_dt: control timestep in seconds (e.g. 0.02 for 50 Hz).
        action_history: optional list of motor-target vectors (one per step) for
            computing action smoothness.

    Returns:
        Dict with keys ``joint_acc_mean``, ``joint_jerk_mean``, ``joint_acc_rms``,
        ``joint_jerk_rms``, ``joint_vel_mean`` (auxiliary), and ``action_jerk_mean``
        (only when ``action_history`` is supplied). A value is ``inf`` when the
        history is too short to evaluate that derivative: acceleration needs 3
        frames, jerk 4.

    Raises:
        ValueError: if ``ctrl_dt`` is not positive, or a state entry has a
            ``qpos`` / ``qvel`` length inconsistent with the rest of the history.
        KeyError: if a state entry lacks ``qpos`` or ``qvel``.
    """
    if ctrl_dt <= 0.0:
        raise ValueError(f"ctrl_dt must be positive, got {ctrl_dt}")

    out: Dict[str, float] = {
        "joint_acc_mean": _INF,
        "joint_jerk_mean": _INF,
        "joint_acc_rms": _INF,
        "joint_jerk_rms": _INF,
        "joint_vel_mean": _INF,
    }
    if action_history is not None:
        out["action_jerk_mean"] = _INF

    # Fewer than 3 frames: no second derivative exists. Report inf and stop.
    if len(state_history) < 3:
        return out

    qpos = _stack_joint_qpos(state_history)

    # 2nd-order central finite difference for acceleration:
    #   a[t] = (q[t+1] - 2 q[t] + q[t-1]) / dt^2
    accel = (qpos[2:] - 2.0 * qpos[1:-1] + qpos[:-2]) / (ctrl_dt ** 2)
    out["joint_acc_mean"] = float(np.mean(np.abs(accel)))
    out["joint_acc_rms"] = float(np.sqrt(np.mean(accel ** 2)))

    # 3rd-order finite difference for jerk (forward diff of acceleration):
    #   j[t] = (a[t+1] - a[t]) / dt
    if accel.shape[0] >= 2:
        jerk = np.diff(accel, axis=0) / ctrl_dt
        out["joint_jerk_mean"] = float(np.mean(np.abs(jerk)))
        out["joint_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))

    # Auxiliary: mean joint speed magnitude (helps disambiguate fast-but-smooth
    # motion from slow-but-jittery motion).
    out["joint_vel_mean"] = float(np.mean(np.abs(_stack_joint_qvel(state_history))))

    # Action-space smoothness (independent of physics).
    if action_history is not None and len(action_history) >= 4:
        actions = np.stack([np.asarray(a, dtype=np.float64) for a in action_history])
        a_accel = (actions[2:] - 2.0 * actions[1:-1] + actions[:-2]) / (ctrl_dt ** 2)
        a_jerk = np.diff(a_accel, axis=0) / ctrl_dt
        out["action_jerk_mean"] = float(np.mean(np.abs(a_jerk)))

    return out


def compute_contact_consistency_from_height(
    state_history: Sequence[Dict],
    ref_traj: Dict,
    mj_model,
    foot_site_names: Sequence[str] = ("left_foot", "right_foot"),
    height_threshold: float = 0.05,
) -> Dict[str, float]:
    """Compare simulated vs. reference foot-contact pattern from foot height.

    This is a lightweight proxy for true contact detection: a foot is considered
    to be in contact whenever its foot site height is below ``height_threshold``
    (m). The sites are the sole markers defined by the robot model, so they sit at
    ground level in a standing pose; body frames such as ``*_ankle_roll_link`` sit
    several centimetres higher and are *not* interchangeable with them.

    The reference contact pattern is read from ``ref_traj['foot_contact']``, a
    (T, n_feet) boolean array. Simulated and reference trajectories may differ in
    length when the tracker terminates early; both are compared over their common
    prefix.

    Args:
        state_history: list of dicts containing ``site_xpos`` per simulation step.
        ref_traj: reference trajectory dict containing ``foot_contact`` (T, n_feet).
        mj_model: MuJoCo model, used to resolve ``foot_site_names`` to site ids.
        foot_site_names: names of the foot sites to check.
        height_threshold: height (in metres) below which a foot is "in contact".

    Returns:
        Dict with ``foot_contact_acc`` (per-frame agreement, 0–1) and
        ``foot_contact_iou`` (intersection-over-union of contact frames per foot,
        averaged across feet, 0–1). Both are ``inf`` for an empty history.

    Raises:
        KeyError: if the model has no site of a requested name, if a state entry
            lacks ``site_xpos``, or if ``ref_traj`` has no ``foot_contact``.
        ValueError: if ``foot_contact`` is not a 2-D array with one column per foot.
    """
    if not state_history:
        return {"foot_contact_acc": _INF, "foot_contact_iou": _INF}

    # ``mj_model.site`` raises KeyError for an unknown name, which is the correct
    # outcome: the metric is undefined against a model that has no foot sites.
    foot_ids = [int(mj_model.site(name).id) for name in foot_site_names]

    sim_contact = np.stack([
        np.asarray(sd["site_xpos"])[foot_ids, 2] < height_threshold
        for sd in state_history
    ]).astype(np.float32)

    ref_contact = np.asarray(ref_traj["foot_contact"], dtype=np.float32)
    if ref_contact.ndim != 2 or ref_contact.shape[1] != sim_contact.shape[1]:
        raise ValueError(
            f"ref_traj['foot_contact'] must have shape (T, {sim_contact.shape[1]}), "
            f"got {ref_contact.shape}"
        )

    n = min(len(sim_contact), len(ref_contact))
    sim = sim_contact[:n]
    ref = (ref_contact[:n] > 0.5).astype(np.float32)

    eps = 1e-6
    inter = np.sum(np.minimum(sim, ref), axis=0)
    union = np.sum(np.maximum(sim, ref), axis=0) + eps
    return {
        "foot_contact_acc": float(np.mean(sim == ref)),
        "foot_contact_iou": float(np.mean(inter / union)),
    }


# Keys of all numeric smoothness metrics produced by this module. Useful for
# constructing summary dicts in eval scripts.
SMOOTHNESS_METRIC_KEYS: List[str] = [
    "joint_acc_mean",
    "joint_jerk_mean",
    "joint_acc_rms",
    "joint_jerk_rms",
    "joint_vel_mean",
    "action_jerk_mean",
    "foot_contact_acc",
    "foot_contact_iou",
]
