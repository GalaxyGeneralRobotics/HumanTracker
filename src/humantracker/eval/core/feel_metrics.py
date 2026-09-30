"""GT-free gait / actuation quantities — what a person sees on the robot.

HumanScore, MPJPE and foot-contact-vs-GT score alignment with a ghost. The
numbers here are computed only from the simulated rollout (and the policy
command stream). They are meant to stand in for “did it stay up, did the
motors buzz, did the gait look regular.”

All values are ``inf`` when the history is too short to define the quantity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from humantracker.eval.core.geometry import quat2euler

_INF = float("inf")

FEEL_METRIC_KEYS: List[str] = [
    "root_lat_sway_m",
    "root_bounce_m",
    "root_roll_std",
    "root_pitch_std",
    "heading_rate_abs",
    "sim_duty_l",
    "sim_duty_r",
    "sim_duty_asym",
    "sim_single_support",
    "step_period_std_s",
    "cadence_hz",
    "action_jerk_p95",
    "action_jerk_med",
    "action_hf_ratio",
]


def _empty(action: bool) -> Dict[str, float]:
    out = {k: _INF for k in FEEL_METRIC_KEYS}
    if not action:
        for k in ("action_jerk_p95", "action_jerk_med", "action_hf_ratio"):
            out.pop(k, None)
    return out


def _heading_lateral(xy: np.ndarray) -> np.ndarray:
    """Signed distance to the start→end travel line, in metres."""
    delta = xy[-1] - xy[0]
    length = float(np.linalg.norm(delta))
    if length < 0.30:
        return xy[:, 1] - xy[0, 1]
    direction = delta / length
    rel = xy - xy[0]
    # 2-D cross product with the travel unit vector.
    return rel[:, 0] * direction[1] - rel[:, 1] * direction[0]


def _contact_from_sites(
    state_history: Sequence[Dict],
    foot_site_ids: Sequence[int],
    height_threshold: float,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(sd["site_xpos"], dtype=np.float64)[list(foot_site_ids), 2]
            < height_threshold
            for sd in state_history
        ]
    )


def _step_periods(contact: np.ndarray, ctrl_dt: float) -> np.ndarray:
    """Intervals between successive strikes of the left foot (column 0)."""
    left = contact[:, 0].astype(bool)
    strike = np.flatnonzero((~left[:-1]) & left[1:]) + 1
    if strike.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(strike.astype(np.float64)) * ctrl_dt


def _action_feel(
    action_history: Sequence[np.ndarray], ctrl_dt: float
) -> Dict[str, float]:
    out = {
        "action_jerk_p95": _INF,
        "action_jerk_med": _INF,
        "action_hf_ratio": _INF,
    }
    if len(action_history) < 4:
        return out
    actions = np.stack([np.asarray(a, dtype=np.float64) for a in action_history])
    accel = (actions[2:] - 2.0 * actions[1:-1] + actions[:-2]) / (ctrl_dt ** 2)
    jerk = np.diff(accel, axis=0) / ctrl_dt
    mag = np.mean(np.abs(jerk), axis=1)
    out["action_jerk_p95"] = float(np.percentile(mag, 95))
    out["action_jerk_med"] = float(np.median(mag))

    # Fraction of command energy above 4 Hz (motor buzz), vs the full spectrum.
    # Demean each joint, rFFT, integrate power.
    x = actions - actions.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    power = np.mean(np.abs(spec) ** 2, axis=1)
    freqs = np.fft.rfftfreq(len(actions), d=ctrl_dt)
    total = float(power.sum())
    if total <= 0.0:
        out["action_hf_ratio"] = 0.0
    else:
        out["action_hf_ratio"] = float(power[freqs >= 4.0].sum() / total)
    return out


def compute_feel_metrics(
    state_history: Sequence[Dict],
    ctrl_dt: float = 0.02,
    action_history: Optional[Sequence[np.ndarray]] = None,
    sim_contact: Optional[np.ndarray] = None,
    foot_site_ids: Optional[Sequence[int]] = None,
    height_threshold: float = 0.05,
) -> Dict[str, float]:
    """Robot-state gait regularity and command harshness. No ghost / GT.

    Args:
        state_history: per-step dicts with at least ``qpos``. ``site_xpos`` is
            required when ``sim_contact`` is omitted and ``foot_site_ids`` is
            given.
        ctrl_dt: control timestep (s).
        action_history: policy motor targets, one vector per step.
        sim_contact: optional ``(T, 2)`` boolean contact (left, right). When
            omitted, contact is inferred from foot site height if site ids
            are supplied; otherwise contact metrics stay ``inf``.
        foot_site_ids: left/right foot site indices into ``site_xpos``.
        height_threshold: contact height (m), same default as the GT contact
            helper.

    Raises:
        ValueError: if ``ctrl_dt`` is not positive, or ``sim_contact`` has the
            wrong shape.
        KeyError: if a state entry is missing ``qpos``.
    """
    if ctrl_dt <= 0.0:
        raise ValueError(f"ctrl_dt must be positive, got {ctrl_dt}")

    have_action = action_history is not None
    if len(state_history) < 3:
        return _empty(have_action)

    qpos = np.stack([np.asarray(s["qpos"], dtype=np.float64) for s in state_history])
    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError(f"qpos must be (T, >=7), got {qpos.shape}")

    rpy = quat2euler(qpos[:, 3:7])
    yaw = np.unwrap(rpy[:, 2])
    out = _empty(have_action)
    out["root_lat_sway_m"] = float(np.std(_heading_lateral(qpos[:, :2])))
    out["root_bounce_m"] = float(np.std(qpos[:, 2]))
    out["root_roll_std"] = float(np.std(rpy[:, 0]))
    out["root_pitch_std"] = float(np.std(rpy[:, 1]))
    out["heading_rate_abs"] = float(np.mean(np.abs(np.diff(yaw) / ctrl_dt)))

    contact = sim_contact
    if contact is None and foot_site_ids is not None:
        if any("site_xpos" not in s for s in state_history):
            raise KeyError("site_xpos required when inferring contact from foot sites")
        contact = _contact_from_sites(state_history, foot_site_ids, height_threshold)
    if contact is not None:
        contact = np.asarray(contact)
        if contact.ndim != 2 or contact.shape[1] != 2:
            raise ValueError(f"sim_contact must be (T, 2), got {contact.shape}")
        n = min(len(contact), len(qpos))
        c = contact[:n] > 0.5
        duty = c.mean(axis=0)
        out["sim_duty_l"] = float(duty[0])
        out["sim_duty_r"] = float(duty[1])
        out["sim_duty_asym"] = float(abs(duty[0] - duty[1]))
        out["sim_single_support"] = float(np.mean(c.sum(axis=1) == 1))
        periods = _step_periods(c, ctrl_dt)
        if periods.size >= 2:
            out["step_period_std_s"] = float(np.std(periods))
            out["cadence_hz"] = float(1.0 / np.mean(periods)) if np.mean(periods) > 1e-6 else _INF
        elif periods.size == 1:
            out["cadence_hz"] = float(1.0 / periods[0]) if periods[0] > 1e-6 else _INF

    if have_action:
        out.update(_action_feel(action_history, ctrl_dt))
    return out


def resolve_foot_site_ids(mj_model, names: Sequence[str] = ("left_foot", "right_foot")) -> List[int]:
    """Left/right foot site ids. ``mj_model.site`` raises if a name is missing."""
    return [int(mj_model.site(name).id) for name in names]
