"""The tracked keypoint set, in the one order everything downstream assumes.

This ordering is a contract, not a convenience. ``WHOLE_BODY_HEIGHT_INDICES`` in
:mod:`humantracker.eval.core.termination_metrics` indexes into it positionally, and
the reward model's feature vector is laid out per keypoint in this sequence, so a
reordering silently invalidates every trained checkpoint. It lives in its own
module so that there is exactly one copy to reorder.
"""

from __future__ import annotations

__all__ = ["KPT_NAMES", "NUM_KPT"]

KPT_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
NUM_KPT = len(KPT_NAMES)
