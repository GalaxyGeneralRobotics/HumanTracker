"""Reference-side corruption models for online-tracking and VLA command streams.

Only the *policy's* reference is corrupted. Metrics, HumanScore features and the
red GT ghost always read the clean motion, so a run measures how well a tracker
survives a degraded command, not how well it matches a degraded target.

The corruption is a staged pipeline rather than one additive noise term, because
the artifacts that actually break real-robot tracking are not zero-mean jitter:

1. ``timing``   causal transport delay, zero-order hold at the sensor/VLA rate,
                and dropped refreshes. A held sample is stale, not noisy.
2. ``bias``     per-clip constant offset plus a slow random walk. Models
                calibration error, body-size mismatch, retarget offset and yaw
                drift. A persistent 2 deg wrist offset is a different failure
                mode from 2 deg of jitter: the policy must hold a wrong pose.
3. ``jitter``   fast zero-mean wobble, but *correlated* across joints through a
                global factor and per-limb factors. Independent per-DoF noise is
                sensor noise; tracker error is structured (depth-scale error
                scales every limb, torso attitude error rotates everything
                downstream) and does not average out over a future window.
4. ``chunk``    piecewise-constant per-chunk offset. A VLA emits an open-loop
                action chunk at a low rate, so its error is smooth inside a
                chunk and *discontinuous at the boundary*.
5. ``velocity`` how the command stream's velocity is produced.

Stage 5 matters more than it looks. Differentiating a corrupted pose at the
control rate amplifies the noise by ~1/(tau*dt): a 3 deg OU wobble with
tau=150 ms at dt=20 ms becomes ~1.2 rad/s of joint-velocity error, larger than
the clean signal itself. No real stack does that -- position and velocity come
out of the same filter -- so ``VelocityMode.FILTERED`` differentiates a causally
low-passed signal, and ``VelocityMode.CLEAN`` leaves velocity untouched for
cross-tracker comparisons.

That last mode exists because the backends do not consume the same reference
fields (see ``REF_FIELDS_CONSUMED`` in each backend module): SONIC reads
reference joint velocity and Humanoid-GPT reads reference keypoint spatial
velocity, while a pose-only policy reads neither. Any velocity corruption
therefore reaches some trackers and not others.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from humantracker.eval.core.geometry import batch_base2navi, quat2mat, wxyz_to_xyzw
from humantracker.eval.core.keypoints import KPT_NAMES

CTRL_DT = 0.02

# The five endpoints a sparse tracker consumes (see ``hgpt_sparse``). Real online
# tracking and most VLA retargets are anchored here (pelvis/IMU, wrists, ankles); everything
# else is filled in and is the part that fights a 14-point tracker.
ENDPOINT_BODY_NAMES = frozenset(
    {
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    }
)

# Joints a tracker localises comparatively well (VR pucks, markers, or an IK
# solve anchored on hands and feet). Scaled by ``endpoint_ratio``.
ENDPOINT_JOINT_SUBSTR = ("wrist", "ankle")

# Limb groups for the correlated-jitter factor model. Matched by substring
# against the MuJoCo joint names, so a joint that belongs to no group simply
# carries global + independent terms.
LIMB_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("left_leg", ("left_hip", "left_knee", "left_ankle")),
    ("right_leg", ("right_hip", "right_knee", "right_ankle")),
    ("waist", ("waist",)),
    ("left_arm", ("left_shoulder", "left_elbow", "left_wrist")),
    ("right_arm", ("right_shoulder", "right_elbow", "right_wrist")),
)

# 14-link tree used to warp a robot FK cloud into human bone ratios
# (ScaleBridge: Xsens segments * 0.75, no IK). The bone *into* each child is
# scaled by a per-clip factor drawn from that child's kind.
BODY_PARENT: Dict[str, Optional[str]] = {
    "pelvis": None,
    "left_hip_roll_link": "pelvis",
    "left_knee_link": "left_hip_roll_link",
    "left_ankle_roll_link": "left_knee_link",
    "right_hip_roll_link": "pelvis",
    "right_knee_link": "right_hip_roll_link",
    "right_ankle_roll_link": "right_knee_link",
    "torso_link": "pelvis",
    "left_shoulder_roll_link": "torso_link",
    "left_elbow_link": "left_shoulder_roll_link",
    "left_wrist_yaw_link": "left_elbow_link",
    "right_shoulder_roll_link": "torso_link",
    "right_elbow_link": "right_shoulder_roll_link",
    "right_wrist_yaw_link": "right_elbow_link",
}

BONE_KIND: Dict[str, str] = {
    "left_hip_roll_link": "thigh",
    "left_knee_link": "thigh",
    "left_ankle_roll_link": "shank",
    "right_hip_roll_link": "thigh",
    "right_knee_link": "thigh",
    "right_ankle_roll_link": "shank",
    "torso_link": "torso",
    "left_shoulder_roll_link": "torso",
    "left_elbow_link": "upper_arm",
    "left_wrist_yaw_link": "forearm",
    "right_shoulder_roll_link": "torso",
    "right_elbow_link": "upper_arm",
    "right_wrist_yaw_link": "forearm",
}

# Time-constant EMA used by Humanoid-GPT deploy (``deploy/retarget.py``).
GMR_EMA_TAU_MS = 38.6


class VelocityMode:
    #: Reference velocity of the delayed/held pose, uncorrupted. Use when
    #: comparing trackers that consume different reference fields.
    CLEAN = "clean"
    #: Backward difference of a causally low-passed noisy pose. What a real
    #: online stack reports.
    FILTERED = "filtered"
    #: Backward difference of the raw noisy pose. Amplifies jitter by ~1/(tau*dt);
    #: kept only to reproduce earlier runs.
    RAW_DIFF = "raw_diff"

    ALL = (CLEAN, FILTERED, RAW_DIFF)


def trajectory_seed(file_name: str, base_seed: int) -> int:
    """Stable per-clip seed, so every tracker and mode sees one realization."""
    digest = hashlib.sha256(f"{int(base_seed)}:{file_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def after_fk_seed(qpos_seed: int) -> int:
    """Separate stream from qpos corruption, shared across every backend."""
    return int(qpos_seed) + 1


@dataclass(frozen=True)
class RefNoiseProfile:
    """One corruption regime. ``scale`` multiplies every magnitude, not the timing.

    Magnitudes at ``scale=1`` are meant to be a plausible mid-range online
    setup, so a sweep brackets reality instead of starting from it.
    """

    name: str

    # ── stage 1: timing ────────────────────────────────────────────────
    latency_ms: float = 50.0
    hold_hz: float = 30.0
    dropout_prob: float = 0.01

    # ── stage 2: bias / drift ─────────────────────────────────────────
    bias_joint_rad: float = float(np.deg2rad(2.0))
    bias_root_pos_m: float = 0.015
    bias_root_yaw_rad: float = float(np.deg2rad(3.0))
    bias_root_rp_rad: float = float(np.deg2rad(1.0))
    drift_tau_s: float = 8.0
    #: Split of stage-2 energy between a per-clip constant and the slow walk.
    bias_const_frac: float = 0.6

    # ── stage 3: correlated jitter ────────────────────────────────────
    jitter_joint_rad: float = float(np.deg2rad(2.0))
    jitter_root_pos_m: float = 0.010
    jitter_root_yaw_rad: float = float(np.deg2rad(1.5))
    jitter_root_rp_rad: float = float(np.deg2rad(1.0))
    jitter_tau_s: float = 0.15
    #: Factor weights, normalised so each joint keeps its marginal sigma.
    w_global: float = 0.4
    w_limb: float = 0.6
    #: Wrists/ankles relative to intermediate joints.
    endpoint_ratio: float = 0.5

    # ── stage 4: VLA chunking ─────────────────────────────────────────
    chunk_hz: float = 0.0
    chunk_joint_rad: float = 0.0
    chunk_root_pos_m: float = 0.0

    # ── stage 5: velocity ─────────────────────────────────────────────
    vel_mode: str = VelocityMode.FILTERED
    vel_filter_hz: float = 6.0

    # ── stage 6: after-FK mid-link residual ───────────────────────────
    # Applied in the heading/gravity-view frame, independently per body, so
    # bone lengths break. Endpoints stay close to the delayed skeleton;
    # intermediate links (elbow/knee/shoulder/hip/torso) carry the retarget
    # / vision fill-in error that a 5-point policy never sees.
    kpt_end_pos_m: float = 0.0
    kpt_mid_pos_m: float = 0.0
    kpt_end_ori_rad: float = 0.0
    kpt_mid_ori_rad: float = 0.0
    kpt_bias_frac: float = 0.75
    kpt_jitter_tau_s: float = 0.20
    kpt_chunk_hz: float = 0.0
    kpt_chunk_end_pos_m: float = 0.0
    kpt_chunk_mid_pos_m: float = 0.0
    kpt_chunk_end_ori_rad: float = 0.0
    kpt_chunk_mid_ori_rad: float = 0.0
    #: Extra mid-joint residual on the stored ``qpos`` only, modelling the
    #: null space of a 5-anchor IK. Not applied to FK-derived body targets.
    #: On ``xsens_human`` this is the side-channel kill: HGPT must not keep a
    #: clean G1 ``qpos`` while its keypoints have been warped off the robot.
    nullspace_joint_rad: float = 0.0

    # ── morphology warp (ScaleBridge / xsens_human) ───────────────────
    # Per-clip multiplicative bone-length scales, drawn once and applied
    # along the 14-link tree from the pelvis. ``scale`` multiplies the
    # *spread* around 1; ``scale=0`` leaves every factor at 1.
    morph_arm_spread: float = 0.0
    morph_leg_spread: float = 0.0
    morph_torso_spread: float = 0.0

    # Same-magnitude heading-frame jitter on every link (mocap segment noise).
    body_jitter_pos_m: float = 0.0
    body_jitter_ori_rad: float = 0.0

    # ── GMR deploy EMA (Humanoid-GPT LiveRefConverter) ────────────────
    #: Time constant in ms. Not scaled: it is a property of the filter.
    #: Gated by ``scale>0`` so ``scale=0`` stays a no-op.
    ema_tau_ms: float = 0.0

    # ── occlusion spikes ──────────────────────────────────────────────
    spike_prob: float = 0.003
    spike_scale: float = 3.0
    spike_max_frames: int = 2

    dt: float = CTRL_DT
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.vel_mode not in VelocityMode.ALL:
            raise ValueError(f"vel_mode must be one of {VelocityMode.ALL}")
        if self.scale < 0:
            raise ValueError(f"scale must be non-negative, got {self.scale}")
        if not 0.0 <= self.bias_const_frac <= 1.0:
            raise ValueError("bias_const_frac must lie in [0, 1]")
        if self.w_global**2 + self.w_limb**2 > 1.0:
            raise ValueError("w_global^2 + w_limb^2 must not exceed 1")
        if not 0.0 <= self.kpt_bias_frac <= 1.0:
            raise ValueError("kpt_bias_frac must lie in [0, 1]")

    @property
    def w_indep(self) -> float:
        """Independent weight that keeps the per-joint marginal sigma at 1."""
        return float(np.sqrt(max(1.0 - self.w_global**2 - self.w_limb**2, 0.0)))

    @property
    def has_after_fk(self) -> bool:
        """True when intermediate links get an extra residual after FK."""
        if self.scale <= 0:
            return False
        return any(
            v > 0.0
            for v in (
                self.kpt_end_pos_m,
                self.kpt_mid_pos_m,
                self.kpt_end_ori_rad,
                self.kpt_mid_ori_rad,
                self.kpt_chunk_end_pos_m,
                self.kpt_chunk_mid_pos_m,
                self.kpt_chunk_end_ori_rad,
                self.kpt_chunk_mid_ori_rad,
            )
        )

    @property
    def has_morphology(self) -> bool:
        """True when the 14-link cloud is re-proportioned off the G1 skeleton."""
        if self.scale <= 0:
            return False
        return any(
            v > 0.0
            for v in (
                self.morph_arm_spread,
                self.morph_leg_spread,
                self.morph_torso_spread,
            )
        )

    @property
    def has_body_jitter(self) -> bool:
        if self.scale <= 0:
            return False
        return self.body_jitter_pos_m > 0.0 or self.body_jitter_ori_rad > 0.0

    @property
    def has_body_warp(self) -> bool:
        """Any after-FK edit of the 14 world-frame bodies (not of qpos)."""
        return self.has_after_fk or self.has_morphology or self.has_body_jitter

    def describe(self) -> Dict:
        """Machine-readable record of what was injected, for result JSONs."""
        s = self.scale
        return {
            "profile": self.name,
            "scale": s,
            "timing": {
                "latency_ms": self.latency_ms * s,
                "hold_hz": self.hold_hz if s > 0 else 0.0,
                "dropout_prob": min(self.dropout_prob * s, 1.0),
            },
            "bias": {
                "joint_deg": float(np.rad2deg(self.bias_joint_rad * s)),
                "root_pos_m": self.bias_root_pos_m * s,
                "root_yaw_deg": float(np.rad2deg(self.bias_root_yaw_rad * s)),
                "drift_tau_s": self.drift_tau_s,
                "const_frac": self.bias_const_frac,
            },
            "jitter": {
                "joint_deg": float(np.rad2deg(self.jitter_joint_rad * s)),
                "root_pos_m": self.jitter_root_pos_m * s,
                "root_yaw_deg": float(np.rad2deg(self.jitter_root_yaw_rad * s)),
                "tau_s": self.jitter_tau_s,
                "w_global": self.w_global,
                "w_limb": self.w_limb,
                "w_indep": self.w_indep,
                "endpoint_ratio": self.endpoint_ratio,
            },
            "chunk": {
                "hz": self.chunk_hz,
                "joint_deg": float(np.rad2deg(self.chunk_joint_rad * s)),
                "root_pos_m": self.chunk_root_pos_m * s,
            },
            "velocity": {"mode": self.vel_mode, "filter_hz": self.vel_filter_hz},
            "spikes": {"prob": self.spike_prob, "scale": self.spike_scale},
            "after_fk": {
                "enabled": self.has_after_fk,
                "end_pos_cm": 100.0 * self.kpt_end_pos_m * s,
                "mid_pos_cm": 100.0 * self.kpt_mid_pos_m * s,
                "end_ori_deg": float(np.rad2deg(self.kpt_end_ori_rad * s)),
                "mid_ori_deg": float(np.rad2deg(self.kpt_mid_ori_rad * s)),
                "bias_frac": self.kpt_bias_frac,
                "chunk_hz": self.kpt_chunk_hz,
                "chunk_mid_pos_cm": 100.0 * self.kpt_chunk_mid_pos_m * s,
                "chunk_mid_ori_deg": float(np.rad2deg(self.kpt_chunk_mid_ori_rad * s)),
            },
            "morphology": {
                "enabled": self.has_morphology,
                "arm_spread": self.morph_arm_spread * s,
                "leg_spread": self.morph_leg_spread * s,
                "torso_spread": self.morph_torso_spread * s,
            },
            "body_jitter": {
                "enabled": self.has_body_jitter,
                "pos_cm": 100.0 * self.body_jitter_pos_m * s,
                "ori_deg": float(np.rad2deg(self.body_jitter_ori_rad * s)),
            },
            "ema_tau_ms": self.ema_tau_ms if s > 0 else 0.0,
            "nullspace_joint_deg": float(np.rad2deg(self.nullspace_joint_rad * s)),
        }


#: Unsmoothed streaming pose estimation / IK: delayed, held at camera rate,
#: biased by calibration, and wobbling in a body-correlated way.
ONLINE_TRACKING = RefNoiseProfile(name="online_tracking")

#: A VLA emitting open-loop action chunks: no sensor-rate hold, error dominated
#: by per-chunk offsets that jump at the boundary rather than by per-frame
#: wobble. Latency is only mildly above the tracking profile because a chunk is
#: a prediction: inference lag is partly paid back by targeting ahead.
VLA_CHUNK = RefNoiseProfile(
    name="vla_chunk",
    latency_ms=60.0,
    hold_hz=0.0,
    dropout_prob=0.0,
    jitter_joint_rad=float(np.deg2rad(0.8)),
    jitter_root_pos_m=0.004,
    jitter_root_yaw_rad=float(np.deg2rad(0.5)),
    jitter_root_rp_rad=float(np.deg2rad(0.4)),
    chunk_hz=10.0,
    chunk_joint_rad=float(np.deg2rad(2.5)),
    chunk_root_pos_m=0.015,
    spike_prob=0.0,
)

#: What the real stack actually looks like for this project's trackers: a
#: delayed, kinematically consistent *anchor* (pelvis + wrists + ankles) plus
#: an after-FK fill-in on the other nine links. A five-point tracker never
#: sees those nine; dense 14-point trackers are forced to track them.
SPARSE_ANCHOR = RefNoiseProfile(
    name="sparse_anchor",
    # Timing stays -- latency is real -- but qpos additives stay off so the
    # five anchors are only stale, not independently jittered.
    bias_joint_rad=0.0,
    bias_root_pos_m=0.0,
    bias_root_yaw_rad=0.0,
    bias_root_rp_rad=0.0,
    jitter_joint_rad=0.0,
    jitter_root_pos_m=0.0,
    jitter_root_yaw_rad=0.0,
    jitter_root_rp_rad=0.0,
    spike_prob=0.0,
    kpt_end_pos_m=0.008,
    kpt_mid_pos_m=0.050,
    kpt_end_ori_rad=float(np.deg2rad(3.0)),
    kpt_mid_ori_rad=float(np.deg2rad(15.0)),
    nullspace_joint_rad=float(np.deg2rad(5.0)),
)

#: VLA / retarget fill-in: the same mid-link conflict, but the extra error is
#: piecewise-constant and jumps at the action-chunk boundary.
VLA_FILL = RefNoiseProfile(
    name="vla_fill",
    latency_ms=60.0,
    hold_hz=0.0,
    dropout_prob=0.0,
    bias_joint_rad=0.0,
    bias_root_pos_m=0.0,
    bias_root_yaw_rad=0.0,
    bias_root_rp_rad=0.0,
    jitter_joint_rad=0.0,
    jitter_root_pos_m=0.0,
    jitter_root_yaw_rad=0.0,
    jitter_root_rp_rad=0.0,
    spike_prob=0.0,
    kpt_end_pos_m=0.006,
    kpt_mid_pos_m=0.025,
    kpt_end_ori_rad=float(np.deg2rad(2.0)),
    kpt_mid_ori_rad=float(np.deg2rad(8.0)),
    kpt_chunk_hz=10.0,
    kpt_chunk_mid_pos_m=0.040,
    kpt_chunk_mid_ori_rad=float(np.deg2rad(12.0)),
    nullspace_joint_rad=float(np.deg2rad(5.0)),
)

#: ScaleBridge live input: Xsens 14 human segments, uniformly scaled (0.75)
#: toward the robot, then streamed as Cartesian bodies with no IK. Bone
#: *ratios* still do not match G1, so the 14-point cloud is not a legal
#: robot pose; the 5-point mask simply never sees the conflicting links.
#: All 14 links are treated equally -- this is not the mid-link fill-in.
XSENS_HUMAN = RefNoiseProfile(
    name="xsens_human",
    latency_ms=20.0,
    hold_hz=90.0,
    dropout_prob=0.0,
    bias_joint_rad=0.0,
    bias_root_pos_m=0.0,
    bias_root_yaw_rad=0.0,
    bias_root_rp_rad=0.0,
    jitter_joint_rad=0.0,
    jitter_root_pos_m=0.0,
    jitter_root_yaw_rad=0.0,
    jitter_root_rp_rad=0.0,
    spike_prob=0.0,
    morph_arm_spread=0.18,
    morph_leg_spread=0.12,
    morph_torso_spread=0.08,
    body_jitter_pos_m=0.008,
    body_jitter_ori_rad=float(np.deg2rad(3.0)),
    nullspace_joint_rad=float(np.deg2rad(5.0)),
    vel_mode=VelocityMode.CLEAN,
)

#: Humanoid-GPT live input: mocap → GMR multi-target IK → time-constant EMA
#: → playout delay. The skeleton stays kinematically consistent; the residual
#: is qpos-level and tighter on wrists/ankles (GMR position tasks) than on
#: hips (rotation-only, low weight). Velocity is rebuilt from adjacent qpos
#: the way ``LiveRefConverter`` does.
GMR_STREAM = RefNoiseProfile(
    name="gmr_stream",
    latency_ms=65.0,
    hold_hz=0.0,
    dropout_prob=0.0,
    ema_tau_ms=GMR_EMA_TAU_MS,
    bias_joint_rad=float(np.deg2rad(1.5)),
    bias_root_pos_m=0.008,
    bias_root_yaw_rad=float(np.deg2rad(2.0)),
    bias_root_rp_rad=float(np.deg2rad(0.5)),
    jitter_joint_rad=float(np.deg2rad(0.8)),
    jitter_root_pos_m=0.003,
    jitter_root_yaw_rad=float(np.deg2rad(0.6)),
    jitter_root_rp_rad=float(np.deg2rad(0.3)),
    endpoint_ratio=0.40,
    spike_prob=0.0,
    vel_mode=VelocityMode.FILTERED,
)

PROFILES: Dict[str, RefNoiseProfile] = {
    ONLINE_TRACKING.name: ONLINE_TRACKING,
    VLA_CHUNK.name: VLA_CHUNK,
    SPARSE_ANCHOR.name: SPARSE_ANCHOR,
    VLA_FILL.name: VLA_FILL,
    XSENS_HUMAN.name: XSENS_HUMAN,
    GMR_STREAM.name: GMR_STREAM,
}


def build_profile(
    name: str,
    scale: float = 1.0,
    vel_mode: Optional[str] = None,
) -> RefNoiseProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown ref-noise profile {name!r}; have {sorted(PROFILES)}")
    profile = PROFILES[name]
    overrides: Dict = {"scale": float(scale)}
    if vel_mode is not None:
        overrides["vel_mode"] = vel_mode
    return replace(profile, **overrides)


# ═══════════════════════════════════════════════════════════════════════════
#   STOCHASTIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def _ou(
    rng: np.random.Generator,
    n_frames: int,
    shape: Tuple[int, ...],
    tau_s: float,
    dt: float,
) -> np.ndarray:
    """Unit-variance stationary Ornstein-Uhlenbeck, causal.

    ``x[t] = a x[t-1] + sqrt(1-a^2) eps``, ``a = exp(-dt/tau)``, so the marginal
    variance is 1 at every ``t`` and callers scale it themselves.
    """
    a = float(np.exp(-dt / tau_s))
    innov = float(np.sqrt(max(1.0 - a * a, 0.0)))
    out = np.empty((n_frames, *shape), dtype=np.float64)
    out[0] = rng.standard_normal(shape)
    for t in range(1, n_frames):
        out[t] = a * out[t - 1] + innov * rng.standard_normal(shape)
    return out


def _rotvec_left_apply(quats_wxyz: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    """Left-multiply world-frame rotation noise onto ``(..., 4)`` wxyz quats.

    The result is put back in the input's hemisphere. Canonicalising to ``w>=0``
    instead would flip the sign of any clip whose reference quaternion starts
    negative, which is the same rotation but a large numerical change in a field
    the policies read directly -- corruption the noise model never asked for.
    """
    shape = quats_wxyz.shape[:-1]
    flat = quats_wxyz.reshape(-1, 4)
    base = R.from_quat(wxyz_to_xyzw(flat))
    noise = R.from_rotvec(rotvecs.reshape(-1, 3))
    xyzw = (noise * base).as_quat()
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)
    wxyz[np.sum(wxyz * flat, axis=-1) < 0] *= -1.0
    return wxyz.reshape(*shape, 4)


def _yaw_rp_rotvecs(
    yaw: np.ndarray, roll_pitch: np.ndarray
) -> np.ndarray:
    """Assemble ``(T, 3)`` world rotation vectors from separate yaw and r/p noise.

    Split because a real base estimate gets gravity from an IMU: roll and pitch
    stay small while yaw is free to drift.
    """
    out = np.empty((len(yaw), 3), dtype=np.float64)
    out[:, :2] = roll_pitch
    out[:, 2] = yaw
    return out


def _lerp_qpos(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Interpolate two qpos rows; ``[3:7]`` is a wxyz quaternion.

    Matches Humanoid-GPT ``deploy/playout.py`` so the GMR-stream EMA is the
    same filter the robot actually runs, not a close cousin.
    """
    out = a * (1.0 - w) + b * w
    q0, q1 = a[3:7], b[3:7]
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = q0 * (1.0 - w) + q1 * w
    n = float(np.linalg.norm(q))
    out[3:7] = q / n if n > 1e-8 else a[3:7]
    return out


def bodies_world_to_kpt2gv(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    gv2w: np.ndarray,
) -> np.ndarray:
    """World ``(T, B, 3)`` / ``(T, B, 4)`` wxyz bodies → ``(T, B, 4, 4)`` gv poses."""
    n_frames, n_bodies = body_pos.shape[:2]
    kpt2w = np.zeros((n_frames, n_bodies, 4, 4), dtype=np.float64)
    kpt2w[:] = np.eye(4)
    kpt2w[:, :, :3, 3] = body_pos
    kpt2w[:, :, :3, :3] = (
        R.from_quat(wxyz_to_xyzw(np.asarray(body_quat, dtype=np.float64).reshape(-1, 4)))
        .as_matrix()
        .reshape(n_frames, n_bodies, 3, 3)
    )
    return np.linalg.inv(gv2w)[:, None] @ kpt2w


# ═══════════════════════════════════════════════════════════════════════════
#   MODEL
# ═══════════════════════════════════════════════════════════════════════════

class RefNoiseModel:
    """Applies a :class:`RefNoiseProfile` to a reference trajectory."""

    def __init__(self, profile: RefNoiseProfile):
        self.profile = profile

    # ── stage 1 ───────────────────────────────────────────────────────
    def source_indices(
        self, n_frames: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Frame each output step actually saw: latency, then hold, then dropout.

        ``scale`` multiplies the latency and the dropout rate and gates the hold,
        so ``scale=0`` is exactly the clean reference. The hold *rate* itself is a
        property of the sensor, not a severity knob, so it is not scaled.
        """
        p = self.profile
        if p.scale <= 0:
            return np.arange(n_frames, dtype=np.int64)
        lag = int(round(p.latency_ms * p.scale / 1000.0 / p.dt))
        step = 1 if p.hold_hz <= 0 else max(1, int(round(1.0 / (p.hold_hz * p.dt))))
        drop = min(p.dropout_prob * p.scale, 1.0)
        src = np.empty(n_frames, dtype=np.int64)
        latched = 0
        for t in range(n_frames):
            if t % step == 0 and (drop <= 0.0 or rng.random() >= drop):
                latched = min(max(t - lag, 0), n_frames - 1)
            src[t] = latched
        return src

    # ── stages 2-4 ────────────────────────────────────────────────────
    def _joint_sigma(self, joint_names: Sequence[str], base: float) -> np.ndarray:
        is_end = np.array(
            [any(tag in name for tag in ENDPOINT_JOINT_SUBSTR) for name in joint_names]
        )
        ratio = np.where(is_end, self.profile.endpoint_ratio, 1.0)
        return base * self.profile.scale * ratio

    def _limb_index(self, joint_names: Sequence[str]) -> np.ndarray:
        """Group id per joint; ``-1`` for joints outside every limb group."""
        out = np.full(len(joint_names), -1, dtype=np.int64)
        for gid, (_, tags) in enumerate(LIMB_GROUPS):
            for j, name in enumerate(joint_names):
                if any(tag in name for tag in tags):
                    out[j] = gid
        return out

    def _correlated_jitter(
        self,
        rng: np.random.Generator,
        n_frames: int,
        joint_names: Sequence[str],
    ) -> np.ndarray:
        """``(T, n_joints)`` jitter with a global factor plus per-limb factors."""
        p = self.profile
        n_joints = len(joint_names)
        n_groups = len(LIMB_GROUPS)
        z_global = _ou(rng, n_frames, (1,), p.jitter_tau_s, p.dt)
        z_limb = _ou(rng, n_frames, (n_groups,), p.jitter_tau_s, p.dt)
        z_indep = _ou(rng, n_frames, (n_joints,), p.jitter_tau_s, p.dt)

        limb_of = self._limb_index(joint_names)
        limb_term = np.zeros((n_frames, n_joints), dtype=np.float64)
        known = limb_of >= 0
        limb_term[:, known] = z_limb[:, limb_of[known]]

        # Reassign a grouped joint's limb weight to its independent term when it
        # has no group, so the marginal variance stays 1 either way.
        w_limb = np.where(known, p.w_limb, 0.0)
        w_indep = np.sqrt(
            np.maximum(1.0 - p.w_global**2 - w_limb**2, 0.0)
        )
        unit = p.w_global * z_global + w_limb[None, :] * limb_term + w_indep[None, :] * z_indep
        sigma = self._joint_sigma(joint_names, p.jitter_joint_rad)
        return unit * sigma[None, :]

    def _slow_bias(
        self,
        rng: np.random.Generator,
        n_frames: int,
        shape: Tuple[int, ...],
        sigma: np.ndarray | float,
    ) -> np.ndarray:
        """Per-clip constant plus a slow random walk, sharing the given sigma."""
        p = self.profile
        const_w = float(np.sqrt(p.bias_const_frac))
        walk_w = float(np.sqrt(1.0 - p.bias_const_frac))
        const = rng.standard_normal(shape)
        walk = _ou(rng, n_frames, shape, p.drift_tau_s, p.dt)
        unit = const_w * const[None, ...] + walk_w * walk
        return unit * np.asarray(sigma)

    def _chunk_offsets(
        self,
        rng: np.random.Generator,
        n_frames: int,
        shape: Tuple[int, ...],
        sigma: np.ndarray | float,
        hz: Optional[float] = None,
    ) -> np.ndarray:
        """Piecewise-constant per-chunk offset; the boundary jump is the point."""
        rate = self.profile.chunk_hz if hz is None else hz
        if rate <= 0:
            return np.zeros((n_frames, *shape), dtype=np.float64)
        length = max(1, int(round(1.0 / (rate * self.profile.dt))))
        n_chunks = int(np.ceil(n_frames / length))
        per_chunk = rng.standard_normal((n_chunks, *shape)) * np.asarray(sigma)
        return np.repeat(per_chunk, length, axis=0)[:n_frames]

    def _spikes(
        self,
        rng: np.random.Generator,
        series: np.ndarray,
        sigma: np.ndarray | float,
    ) -> None:
        """Rare short kicks, added in place on a ``(T, n)`` series."""
        p = self.profile
        if p.spike_prob <= 0:
            return
        n_frames, n = series.shape
        hits = rng.random((n_frames, n)) < p.spike_prob
        if not np.any(hits):
            return
        sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,))
        for t, i in zip(*np.nonzero(hits)):
            dur = int(rng.integers(1, p.spike_max_frames + 1))
            series[t : t + dur, i] += p.spike_scale * sigma[i] * rng.standard_normal()

    # ── stage 6: after-FK, heading-frame, independent per body ────────
    def _body_sigma(
        self, body_names: Sequence[str], end: float, mid: float
    ) -> np.ndarray:
        return np.array(
            [
                end if name in ENDPOINT_BODY_NAMES else mid
                for name in body_names
            ],
            dtype=np.float64,
        ) * self.profile.scale

    def body_residuals(
        self,
        n_frames: int,
        body_names: Sequence[str],
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Independent heading-frame ``(T, B, 3)`` position and rotvec residuals.

        Independence is the point: a shared factor would keep bone lengths, and
        then 14-point would still be a strictly more informative observation.
        """
        p = self.profile
        n_bodies = len(body_names)
        rng = np.random.default_rng(int(seed))
        pos_s = self._body_sigma(body_names, p.kpt_end_pos_m, p.kpt_mid_pos_m)
        ori_s = self._body_sigma(body_names, p.kpt_end_ori_rad, p.kpt_mid_ori_rad)
        chunk_pos = self._body_sigma(
            body_names, p.kpt_chunk_end_pos_m, p.kpt_chunk_mid_pos_m
        )
        chunk_ori = self._body_sigma(
            body_names, p.kpt_chunk_end_ori_rad, p.kpt_chunk_mid_ori_rad
        )
        slow_w = float(np.sqrt(p.kpt_bias_frac))
        fast_w = float(np.sqrt(1.0 - p.kpt_bias_frac))

        def residual(sigma: np.ndarray, chunk_s: np.ndarray) -> np.ndarray:
            slow = self._slow_bias(rng, n_frames, (n_bodies, 3), sigma[:, None])
            fast = _ou(rng, n_frames, (n_bodies, 3), p.kpt_jitter_tau_s, p.dt)
            out = slow_w * slow + fast_w * fast * sigma[:, None]
            if p.kpt_chunk_hz > 0:
                out = out + self._chunk_offsets(
                    rng, n_frames, (n_bodies, 3), chunk_s[:, None], hz=p.kpt_chunk_hz
                )
            return out

        return residual(pos_s, chunk_pos), residual(ori_s, chunk_ori)

    def _nullspace_joint_residual(
        self,
        n_frames: int,
        joint_names: Sequence[str],
        seed: int,
    ) -> np.ndarray:
        """``(T, n_joints)`` mid-joint residual; wrists/ankles stay at zero."""
        p = self.profile
        if p.nullspace_joint_rad <= 0 or p.scale <= 0:
            return np.zeros((n_frames, len(joint_names)), dtype=np.float64)
        is_end = np.array(
            [any(tag in name for tag in ENDPOINT_JOINT_SUBSTR) for name in joint_names]
        )
        sigma = np.where(is_end, 0.0, p.nullspace_joint_rad * p.scale)
        rng = np.random.default_rng(int(seed))
        return self._slow_bias(rng, n_frames, (len(joint_names),), sigma)

    def clip_bone_scales(self, seed: int) -> Dict[str, float]:
        """Per-clip multiplicative factors for ``upper_arm`` / ``forearm`` / …"""
        p = self.profile
        rng = np.random.default_rng(int(seed))
        s = p.scale

        def draw(spread: float) -> float:
            return 1.0 + s * spread * float(rng.uniform(-1.0, 1.0))

        return {
            "upper_arm": draw(p.morph_arm_spread),
            "forearm": draw(p.morph_arm_spread),
            "thigh": draw(p.morph_leg_spread),
            "shank": draw(p.morph_leg_spread),
            "torso": draw(p.morph_torso_spread),
        }

    def apply_morphology(
        self,
        body_pos: np.ndarray,
        body_names: Sequence[str],
        seed: int,
    ) -> np.ndarray:
        """Scale bone vectors from the pelvis; factors are constant across time."""
        if not self.profile.has_morphology:
            return body_pos
        pos = np.asarray(body_pos, dtype=np.float64)
        out = pos.copy()
        index = {n: i for i, n in enumerate(body_names)}
        factors = self.clip_bone_scales(seed)
        # Parents before children. Names not in the 14-link tree stay put.
        pending: List[str] = [n for n in body_names if n in BODY_PARENT]
        placed = {n for n, parent in BODY_PARENT.items() if parent is None}
        while pending:
            progress = False
            remain: List[str] = []
            for name in pending:
                parent = BODY_PARENT[name]
                if parent is None or parent not in index:
                    placed.add(name)
                    progress = True
                    continue
                if parent not in placed:
                    remain.append(name)
                    continue
                if name not in index:
                    placed.add(name)
                    progress = True
                    continue
                pi, ci = index[parent], index[name]
                kind = BONE_KIND.get(name)
                scale = factors[kind] if kind in factors else 1.0
                out[:, ci] = out[:, pi] + scale * (pos[:, ci] - pos[:, pi])
                placed.add(name)
                progress = True
            if not progress:
                break
            pending = remain
        return out

    def uniform_body_residuals(
        self, n_frames: int, n_bodies: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Same-magnitude heading-frame residual on every body."""
        p = self.profile
        rng = np.random.default_rng(int(seed))
        pos_s = np.full(n_bodies, p.body_jitter_pos_m * p.scale)
        ori_s = np.full(n_bodies, p.body_jitter_ori_rad * p.scale)
        slow_w = float(np.sqrt(p.kpt_bias_frac))
        fast_w = float(np.sqrt(1.0 - p.kpt_bias_frac))

        def residual(sigma: np.ndarray) -> np.ndarray:
            slow = self._slow_bias(rng, n_frames, (n_bodies, 3), sigma[:, None])
            fast = _ou(rng, n_frames, (n_bodies, 3), p.kpt_jitter_tau_s, p.dt)
            return slow_w * slow + fast_w * fast * sigma[:, None]

        return residual(pos_s), residual(ori_s)

    def _apply_heading_residual(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        pos_gv: np.ndarray,
        ori_gv: np.ndarray,
        gv_rot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_w = body_pos + np.einsum("tij,tbj->tbi", gv_rot, pos_gv)
        ori_w = np.einsum("tij,tbj->tbi", gv_rot, ori_gv)
        return pos_w, _rotvec_left_apply(body_quat, ori_w)

    def apply_bodies_world(
        self,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        body_names: Sequence[str],
        seed: int,
        gv_rot: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Morphology + heading-frame residuals on world ``(T, B, 3)`` / ``(T, B, 4)``."""
        pos = np.asarray(body_pos, dtype=np.float64)
        quat = np.asarray(body_quat, dtype=np.float64)
        if self.profile.has_morphology:
            pos = self.apply_morphology(pos, body_names, seed)
        if self.profile.has_after_fk:
            pos_gv, ori_gv = self.body_residuals(len(pos), body_names, seed)
            pos, quat = self._apply_heading_residual(pos, quat, pos_gv, ori_gv, gv_rot)
        if self.profile.has_body_jitter:
            pos_gv, ori_gv = self.uniform_body_residuals(len(pos), len(body_names), seed + 19)
            pos, quat = self._apply_heading_residual(pos, quat, pos_gv, ori_gv, gv_rot)
        return pos, quat

    def apply_kpt2gv(
        self, kpt2gv: np.ndarray, body_names: Sequence[str], seed: int
    ) -> np.ndarray:
        """Morphology + heading-frame residuals on ``(T, B, 4, 4)`` gv poses."""
        if not self.profile.has_body_warp:
            return kpt2gv
        out = np.asarray(kpt2gv, dtype=np.float64).copy()
        if self.profile.has_morphology:
            out[:, :, :3, 3] = self.apply_morphology(out[:, :, :3, 3], body_names, seed)
        if self.profile.has_after_fk:
            pos_gv, ori_gv = self.body_residuals(len(out), body_names, seed)
            noise_r = (
                R.from_rotvec(ori_gv.reshape(-1, 3))
                .as_matrix()
                .reshape(*ori_gv.shape[:-1], 3, 3)
            )
            out[:, :, :3, :3] = noise_r @ out[:, :, :3, :3]
            out[:, :, :3, 3] += pos_gv
        if self.profile.has_body_jitter:
            pos_gv, ori_gv = self.uniform_body_residuals(
                len(out), len(body_names), seed + 19
            )
            noise_r = (
                R.from_rotvec(ori_gv.reshape(-1, 3))
                .as_matrix()
                .reshape(*ori_gv.shape[:-1], 3, 3)
            )
            out[:, :, :3, :3] = noise_r @ out[:, :, :3, :3]
            out[:, :, :3, 3] += pos_gv
        return out

    def cvel_from_kpt2gv(self, kpt2gv: np.ndarray) -> np.ndarray:
        """Causal ``(T, B, 6)`` gravity-view spatial velocity from keypoint poses.

        Used only after after-FK corruption: a mid-link that has left the
        skeleton no longer has a trustworthy FK ``cvel``, and Humanoid-GPT
        reads that field.
        """
        # No extra low-pass: mid-link vision/VLA fill-in is not jointly
        # filtered with the anchors, and Humanoid-GPT reads this velocity.
        kpt2gv = np.asarray(kpt2gv, dtype=np.float64)
        n_frames, n_bodies = kpt2gv.shape[:2]
        cvel = np.zeros((n_frames, n_bodies, 6), dtype=np.float64)
        pos = kpt2gv[:, :, :3, 3]
        cvel[1:, :, 3:] = (pos[1:] - pos[:-1]) / self.profile.dt
        rots = R.from_matrix(kpt2gv[:, :, :3, :3].reshape(-1, 3, 3)).as_matrix().reshape(
            n_frames, n_bodies, 3, 3
        )
        for t in range(1, n_frames):
            rel = np.einsum("bij,bkj->bik", rots[t], rots[t - 1])
            cvel[t, :, :3] = R.from_matrix(rel).as_rotvec() / self.profile.dt
        cvel[0] = cvel[1] if n_frames > 1 else 0.0
        return cvel

    def _lowpass_kpt2gv(self, kpt2gv: np.ndarray) -> np.ndarray:
        """Causal one-pole low-pass on gravity-view keypoint poses."""
        rc = 1.0 / (2.0 * np.pi * self.profile.vel_filter_hz)
        alpha = self.profile.dt / (self.profile.dt + rc)
        out = kpt2gv.copy()
        for t in range(1, len(kpt2gv)):
            out[t, :, :3, 3] = out[t - 1, :, :3, 3] + alpha * (
                kpt2gv[t, :, :3, 3] - out[t - 1, :, :3, 3]
            )
            prev = R.from_matrix(out[t - 1, :, :3, :3].reshape(-1, 3, 3))
            target = R.from_matrix(kpt2gv[t, :, :3, :3].reshape(-1, 3, 3))
            step = R.from_rotvec(alpha * (prev.inv() * target).as_rotvec())
            out[t, :, :3, :3] = (prev * step).as_matrix().reshape(-1, 3, 3)
        return out

    # ── stage 5 ───────────────────────────────────────────────────────
    def _ema_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Causal time-constant EMA; same recurrence as ``TimeConstantEMA``.

        ``y[t] = lerp(x[t], y[t-1], exp(-dt/tau))``. Tau is not scaled.
        """
        tau = self.profile.ema_tau_ms / 1000.0
        if tau <= 0.0:
            return qpos
        alpha = float(np.exp(-self.profile.dt / tau))
        out = np.asarray(qpos, dtype=np.float64).copy()
        for t in range(1, len(out)):
            out[t] = _lerp_qpos(out[t], out[t - 1], alpha)
        return out

    def _lowpass_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Causal one-pole low-pass; SO(3) part filtered in the tangent space."""
        p = self.profile
        rc = 1.0 / (2.0 * np.pi * p.vel_filter_hz)
        alpha = p.dt / (p.dt + rc)
        out = qpos.copy()
        for t in range(1, len(qpos)):
            out[t, :3] = out[t - 1, :3] + alpha * (qpos[t, :3] - out[t - 1, :3])
            out[t, 7:] = out[t - 1, 7:] + alpha * (qpos[t, 7:] - out[t - 1, 7:])
        prev = R.from_quat(wxyz_to_xyzw(qpos[0, 3:7]))
        for t in range(1, len(qpos)):
            target = R.from_quat(wxyz_to_xyzw(qpos[t, 3:7]))
            step = R.from_rotvec(alpha * (prev.inv() * target).as_rotvec())
            prev = prev * step
            xyzw = prev.as_quat()
            out[t, 3:7] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        return out

    def qvel_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Causal backward-difference ``(T, 6+n)`` qvel in the dataset convention.

        World-frame linear and angular velocity, then joint rates -- verified
        against the dataset's own ``qvel`` rather than assumed.
        """
        p = self.profile
        qpos = np.asarray(qpos, dtype=np.float64)
        n_frames, nq = qpos.shape
        qvel = np.zeros((n_frames, nq - 1), dtype=np.float64)
        qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / p.dt
        qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) / p.dt
        rots = R.from_quat(wxyz_to_xyzw(qpos[:, 3:7]))
        for t in range(1, n_frames):
            r_prev = rots[t - 1]
            w_local = (r_prev.inv() * rots[t]).as_rotvec() / p.dt
            qvel[t, 3:6] = r_prev.apply(w_local)
        qvel[0] = qvel[1] if n_frames > 1 else 0.0
        return qvel

    # ── entry points ──────────────────────────────────────────────────
    def corrupt_qpos(
        self,
        qpos: np.ndarray,
        seed: int,
        joint_names: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(noisy_qpos, source_indices)`` for a ``(T, 7+n)`` reference."""
        p = self.profile
        qpos = np.asarray(qpos, dtype=np.float64)
        n_frames, nq = qpos.shape
        n_joints = nq - 7
        if len(joint_names) != n_joints:
            raise ValueError(
                f"{len(joint_names)} joint names for {n_joints} articulated joints"
            )
        rng = np.random.default_rng(int(seed))

        src = self.source_indices(n_frames, rng)
        out = qpos[src].copy()

        s = p.scale
        joint_res = (
            self._slow_bias(
                rng, n_frames, (n_joints,), self._joint_sigma(joint_names, p.bias_joint_rad)
            )
            + self._correlated_jitter(rng, n_frames, joint_names)
            + self._chunk_offsets(
                rng, n_frames, (n_joints,), self._joint_sigma(joint_names, p.chunk_joint_rad)
            )
        )
        self._spikes(rng, joint_res, self._joint_sigma(joint_names, p.jitter_joint_rad))
        out[:, 7:] += joint_res

        root_pos_res = (
            self._slow_bias(rng, n_frames, (3,), p.bias_root_pos_m * s)
            + _ou(rng, n_frames, (3,), p.jitter_tau_s, p.dt) * (p.jitter_root_pos_m * s)
            + self._chunk_offsets(rng, n_frames, (3,), p.chunk_root_pos_m * s)
        )
        self._spikes(rng, root_pos_res, p.jitter_root_pos_m * s)
        out[:, :3] += root_pos_res

        yaw = (
            self._slow_bias(rng, n_frames, (1,), p.bias_root_yaw_rad * s)[:, 0]
            + _ou(rng, n_frames, (1,), p.jitter_tau_s, p.dt)[:, 0] * (p.jitter_root_yaw_rad * s)
        )
        roll_pitch = (
            self._slow_bias(rng, n_frames, (2,), p.bias_root_rp_rad * s)
            + _ou(rng, n_frames, (2,), p.jitter_tau_s, p.dt) * (p.jitter_root_rp_rad * s)
        )
        out[:, 3:7] = _rotvec_left_apply(out[:, 3:7], _yaw_rp_rotvecs(yaw, roll_pitch))
        if p.ema_tau_ms > 0.0 and p.scale > 0.0:
            out = self._ema_qpos(out)
        return out, src

    @staticmethod
    def gravity_view(qpos: np.ndarray) -> np.ndarray:
        """``(T, 4, 4)`` gravity-view-to-world transform, dataset convention.

        Heading-only rotation, root xy translation, z pinned to the ground.
        """
        n_frames = len(qpos)
        gv2w = np.repeat(np.eye(4, dtype=np.float64)[None], n_frames, axis=0)
        gv2w[:, :3, :3] = batch_base2navi(
            np.stack([quat2mat(q) for q in qpos[:, 3:7]])
        )
        gv2w[:, :2, 3] = qpos[:, :2]
        return gv2w

    @staticmethod
    def gv_vel(gv2w: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """``(T, 3)`` planar velocity command: gravity-view vx, vy and yaw rate."""
        lin = np.einsum("tji,tj->ti", gv2w[:, :3, :3], qvel[:, :3])
        ang = np.einsum("tji,tj->ti", gv2w[:, :3, :3], qvel[:, 3:6])
        return np.stack([lin[:, 0], lin[:, 1], ang[:, 2]], axis=-1)

    def keypoint_fields(
        self, mj_model, qpos: np.ndarray, qvel: np.ndarray, gv2w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gravity-view keypoint poses and spatial velocities, dataset layout."""
        import mujoco

        n_frames = len(qpos)
        body_ids = np.array([mj_model.body(n).id for n in KPT_NAMES])
        data = mujoco.MjData(mj_model)
        n_kpt = len(KPT_NAMES)
        kpt2gv = np.zeros((n_frames, n_kpt, 4, 4), dtype=np.float32)
        cvel_gv = np.zeros((n_frames, n_kpt, 6), dtype=np.float32)
        ident = np.eye(4, dtype=np.float64)
        for t in range(n_frames):
            data.qpos[:] = qpos[t]
            data.qvel[:] = qvel[t]
            mujoco.mj_forward(mj_model, data)
            kpt2w = np.repeat(ident[None], n_kpt, axis=0)
            kpt2w[:, :3, :3] = data.xmat[body_ids].reshape(-1, 3, 3)
            kpt2w[:, :3, 3] = data.xpos[body_ids]
            kpt2gv[t] = (np.linalg.inv(gv2w[t]) @ kpt2w).astype(np.float32)
            cvel_w = np.asarray(data.cvel[body_ids])
            w2gv = gv2w[t, :3, :3].T
            cvel_gv[t, :, :3] = (w2gv @ cvel_w[:, :3].T).T
            cvel_gv[t, :, 3:] = (w2gv @ cvel_w[:, 3:].T).T
        return kpt2gv, cvel_gv

    def corrupt_traj(
        self,
        ref_traj: Dict,
        mj_model,
        seed: int,
        joint_names: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> Dict:
        """Copy of ``ref_traj`` with a corrupted command stream for the policy.

        ``fields`` restricts recomputation to what the caller's policy reads; the
        keypoint pass is a per-frame ``mj_forward``, so skipping it for a backend
        that ignores keypoints is worth the plumbing.
        """
        if joint_names is None:
            joint_names = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]
        noisy_qpos, src = self.corrupt_qpos(ref_traj["qpos"], seed, joint_names)

        mode = self.profile.vel_mode
        if mode == VelocityMode.CLEAN:
            noisy_qvel = np.asarray(ref_traj["qvel"], dtype=np.float64)[src]
        elif mode == VelocityMode.FILTERED:
            noisy_qvel = self.qvel_from_qpos(self._lowpass_qpos(noisy_qpos))
        else:
            noisy_qvel = self.qvel_from_qpos(noisy_qpos)

        out = dict(ref_traj)
        out["qpos"] = noisy_qpos
        out["qvel"] = noisy_qvel
        wanted = None if fields is None else set(fields)

        def requested(key: str) -> bool:
            return key in ref_traj and (wanted is None or key in wanted)

        # The root pose and planar velocity are also published as separate
        # fields. Leaving them clean would hand the policy an uncorrupted side
        # channel that contradicts its own noisy qpos, which reads as extra
        # noise rather than less.
        gv2w = self.gravity_view(noisy_qpos)
        if requested("gv2wrd_pose"):
            out["gv2wrd_pose"] = gv2w.astype(np.float32)
        if requested("gv_vel"):
            # Applied as a residual so whatever convention produced the stored
            # field survives, and only our injected difference is added.
            clean_qpos = np.asarray(ref_traj["qpos"], dtype=np.float64)
            clean_qvel = np.asarray(ref_traj["qvel"], dtype=np.float64)
            delta = self.gv_vel(gv2w, noisy_qvel) - self.gv_vel(
                self.gravity_view(clean_qpos), clean_qvel
            )[src]
            out["gv_vel"] = (
                np.asarray(ref_traj["gv_vel"], dtype=np.float64)[src] + delta
            ).astype(np.float32)

        kpt_keys = [
            k for k in ("kpt2gv_pose", "kpt_cvel_in_gv", "kpt_cvel") if requested(k)
        ]
        # FK keypoints from the *anchor* qpos (timing / GMR residual only).
        # Null-space joint noise is stored on ``qpos`` for joint-reading
        # policies (HGPT must not keep a clean G1 side channel under
        # ``xsens_human``) but must not move the bodies that after-FK / morph
        # then treat as the command.
        fk_qpos, fk_qvel = noisy_qpos, noisy_qvel
        if self.profile.nullspace_joint_rad > 0.0 and self.profile.scale > 0.0:
            mid_joint = self._nullspace_joint_residual(
                noisy_qpos.shape[0], joint_names, after_fk_seed(seed) + 1
            )
            joint_qpos = noisy_qpos.copy()
            joint_qpos[:, 7:] = joint_qpos[:, 7:] + mid_joint
            out["qpos"] = joint_qpos
            if mode != VelocityMode.CLEAN:
                out["qvel"] = (
                    self.qvel_from_qpos(self._lowpass_qpos(joint_qpos))
                    if mode == VelocityMode.FILTERED
                    else self.qvel_from_qpos(joint_qpos)
                )

        if kpt_keys:
            kpt2gv, cvel = self.keypoint_fields(mj_model, fk_qpos, fk_qvel, gv2w)
            if self.profile.has_body_warp:
                kpt2gv = self.apply_kpt2gv(kpt2gv, KPT_NAMES, after_fk_seed(seed))
                cvel = self.cvel_from_kpt2gv(kpt2gv)
            for key in kpt_keys:
                out[key] = (
                    kpt2gv.astype(np.float32)
                    if key == "kpt2gv_pose"
                    else cvel.astype(np.float32)
                )
        if requested("foot_contact"):
            out["foot_contact"] = np.asarray(ref_traj["foot_contact"])[src]
        return out


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def measure(
    model: RefNoiseModel,
    ref_traj: Dict,
    mj_model,
    seed: int,
) -> Dict[str, float]:
    """Injected-noise statistics, so a sweep can report what it actually applied."""
    joint_names = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]
    clean_qpos = np.asarray(ref_traj["qpos"], dtype=np.float64)
    clean_qvel = np.asarray(ref_traj["qvel"], dtype=np.float64)
    noisy = model.corrupt_traj(ref_traj, mj_model, seed, joint_names)

    dv = noisy["qvel"][:, 6:] - clean_qvel[:, 6:]
    return {
        "joint_pos_rms_deg": float(np.rad2deg(_rms(noisy["qpos"][:, 7:] - clean_qpos[:, 7:]))),
        "root_pos_rms_m": _rms(noisy["qpos"][:, :3] - clean_qpos[:, :3]),
        "joint_vel_rms": _rms(dv),
        "joint_vel_signal_rms": _rms(clean_qvel[:, 6:]),
        "joint_vel_noise_ratio": _rms(dv) / max(_rms(clean_qvel[:, 6:]), 1e-9),
        "root_vel_noise_ratio": _rms(noisy["qvel"][:, :3] - clean_qvel[:, :3])
        / max(_rms(clean_qvel[:, :3]), 1e-9),
    }


def stage_breakdown(
    profile: RefNoiseProfile,
    qpos: np.ndarray,
    joint_names: Sequence[str],
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Reference error contributed by each stage, enabled cumulatively.

    Worth reporting because the stages are not comparable in size: at a realistic
    latency the *staleness* of the command can exceed everything the additive
    terms contribute, and a sweep over ``scale`` that hides this looks flat.
    """
    off = dict(
        bias_joint_rad=0.0, bias_root_pos_m=0.0, bias_root_yaw_rad=0.0,
        bias_root_rp_rad=0.0, jitter_joint_rad=0.0, jitter_root_pos_m=0.0,
        jitter_root_yaw_rad=0.0, jitter_root_rp_rad=0.0, spike_prob=0.0,
        chunk_hz=0.0, chunk_joint_rad=0.0, chunk_root_pos_m=0.0,
    )
    def keep(*names: str) -> Dict[str, float]:
        """Zero out every stage except the ones whose fields start with ``names``."""
        return {k: v for k, v in off.items() if not k.startswith(names)}

    stages = {
        "timing": replace(profile, **off),
        "timing+bias": replace(profile, **keep("bias")),
        "timing+bias+jitter": replace(profile, **keep("bias", "jitter", "spike")),
        "all": profile,
    }
    qpos = np.asarray(qpos, dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for name, stage in stages.items():
        noisy, _ = RefNoiseModel(stage).corrupt_qpos(qpos, seed, joint_names)
        out[name] = {
            "joint_pos_rms_deg": float(np.rad2deg(_rms(noisy[:, 7:] - qpos[:, 7:]))),
            "root_pos_rms_cm": 100.0 * _rms(noisy[:, :3] - qpos[:, :3]),
        }
    return out
