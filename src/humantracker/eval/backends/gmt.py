"""GMT policy and simulation backend for ``eval_parallel_tracker``."""

import os
import sys

_cudnn_dir = list(__import__("nvidia.cudnn", fromlist=["cudnn"]).__path__)[0] + "/lib"
_cuda_rt_dir = list(
    __import__("nvidia.cuda_runtime", fromlist=["cuda_runtime"]).__path__
)[0] + "/lib"
os.environ["LD_LIBRARY_PATH"] = (
    _cuda_rt_dir + ":" + _cudnn_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

import numpy as np
import mujoco
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Optional

# <repo>/src/humantracker/eval/backends/<this file>
ROOT = Path(__file__).resolve().parents[4]

if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

from humantracker.eval.core.geometry import (
    quat2euler,
    quat_conj,
    quat_rotate,
    get_sensor_data,
    render_frame,
)
from humantracker.eval.core.smoothness_metrics import (
    compute_smoothness_metrics,
    compute_contact_consistency_from_height,
)
from humantracker.eval.core.rollout_export import save_tool_rollout_npz
from humantracker.eval.core.rm_feature_extractor import extract_frame_fields, sequence_features_from_history
from humantracker.eval.core.rm_scorer import (
    load_reward_model,
    score_reward_model,
)
from humantracker.eval.core.mj_sim import MJSim
# Four of the eight protocol names in backends/__init__.py, re-exported unchanged.
from humantracker.eval.core.summary import (
    compute_category_summary,
    compute_overall_summary,
    print_category_summary,
    print_overall_summary,
)
from humantracker.eval.core.termination_metrics import calculate_trajectory_length
from humantracker.eval.core.tracking_errors import (
    calculate_joint_tracking_error,
    calculate_kpt_mae_error,
    calculate_root_tracking_error,
)
from humantracker.eval.backends import evaluate_uniform, load_ref_traj
from humantracker.eval.paths import required_file


# GMT uses 23 DOFs (no wrist joints), sim XML uses 29 DOFs
GMT_NUM_JOINTS = 23
SIM_NUM_JOINTS = 29
CTRL_DT = 0.02       # 50 Hz control
SIM_DT = 0.001       # 1000 Hz physics

# ── Joint mapping: GMT 23-DOF indices within the 29-DOF sim ──
# GMT DOFs: 6 left leg + 6 right leg + 3 waist + 4 left arm + 4 right arm = 23
# Sim DOFs: same + 3 left wrist + 3 right wrist = 29
GMT_TO_SIM_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           15, 16, 17, 18, 22, 23, 24, 25])

# ── Observation dimensions (from sim2sim.py) ──
# mimic_obs per step: root_z(1) + roll(1) + pitch(1) + root_vel(3) + yaw_angvel(1) + dof_pos(23) = 30
N_MIMIC_OBS_SINGLE = 30
N_MIMIC_STEPS = 20
N_MIMIC_OBS = N_MIMIC_STEPS * N_MIMIC_OBS_SINGLE  # 600
# proprio: ang_vel(3) + rpy[:2](2) + dof_pos(23) + dof_vel(23) + last_action(23) = 74
N_PROPRIO = 74
HISTORY_LEN = 20
N_HISTORY = HISTORY_LEN * N_PROPRIO  # 1480
NUM_OBSERVATIONS = N_MIMIC_OBS + N_PROPRIO + N_HISTORY  # 2154

# ── Scaling factors (from sim2sim.py) ──
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.5

# ── Default standing angles for GMT 23 DOFs (from sim2sim.py) ──
GMT_DEFAULT_ANGLES = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # right leg
    0.0, 0.0, 0.0,                           # waist
    0.0, 0.4, 0.0, 1.2,                     # left arm (no wrist)
    0.0, -0.4, 0.0, 1.2,                    # right arm (no wrist)
], dtype=np.float64)

# ── Default standing angles for 29-DOF sim (same as TWIST2) ──
SIM_DEFAULT_ANGLES = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # right leg
    0.0, 0.0, 0.0,                           # waist
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,    # left arm
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,   # right arm
], dtype=np.float64)

# ── PD gains for 29-DOF sim ──
GMT_KPS = np.array([
    100, 100, 100, 150, 40, 40,   # left leg
    100, 100, 100, 150, 40, 40,   # right leg
    150, 150, 150,                  # waist
    40, 40, 40, 40, 20, 20, 20,   # left arm
    40, 40, 40, 40, 20, 20, 20,   # right arm
], dtype=np.float64)

GMT_KDS = np.array([
    2, 2, 2, 4, 2, 2,     # left leg
    2, 2, 2, 4, 2, 2,     # right leg
    4, 4, 4,                # waist
    5, 5, 5, 5, 1, 1, 1,  # left arm
    5, 5, 5, 5, 1, 1, 1,  # right arm
], dtype=np.float64)

GMT_TORQUE_LIMIT = np.array([
    88, 139, 88, 139, 50, 50,      # left leg  (aligned with sim2sim)
    88, 139, 88, 139, 50, 50,      # right leg (aligned with sim2sim)
    88, 50, 50,                      # waist     (aligned with sim2sim)
    25, 25, 25, 25, 25, 5, 5,      # left arm + wrist
    25, 25, 25, 25, 25, 5, 5,      # right arm + wrist
], dtype=np.float64)

# Ankle indices in GMT 23-DOF space (for zeroing ankle dof velocity in obs)
GMT_ANKLE_IDX = [4, 5, 10, 11]

# ── init qpos (MuJoCo format: [pos3, quat4, joints29]) ──
GMT_DEFAULT_QPOS = np.zeros(36, dtype=np.float64)
GMT_DEFAULT_QPOS[2] = 1.0        # standing height
GMT_DEFAULT_QPOS[3] = 1.0        # quat w
GMT_DEFAULT_QPOS[7:] = SIM_DEFAULT_ANGLES

# ── Mimic observation future timesteps (from sim2sim.py) ──
TAR_OBS_STEPS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


# ═══════════════════════════════════════════════════════════════════════════
#   QUATERNION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def quatToEuler_xyzw(quat_xyzw):
    """Convert quaternion (x,y,z,w) to euler (roll, pitch, yaw). Same as sim2sim.py."""
    qw = quat_xyzw[3]
    qx = quat_xyzw[0]
    qy = quat_xyzw[1]
    qz = quat_xyzw[2]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# ═══════════════════════════════════════════════════════════════════════════
#   GMT OBSERVATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class GmtObsBuilder:
    """Build 2154-D observation for GMT JIT policy.

    Observation layout (from sim2sim.py):
        obs_buf = [mimic_obs(600), proprio_obs(74), history(1480)]
        mimic_obs = 20 future steps × 30 dims each
        proprio_obs = ang_vel(3) + rpy[:2](2) + dof_pos(23) + dof_vel(23) + last_action(23)
        history = 20 × proprio_obs(74)
    """

    def __init__(self, ref_traj: Dict):
        self.traj_len = len(ref_traj["qpos"])
        qpos = ref_traj["qpos"]   # (T, 36)
        qvel = ref_traj["qvel"]   # (T, 35)

        # Extract 23-DOF subset from 29-DOF reference
        self.ref_jpos_23 = qpos[:, 7:][:, GMT_TO_SIM_IDX]       # (T, 23)
        self.ref_root_pos = qpos[:, :3]                           # (T, 3)
        self.ref_root_quat_wxyz = qpos[:, 3:7]                   # (T, 4) wxyz
        self.ref_root_vel = qvel[:, :3]                           # (T, 3)
        self.ref_root_angvel = qvel[:, 3:6]                       # (T, 3)

        # History ring-buffer
        self.hist_buf = np.zeros((HISTORY_LEN, N_PROPRIO), dtype=np.float32)

    def reset(self):
        self.hist_buf[:] = 0.0

    def _build_mimic_obs(self, curr_time_step: int) -> np.ndarray:
        """Build 600-D mimic observation: 20 future steps × 30 dims.

        Per step: root_z(1) + roll(1) + pitch(1) + root_vel_local(3) + yaw_angvel_local(1) + dof_pos_23(23) = 30
        This matches sim2sim.py _get_mimic_obs.
        """
        mimic_parts = []
        for step_offset in TAR_OBS_STEPS:
            future_time = (curr_time_step + step_offset) * CTRL_DT
            # Find the frame index (with looping)
            total_duration = self.traj_len * CTRL_DT
            if total_duration > 0:
                future_time = future_time % total_duration
            frame = min(int(future_time / CTRL_DT), self.traj_len - 1)

            ref_quat = self.ref_root_quat_wxyz[frame]
            ref_euler = quat2euler(ref_quat)
            ref_roll, ref_pitch = ref_euler[0], ref_euler[1]

            # Root velocity in local frame
            ref_vel_world = self.ref_root_vel[frame]
            ref_vel_local = quat_rotate(quat_conj(ref_quat), ref_vel_world)

            # Root angular velocity in local frame
            ref_angvel_world = self.ref_root_angvel[frame]
            ref_angvel_local = quat_rotate(quat_conj(ref_quat), ref_angvel_world)

            step_obs = np.concatenate([
                self.ref_root_pos[frame, 2:3],     # 1: root height z
                np.array([ref_roll]),                # 1: roll
                np.array([ref_pitch]),               # 1: pitch
                ref_vel_local,                       # 3: root vel (local)
                ref_angvel_local[2:3],               # 1: yaw angular velocity (local)
                self.ref_jpos_23[frame],             # 23: dof positions
            ]).astype(np.float32)
            mimic_parts.append(step_obs)

        return np.concatenate(mimic_parts)  # (600,)

    def _build_proprio_obs(self, ang_vel: np.ndarray, roll: float, pitch: float,
                            dof_pos_23: np.ndarray, dof_vel_23: np.ndarray,
                            last_action: np.ndarray) -> np.ndarray:
        """Build 74-D proprioceptive observation.

        Layout: ang_vel(3) + rpy[:2](2) + dof_pos(23) + dof_vel(23) + last_action(23)
        """
        dof_vel_obs = dof_vel_23.copy()
        for idx in GMT_ANKLE_IDX:
            dof_vel_obs[idx] = 0.0

        proprio = np.concatenate([
            ang_vel * ANG_VEL_SCALE,                          # 3
            np.array([roll, pitch]),                           # 2
            (dof_pos_23 - GMT_DEFAULT_ANGLES) * DOF_POS_SCALE,  # 23
            dof_vel_obs * DOF_VEL_SCALE,                      # 23
            last_action,                                       # 23
        ]).astype(np.float32)
        return proprio  # (74,)

    def build_obs(self, curr_time_step: int, ang_vel: np.ndarray, roll: float, pitch: float,
                  dof_pos_23: np.ndarray, dof_vel_23: np.ndarray,
                  last_action: np.ndarray) -> np.ndarray:
        """Build full 2154-D observation and update history."""
        mimic_obs = self._build_mimic_obs(curr_time_step)       # (600,)
        proprio_obs = self._build_proprio_obs(ang_vel, roll, pitch, dof_pos_23, dof_vel_23, last_action)  # (74,)

        # History (oldest first, 20 * 74 = 1480)
        history = self.hist_buf.ravel()  # (1480,)

        # Update history buffer (push current proprio, FIFO)
        self.hist_buf[:-1] = self.hist_buf[1:]
        self.hist_buf[-1] = proprio_obs

        # Concatenate: [mimic(600), proprio(74), history(1480)] = 2154
        obs = np.concatenate([mimic_obs, proprio_obs, history]).astype(np.float32)
        assert obs.shape[0] == NUM_OBSERVATIONS, f"obs dim {obs.shape[0]} != {NUM_OBSERVATIONS}"
        return obs[np.newaxis]  # (1, 2154)


# ═══════════════════════════════════════════════════════════════════════════
#   MUJOCO SIMULATION WRAPPER FOR GMT
# ═══════════════════════════════════════════════════════════════════════════

class GmtMjSim(MJSim):
    def __init__(
        self,
        init_qpos: np.ndarray,
        headless: bool = True,
        xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    ):
        super().__init__(xml_path=xml_path, ctrl_dt=CTRL_DT, sim_dt=SIM_DT, headless=headless)
        self.kps = GMT_KPS.copy()
        self.kds = GMT_KDS.copy()
        self.torque_limit = GMT_TORQUE_LIMIT.copy()
        self.init_qpos = init_qpos.copy()


# ═══════════════════════════════════════════════════════════════════════════
#   GMT JIT INFERENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class GmtJitPolicy:
    def __init__(self, policy_path: str, device: str = "cuda"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the GMT policy")
        self.device = device
        print(f"[Policy] Loading JIT model from: {policy_path}")
        self.model = torch.jit.load(policy_path, map_location=device)
        self.model.eval()
        print(f"[Policy] Loaded on {device}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """obs: (1, 2154) -> action (1, 23) in GMT DOF order"""
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.model(obs_tensor).cpu().numpy()
        return action


# ═══════════════════════════════════════════════════════════════════════════
#   ACTION -> MOTOR TARGET CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def action_to_motor_targets_29dof(nn_action_23: np.ndarray) -> np.ndarray:
    """Convert 23-D GMT action to 29-D sim motor target.
    q_target = default_29 + expanded_action * action_scale
    """
    target = SIM_DEFAULT_ANGLES.copy()
    target[GMT_TO_SIM_IDX] += nn_action_23 * ACTION_SCALE
    return target


# ═══════════════════════════════════════════════════════════════════════════
#   SINGLE TRAJECTORY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_trajectory(
    traj_id: int,
    ref_traj: Dict,
    file_name: str,
    policy: GmtJitPolicy,
    reward_model=None,
    xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    verbose: bool = True,
    record_video: bool = False,
    video_path: Optional[str] = None,
    rollout_dir: Optional[str] = None,
    rollout_tracker: str = "gmt",
    rollout_run_id: Optional[str] = None,
    rollout_group: str = "rlhf_rollout",
    source_path: Optional[str] = None,
    category: Optional[str] = None,
    video_width: int = 3840,
    video_height: int = 2160,
    render_only: bool = False,
    termination_metric: str = "trunk",
) -> Dict:
    # ── init sim ──
    init_qpos = ref_traj["qpos"][0].copy()
    init_qpos[:2] = 0.0
    mj_sim = GmtMjSim(init_qpos=init_qpos, headless=True, xml_path=xml_path)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    renderer = None
    free_cam = None
    video_writer = None
    if record_video:
        if video_path is None:
            raise ValueError("record_video=True requires video_path")
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(mj_sim.mj_model, height=video_height, width=video_width)
        free_cam = mujoco.MjvCamera()
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            50,
            (video_width, video_height),
        )
        if not video_writer.isOpened():
            renderer.close()
            raise RuntimeError(f"Failed to open video writer: {video_path}")

    # ── init obs builder ──
    obs_builder = GmtObsBuilder(ref_traj)
    obs_builder.reset()

    traj_len = len(ref_traj["qpos"])

    # ── metric accumulators ──
    kpt_pos_errs, kpt_rot_errs = [], []
    joint_pos_errs, joint_vel_errs = [], []
    root_pos_errs, root_vel_errs, root_yaw_errs = [], [], []
    state_history: List[Dict] = []
    motor_target_history: List[np.ndarray] = []
    feature_history: List[Dict] = []

    last_nn_action = np.zeros(GMT_NUM_JOINTS, dtype=np.float32)

    for step in range(traj_len):
        ref_curr = {k: v[step][np.newaxis] for k, v in ref_traj.items()
                    if isinstance(v, np.ndarray) and len(v) == traj_len}
        next_idx = min(step + 1, traj_len - 1)
        ref_next = {k: v[next_idx][np.newaxis] for k, v in ref_traj.items()
                    if isinstance(v, np.ndarray) and len(v) == traj_len}

        # ── read sim state ──
        qpos = state.mj_data.qpos.copy()
        qvel = state.mj_data.qvel.copy()
        base_quat_wxyz = qpos[3:7]

        # angular velocity in body frame
        ang_vel = get_sensor_data(mj_sim.mj_model, state.mj_data, "gyro_pelvis")

        # roll / pitch from quaternion (GMT uses xyzw internally via quatToEuler)
        # MuJoCo quat is wxyz, convert to xyzw for quatToEuler
        quat_xyzw = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
        rpy = quatToEuler_xyzw(quat_xyzw)
        roll, pitch = rpy[0], rpy[1]

        # Extract 23-DOF from 29-DOF sim state
        dof_pos_29 = qpos[7:]
        dof_vel_29 = qvel[6:]
        dof_pos_23 = dof_pos_29[GMT_TO_SIM_IDX]
        dof_vel_23 = dof_vel_29[GMT_TO_SIM_IDX]

        # ── build obs & infer ──
        obs = obs_builder.build_obs(
            curr_time_step=step, ang_vel=ang_vel, roll=roll, pitch=pitch,
            dof_pos_23=dof_pos_23, dof_vel_23=dof_vel_23, last_action=last_nn_action,
        )
        nn_action = policy.infer(obs)[0]  # (23,)
        last_nn_action = nn_action.copy()  # save BEFORE clipping (matches sim2sim)
        nn_action = np.clip(nn_action, -10., 10.)

        # ── action -> motor target (29-DOF) ──
        motor_target = action_to_motor_targets_29dof(nn_action)

        # ── step sim ──
        state = mj_sim.step(state, motor_target)

        if renderer is not None:
            frame = render_frame(renderer, state.mj_data, free_cam)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if render_only:
            continue

        # ── metrics ──
        if "kpt2gv_pose" in ref_traj:
            kpe, kre = calculate_kpt_mae_error(state, ref_curr, ref_next, mj_sim.mj_model)
            kpt_pos_errs.append(kpe)
            kpt_rot_errs.append(kre)

        jpe, jve = calculate_joint_tracking_error(state, ref_curr)
        joint_pos_errs.append(jpe)
        joint_vel_errs.append(jve)

        rpe, rve, rye = calculate_root_tracking_error(state, ref_curr)
        root_pos_errs.append(rpe)
        root_vel_errs.append(rve)
        root_yaw_errs.append(rye)

        state_history.append({
            "qpos": state.mj_data.qpos.copy(),
            "qvel": state.mj_data.qvel.copy(),
            "xpos": state.mj_data.xpos.copy(),
            "xmat": state.mj_data.xmat.copy(),
            "site_xpos": state.mj_data.site_xpos.copy(),
        })
        motor_target_history.append(motor_target.copy())

        # Real per-frame RM features (always). Map the 23-DOF GMT action into
        # the 29-joint MuJoCo order; unused joints carry a real zero residual.
        nn_action_29 = np.zeros(29, dtype=np.float32)
        nn_action_29[GMT_TO_SIM_IDX] = nn_action
        feature_history.append(
            extract_frame_fields(
                mj_sim.mj_model, state.mj_data, ref_curr, ref_next, nn_action_29, motor_target
            )
        )

    if renderer is not None:
        video_writer.release()
        renderer.close()
    if render_only:
        return {"traj_id": traj_id, "file_name": file_name}

    # ── aggregate ──
    traj_len_ratio, term_step = calculate_trajectory_length(
        state_history, ref_traj, mj_sim.mj_model, termination_metric
    )

    result = {
        "traj_id": traj_id,
        "file_name": file_name,
        "length_ratio": traj_len_ratio,
        "termination_step": term_step,
        "total_frames": traj_len,
        "joint_pos_mae": float(np.mean(joint_pos_errs)) if joint_pos_errs else float("inf"),
        "joint_vel_mae": float(np.mean(joint_vel_errs)) if joint_vel_errs else float("inf"),
        "root_pos_err_mm": float(np.mean(root_pos_errs)) if root_pos_errs else float("inf"),
        "root_vel_err_mms": float(np.mean(root_vel_errs)) if root_vel_errs else float("inf"),
        "root_yaw_err": float(np.mean(root_yaw_errs)) if root_yaw_errs else float("inf"),
    }

    if kpt_pos_errs:
        result["kpt_pos_mae"] = float(np.mean(kpt_pos_errs))
        result["kpt_rot_mae"] = float(np.mean(kpt_rot_errs))
    else:
        result["kpt_pos_mae"] = float("inf")
        result["kpt_rot_mae"] = float("inf")

    # ── kinematic smoothness (acceleration / jerk) and contact consistency ──
    sm = compute_smoothness_metrics(
        state_history, ctrl_dt=CTRL_DT, action_history=motor_target_history,
    )
    result.update(sm)
    cc = compute_contact_consistency_from_height(
        state_history, ref_traj, mj_sim.mj_model,
    )
    result.update(cc)

    score_prompt_feats, score_traj_feats = sequence_features_from_history(feature_history, fps=int(round(1.0 / CTRL_DT)))
    result.update(score_reward_model(reward_model, score_prompt_feats, score_traj_feats))

    if rollout_dir:
        rollout_path, rollout_meta_path = save_tool_rollout_npz(
            output_root=rollout_dir,
            tracker_name=rollout_tracker,
            source_path=source_path or file_name,
            ref_traj=ref_traj,
            state_history=state_history,
            feature_history=feature_history,
            metrics=result,
            category=category,
            fps=int(round(1.0 / CTRL_DT)),
            ref_start_index=0,
            run_id=rollout_run_id,
            group=rollout_group,
        )
        result["rollout_path"] = str(rollout_path)
        result["rollout_meta_path"] = str(rollout_meta_path)

    if verbose:
        sr_text = "PASS" if traj_len_ratio >= 1.0 else f"FAIL@{term_step}"
        print(f"  [{traj_id:5d}] {file_name:50s}  "
              f"len={traj_len_ratio:.2f} ({sr_text})  "
              f"MPJPE={result['joint_pos_mae']:.4f}  "
              f"kpt={result['kpt_pos_mae']:.4f}"
              + (f"  RM={result['score']:.3f}" if result['score'] is not None else ""))

    return result


# ═══════════════════════════════════════════════════════════════════════════
#   EVALUATION BACKEND PROTOCOL
#   See humantracker/eval/backends/__init__.py for what the runner expects.
# ═══════════════════════════════════════════════════════════════════════════

OPTIONS = (
    ("--policy", {
        "default": "thirdparty/humanoid-general-motion-tracking/assets/pretrained_checkpoints/pretrained.pt",
        "help": "GMT TorchScript policy",
    }),
)


def validate(args) -> None:
    required_file(args.policy)


def build_context(args, xml_path: str) -> Dict:
    return {
        "args": args,
        "xml_path": xml_path,
        "policy": GmtJitPolicy(required_file(args.policy), args.device),
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
        "load_ref_traj": load_ref_traj,
        "simulate": evaluate_single_trajectory,
    }


evaluate = evaluate_uniform
