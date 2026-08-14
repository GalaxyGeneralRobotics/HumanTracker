"""TWIST2 policy and simulation backend for ``eval_parallel_tracker``."""

import os
import sys

# suppress ONNX Runtime GPU discovery warning (must be before ort import)
os.environ["ORT_LOG_LEVEL"] = "ERROR"

_cudnn_dir = list(__import__("nvidia.cudnn", fromlist=["cudnn"]).__path__)[0] + "/lib"
_cuda_rt_dir = list(
    __import__("nvidia.cuda_runtime", fromlist=["cuda_runtime"]).__path__
)[0] + "/lib"
os.environ["LD_LIBRARY_PATH"] = (
    _cuda_rt_dir + ":" + _cudnn_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

import warnings
import numpy as np
import mujoco
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", message=".*device_discovery.*")
ort.set_default_logger_severity(3)

# <repo>/src/humantracker/eval/backends/<this file>
ROOT = Path(__file__).resolve().parents[4]

if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

from humantracker.eval.core.geometry import (
    quat2mat,
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


# ═══════════════════════════════════════════════════════════════════════════
#   TWIST2 POLICY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

NUM_JOINTS = 29
CTRL_DT = 0.02       # 50 Hz control (decimation=10, sim_dt=0.002)
SIM_DT = 0.001       # 1000 Hz physics

# ── Observation dimensions (from g1_mimic_future_config.py / save_onnx.py) ──
N_MIMIC_OBS_SINGLE = 35   # 6 (root_vel_xy + root_pos_z + roll + pitch + yaw_ang_vel) + 29 (dof_pos)
N_PROPRIO = 92             # 3 (ang_vel) + 2 (imu) + 29 (dof_pos) + 29 (dof_vel) + 29 (last_action)
N_OBS_SINGLE = N_MIMIC_OBS_SINGLE + N_PROPRIO  # 127
HISTORY_LEN = 10
N_FUTURE_STEPS = 1         # TAR_MOTION_STEPS_FUTURE = [0]
N_FUTURE_OBS = N_FUTURE_STEPS * N_MIMIC_OBS_SINGLE  # 35
NUM_OBSERVATIONS = N_OBS_SINGLE * (HISTORY_LEN + 1) + N_FUTURE_OBS  # 127*11 + 35 = 1432

# ── Scaling factors (from g1.yaml) ──
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.5

# ── Default standing angles (MuJoCo order, from g1.yaml) ──
DEFAULT_ANGLES_MUJ = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,       # right leg
    0.0, 0.0, 0.0,                           # waist
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,    # left arm
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,   # right arm
], dtype=np.float64)

# ── PD gains (from official server_low_level_g1_sim.py) ──
TWIST2_KPS = np.array([
    100, 100, 100, 150, 40, 40,       # left leg
    100, 100, 100, 150, 40, 40,       # right leg
    150, 150, 150,                      # waist
    40, 40, 40, 40, 4.0, 4.0, 4.0,   # left arm
    40, 40, 40, 40, 4.0, 4.0, 4.0,   # right arm
], dtype=np.float64)

TWIST2_KDS = np.array([
    2, 2, 2, 4, 2, 2,         # left leg
    2, 2, 2, 4, 2, 2,         # right leg
    4, 4, 4,                    # waist
    5, 5, 5, 5, 0.2, 0.2, 0.2,  # left arm
    5, 5, 5, 5, 0.2, 0.2, 0.2,  # right arm
], dtype=np.float64)

# Torque limits (from official server_low_level_g1_sim.py)
TWIST2_TORQUE_LIMIT = np.array([
    100, 100, 100, 150, 40, 40,       # left leg
    100, 100, 100, 150, 40, 40,       # right leg
    150, 150, 150,                      # waist
    40, 40, 40, 40, 4.0, 4.0, 4.0,   # left arm
    40, 40, 40, 40, 4.0, 4.0, 4.0,   # right arm
], dtype=np.float64)

# Ankle indices (for zeroing ankle dof velocity in obs)
ANKLE_IDX = [4, 5, 10, 11]

# ── init qpos (MuJoCo format: [pos3, quat4, joints29]) ──
TWIST2_DEFAULT_QPOS = np.zeros(36, dtype=np.float64)
TWIST2_DEFAULT_QPOS[2] = 0.793      # standing height (from official sim2sim)
TWIST2_DEFAULT_QPOS[3] = 1.0        # quat w
TWIST2_DEFAULT_QPOS[7:] = DEFAULT_ANGLES_MUJ

# ═══════════════════════════════════════════════════════════════════════════
#   TWIST2 OBSERVATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class Twist2ObsBuilder:
    """Build 1432-D observation for TWIST2 ONNX policy.

    Observation layout (from g1_mimic_future.py compute_observations):
        obs_buf = [mimic_obs(35), proprio_obs(92)]  -> n_obs_single = 127
        final   = [current(127), history(10*127), future(35)]  -> 1432
    """

    def __init__(self, ref_traj: Dict):
        self.traj_len = len(ref_traj["qpos"])
        qpos = ref_traj["qpos"]   # (T, 36)
        qvel = ref_traj["qvel"]   # (T, 35)

        # Reference data in MuJoCo order
        self.ref_jpos = qpos[:, 7:]       # (T, 29)
        self.ref_jvel = qvel[:, 6:]       # (T, 29)
        self.ref_root_pos = qpos[:, :3]   # (T, 3)
        self.ref_root_quat = qpos[:, 3:7] # (T, 4) wxyz
        self.ref_root_vel = qvel[:, :3]   # (T, 3)
        self.ref_root_angvel = qvel[:, 3:6] # (T, 3)

        # History ring-buffer (10 frames of n_obs_single=127)
        self.hist_size = HISTORY_LEN
        self.hist_buf = np.zeros((self.hist_size, N_OBS_SINGLE), dtype=np.float32)

    def reset(self):
        self.hist_buf[:] = 0.0

    def _build_mimic_obs(self, frame: int, base_quat_wxyz: np.ndarray,
                          base_root_vel_local: np.ndarray) -> np.ndarray:
        """Build 35-D mimic observation for a given reference frame.

        Layout: root_vel_xy(2) + root_pos_z(1) + roll(1) + pitch(1) + yaw_ang_vel(1) + dof_pos(29)
        """
        # Reference root velocity in local frame (xy)
        ref_root_vel_world = self.ref_root_vel[frame]
        ref_root_quat = self.ref_root_quat[frame]
        # Rotate world velocity to local frame
        ref_root_vel_local = quat_rotate(quat_conj(ref_root_quat), ref_root_vel_world)

        # Reference root angular velocity in local frame
        ref_root_angvel_world = self.ref_root_angvel[frame]
        ref_root_angvel_local = quat_rotate(quat_conj(ref_root_quat), ref_root_angvel_world)

        # Reference roll/pitch from quaternion
        ref_euler = quat2euler(ref_root_quat)  # xyz order
        ref_roll = ref_euler[0]
        ref_pitch = ref_euler[1]

        mimic = np.concatenate([
            ref_root_vel_local[:2],                    # 2: root vel xy (local)
            self.ref_root_pos[frame, 2:3],             # 1: root pos z
            np.array([ref_roll]),                       # 1: roll
            np.array([ref_pitch]),                      # 1: pitch
            ref_root_angvel_local[2:3],                # 1: yaw angular velocity
            self.ref_jpos[frame],                       # 29: dof positions
        ]).astype(np.float32)
        return mimic  # (35,)

    def _build_proprio_obs(self, ang_vel: np.ndarray, roll: float, pitch: float,
                            dof_pos: np.ndarray, dof_vel: np.ndarray,
                            last_action: np.ndarray) -> np.ndarray:
        """Build 92-D proprioceptive observation.

        Layout: ang_vel(3) + imu(2) + dof_pos(29) + dof_vel(29) + last_action(29)
        """
        # Zero ankle dof velocity
        dof_vel_obs = dof_vel.copy()
        for idx in ANKLE_IDX:
            dof_vel_obs[idx] = 0.0

        proprio = np.concatenate([
            ang_vel * ANG_VEL_SCALE,                    # 3
            np.array([roll, pitch]),                     # 2
            (dof_pos - DEFAULT_ANGLES_MUJ) * DOF_POS_SCALE,  # 29
            dof_vel_obs * DOF_VEL_SCALE,                # 29
            last_action,                                 # 29
        ]).astype(np.float32)
        return proprio  # (92,)

    def build_obs(self, frame: int, ang_vel: np.ndarray, roll: float, pitch: float,
                  dof_pos: np.ndarray, dof_vel: np.ndarray, last_action: np.ndarray,
                  base_quat_wxyz: np.ndarray, base_root_vel_local: np.ndarray) -> np.ndarray:
        """Build full 1432-D observation and update history."""
        # Current mimic obs (for current reference frame)
        mimic_obs = self._build_mimic_obs(frame, base_quat_wxyz, base_root_vel_local)
        # Current proprio obs
        proprio_obs = self._build_proprio_obs(ang_vel, roll, pitch, dof_pos, dof_vel, last_action)
        # Current obs_single
        current_obs = np.concatenate([mimic_obs, proprio_obs])  # (127,)

        # History (oldest first, 10 * 127 = 1270)
        history = self.hist_buf.ravel()  # (1270,)

        # Future obs: mimic obs for the future frame (same as current frame for TAR_MOTION_STEPS_FUTURE=[0])
        future_obs = self._build_mimic_obs(frame, base_quat_wxyz, base_root_vel_local)  # (35,)

        # Update history buffer (push current, FIFO)
        self.hist_buf[:-1] = self.hist_buf[1:]
        self.hist_buf[-1] = current_obs

        # Concatenate: [current(127), history(1270), future(35)] = 1432
        obs = np.concatenate([current_obs, history, future_obs]).astype(np.float32)
        assert obs.shape[0] == NUM_OBSERVATIONS, f"obs dim {obs.shape[0]} != {NUM_OBSERVATIONS}"
        return obs[np.newaxis]  # (1, 1432)


# ═══════════════════════════════════════════════════════════════════════════
#   MUJOCO SIMULATION WRAPPER FOR TWIST2
# ═══════════════════════════════════════════════════════════════════════════

class Twist2MjSim(MJSim):
    def __init__(
        self,
        init_qpos: np.ndarray,
        headless: bool = True,
        xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    ):
        super().__init__(xml_path=xml_path, ctrl_dt=CTRL_DT, sim_dt=SIM_DT, headless=headless)
        self.kps = TWIST2_KPS.copy()
        self.kds = TWIST2_KDS.copy()
        self.torque_limit = TWIST2_TORQUE_LIMIT.copy()
        self.init_qpos = init_qpos.copy()


# ═══════════════════════════════════════════════════════════════════════════
#   TWIST2 ONNX INFERENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class Twist2OnnxPolicy:
    def __init__(self, policy_path: str, device: str = "cpu"):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for the TWIST2 policy")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is unavailable for the TWIST2 policy")
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Unsupported TWIST2 device: {device}")

        session_options = ort.SessionOptions()
        session_options.add_free_dimension_override_by_name("batch_size", 1)
        if providers[0] != "CPUExecutionProvider":
            # Refuse a silent downgrade to CPU when a GPU provider was asked for. ORT rejects
            # this setting when the CPU EP is itself the requested provider, so it is only
            # meaningful in the GPU case.
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            policy_path, sess_options=session_options, providers=providers
        )
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        print(f"[Policy] Providers: {self.session.get_providers()}")
        if device.startswith("cuda") and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("TWIST2 ONNX session did not bind to CUDAExecutionProvider")
        # Verify input dim
        in_shape = self.session.get_inputs()[0].shape
        print(f"[Policy] Input shape: {in_shape}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """obs: (1, 1432) -> action (1, 29) in MuJoCo order"""
        return self.session.run([self.out_name], {self.in_name: obs})[0]


# ═══════════════════════════════════════════════════════════════════════════
#   ACTION -> MOTOR TARGET CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def action_to_motor_targets(nn_action: np.ndarray) -> np.ndarray:
    """Convert raw NN output (29-D, MuJoCo order) to joint target.
    q_target = default + action * action_scale
    """
    return DEFAULT_ANGLES_MUJ + nn_action * ACTION_SCALE


# ═══════════════════════════════════════════════════════════════════════════
#   SINGLE TRAJECTORY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_trajectory(
    traj_id: int,
    ref_traj: Dict,
    file_name: str,
    policy: Twist2OnnxPolicy,
    reward_model=None,
    xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    verbose: bool = True,
    record_video: bool = False,
    video_path: Optional[str] = None,
    rollout_dir: Optional[str] = None,
    rollout_tracker: str = "twist2",
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
    mj_sim = Twist2MjSim(init_qpos=init_qpos, headless=True, xml_path=xml_path)
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
    obs_builder = Twist2ObsBuilder(ref_traj)
    obs_builder.reset()

    traj_len = len(ref_traj["qpos"])

    # ── metric accumulators ──
    kpt_pos_errs, kpt_rot_errs = [], []
    joint_pos_errs, joint_vel_errs = [], []
    root_pos_errs, root_vel_errs, root_yaw_errs = [], [], []
    state_history: List[Dict] = []
    motor_target_history: List[np.ndarray] = []
    feature_history: List[Dict] = []

    last_nn_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    for step in range(traj_len):
        ref_curr = {k: v[step][np.newaxis] for k, v in ref_traj.items()
                    if isinstance(v, np.ndarray) and len(v) == traj_len}
        next_idx = min(step + 1, traj_len - 1)
        ref_next = {k: v[next_idx][np.newaxis] for k, v in ref_traj.items()
                    if isinstance(v, np.ndarray) and len(v) == traj_len}

        # ── read sim state ──
        qpos = state.mj_data.qpos.copy()
        qvel = state.mj_data.qvel.copy()
        base_quat = qpos[3:7]

        # angular velocity in body frame
        ang_vel = get_sensor_data(mj_sim.mj_model, state.mj_data, "gyro_pelvis")

        # roll / pitch from quaternion
        euler = quat2euler(base_quat)
        roll, pitch = euler[0], euler[1]

        dof_pos = qpos[7:]
        dof_vel = qvel[6:]

        # root velocity in local frame (for mimic obs builder)
        rot_mat = quat2mat(base_quat[np.newaxis])[0]
        root_vel_local = rot_mat.T @ qvel[:3]

        # ── build obs & infer ──
        obs = obs_builder.build_obs(
            frame=step, ang_vel=ang_vel, roll=roll, pitch=pitch,
            dof_pos=dof_pos, dof_vel=dof_vel, last_action=last_nn_action,
            base_quat_wxyz=base_quat, base_root_vel_local=root_vel_local,
        )
        nn_action = policy.infer(obs)[0]  # (29,) MuJoCo order
        last_nn_action = nn_action.copy()

        # ── action -> motor target ──
        motor_target = action_to_motor_targets(nn_action)

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

        # Real per-frame RM features (always, so rollouts are training-ready).
        feature_history.append(
            extract_frame_fields(
                mj_sim.mj_model, state.mj_data, ref_curr, ref_next, nn_action, motor_target
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
        "default": "thirdparty/TWIST2/assets/ckpts/twist2_1017_25k.onnx",
        "help": "TWIST2 policy ONNX",
    }),
)


def validate(args) -> None:
    required_file(args.policy)


def build_context(args, xml_path: str) -> Dict:
    return {
        "args": args,
        "xml_path": xml_path,
        "policy": Twist2OnnxPolicy(required_file(args.policy), args.device),
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
        "load_ref_traj": load_ref_traj,
        "simulate": evaluate_single_trajectory,
    }


evaluate = evaluate_uniform
