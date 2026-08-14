"""GEAR-SONIC policy and simulation backend for ``eval_parallel_tracker``."""

import os
import sys

# suppress ONNX Runtime GPU discovery warning (must be before ort import)
os.environ["ORT_LOG_LEVEL"] = "ERROR"

# ── Add cuDNN lib dir to LD_LIBRARY_PATH so ORT can find libcudnn.so.9 ──
# NOTE: We only modify LD_LIBRARY_PATH here instead of using ctypes RTLD_GLOBAL
# because RTLD_GLOBAL would expose CUDA 12.8 runtime symbols globally, conflicting
# with torch when the NVIDIA driver only supports CUDA ≤12.2.
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
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", message=".*device_discovery.*")
ort.set_default_logger_severity(3)   # 3 = ERROR, suppresses C++ warnings

# <repo>/src/humantracker/eval/backends/<this file>
ROOT = Path(__file__).resolve().parents[4]

from humantracker.eval.core.geometry import (
    quat_conj,
    quat_mul,
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

if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"


# ═══════════════════════════════════════════════════════════════════════════
#   SONIC POLICY CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

NUM_JOINTS = 29
CTRL_DT = 0.02       # 50 Hz control
SIM_DT = 0.001       # 1000 Hz physics

# ── Joint order mapping (semantics verified against C++ action code) ──
# In C++ source: the array named `isaaclab_to_mujoco` is actually
# indexed by **MuJoCo index** and yields the **IsaacLab index**, and
# `mujoco_to_isaaclab` is indexed by **IsaacLab index** → MuJoCo.
# We use clear names here.
MUJ_TO_ISAB = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
     2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)  # muj_to_isab[muj_idx] = isab_idx
ISAB_TO_MUJ = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
     4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)  # isab_to_muj[isab_idx] = muj_idx

# ── Default standing angles (MuJoCo order) ──
DEFAULT_ANGLES_MUJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,     # right leg
    0.0, 0.0, 0.0,                               # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,        # right arm
])
DEFAULT_ANGLES_ISAB = DEFAULT_ANGLES_MUJ[ISAB_TO_MUJ]

# ── Motor armature / PD gain computation ──
_OMEGA = 10.0 * 2.0 * np.pi        # natural frequency (rad/s)
_ZETA = 2.0                          # damping ratio

_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}

def _stiff(motor: str) -> float:
    return _ARMATURE[motor] * _OMEGA ** 2

def _damp(motor: str) -> float:
    return 2.0 * _ZETA * _ARMATURE[motor] * _OMEGA

# Motor type for each joint (MuJoCo order)
_MOTOR_TYPES_MUJ = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",  # left leg
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",  # right leg
    "7520_14", "5020", "5020",                                     # waist
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",       # left arm
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",       # right arm
]

# Joints with 2× kp/kd multiplier: ankle pitch/roll (4,5,10,11), waist roll/pitch (13,14)
_KP_2X_JOINTS = {4, 5, 10, 11, 13, 14}

SONIC_KPS = np.array([
    (2.0 if i in _KP_2X_JOINTS else 1.0) * _stiff(_MOTOR_TYPES_MUJ[i])
    for i in range(NUM_JOINTS)
])
SONIC_KDS = np.array([
    (2.0 if i in _KP_2X_JOINTS else 1.0) * _damp(_MOTOR_TYPES_MUJ[i])
    for i in range(NUM_JOINTS)
])
SONIC_TORQUE_LIMIT = np.array([_EFFORT[_MOTOR_TYPES_MUJ[i]] for i in range(NUM_JOINTS)])

# Action scale (MuJoCo order): 0.25 * effort / stiffness_without_multiplier
SONIC_ACTION_SCALE_MUJ = np.array([
    0.25 * _EFFORT[_MOTOR_TYPES_MUJ[i]] / _stiff(_MOTOR_TYPES_MUJ[i])
    for i in range(NUM_JOINTS)
])

# ── init qpos (MuJoCo format: [pos3, quat4, joints29]) ──
SONIC_DEFAULT_QPOS = np.zeros(36, dtype=np.float32)
SONIC_DEFAULT_QPOS[2] = 0.78       # standing height
SONIC_DEFAULT_QPOS[3] = 1.0        # quat w
SONIC_DEFAULT_QPOS[7:] = DEFAULT_ANGLES_MUJ

# ═══════════════════════════════════════════════════════════════════════════
#   QUATERNION HELPERS (wxyz convention, matching Sonic C++)
# ═══════════════════════════════════════════════════════════════════════════

def _calc_heading_quat(q):
    """Extract yaw-only quaternion (rotation around z)."""
    h = np.array([q[0], 0.0, 0.0, q[3]])
    n = np.linalg.norm(h)
    return h / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])

def _calc_heading_quat_inv(q):
    return quat_conj(_calc_heading_quat(q))

def _quat_to_rotmat(q):
    """Convert wxyz quaternion to 3×3 rotation matrix."""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def _compute_gravity_dir(base_quat_wxyz):
    """Projected gravity in body frame: R^T @ [0,0,-1]."""
    return quat_rotate(quat_conj(base_quat_wxyz), np.array([0.0, 0.0, -1.0]))


# ═══════════════════════════════════════════════════════════════════════════
#   SONIC OBSERVATION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class SonicObsBuilder:
    """Build encoder and decoder observations for GEAR-SONIC."""

    def __init__(self, ref_traj: Dict, encoder_dim: int):
        """
        ref_traj: OmniTracker npz dict with keys:
            qpos  (T, 36)  — [root_pos(3), root_quat_wxyz(4), 29_joints_muj]
            qvel  (T, 35)  — [root_vel(6), 29_joints_muj]
        """
        self.encoder_dim = int(encoder_dim)
        self.traj_len = len(ref_traj["qpos"])
        qpos = ref_traj["qpos"]
        qvel = ref_traj["qvel"]

        # Reference joint pos/vel in IsaacLab order
        self.ref_jpos_isab = qpos[:, 7:][:, ISAB_TO_MUJ]
        self.ref_jvel_isab = qvel[:, 6:][:, ISAB_TO_MUJ]
        # Reference root
        self.ref_root_z = qpos[:, 2].copy()
        self.ref_root_quat = qpos[:, 3:7].copy()  # wxyz

        # History ring-buffers (10 frames, oldest-first)
        self.hist_size = 10
        self.hist_ang_vel = np.zeros((self.hist_size, 3))
        self.hist_body_q = np.zeros((self.hist_size, NUM_JOINTS))
        self.hist_body_dq = np.zeros((self.hist_size, NUM_JOINTS))
        self.hist_last_action = np.zeros((self.hist_size, NUM_JOINTS))
        self.hist_gravity_dir = np.tile([0.0, 0.0, -1.0], (self.hist_size, 1))

        # Heading alignment
        self.apply_delta = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_ref_root_quat = self.ref_root_quat[0].copy()

    # ── reset / update ──────────────────────────────────────────────────

    def reset(self, base_quat_wxyz: np.ndarray):
        init_heading = _calc_heading_quat(base_quat_wxyz)
        self.init_ref_root_quat = self.ref_root_quat[0].copy()
        data_heading_inv = _calc_heading_quat_inv(self.init_ref_root_quat)
        self.apply_delta = quat_mul(init_heading, data_heading_inv)

        grav = _compute_gravity_dir(base_quat_wxyz)
        self.hist_ang_vel[:] = 0.0
        self.hist_body_q[:] = 0.0
        self.hist_body_dq[:] = 0.0
        self.hist_last_action[:] = 0.0
        self.hist_gravity_dir[:] = grav

    def update_history(
        self,
        ang_vel: np.ndarray,
        body_q_isab: np.ndarray,
        body_dq_isab: np.ndarray,
        last_action: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ):
        """Push one new time-step onto the history (FIFO)."""
        self.hist_ang_vel[:-1] = self.hist_ang_vel[1:]
        self.hist_ang_vel[-1] = ang_vel
        self.hist_body_q[:-1] = self.hist_body_q[1:]
        self.hist_body_q[-1] = body_q_isab
        self.hist_body_dq[:-1] = self.hist_body_dq[1:]
        self.hist_body_dq[-1] = body_dq_isab
        self.hist_last_action[:-1] = self.hist_last_action[1:]
        self.hist_last_action[-1] = last_action
        self.hist_gravity_dir[:-1] = self.hist_gravity_dir[1:]
        self.hist_gravity_dir[-1] = _compute_gravity_dir(base_quat_wxyz)

    # ── encoder observation ─────────────────────────────────────────────

    def build_encoder_obs(
        self, current_frame: int, base_quat_wxyz: np.ndarray
    ) -> np.ndarray:
        parts: list = []
        fidx = self._future_frames(current_frame, 10, 5)
        command_multi_future = np.concatenate(
            [self.ref_jpos_isab[fidx].ravel(), self.ref_jvel_isab[fidx].ravel()]
        )
        command_z_multi_future = self.ref_root_z[fidx].reshape(-1)
        anchor_mf = np.concatenate([self._anchor_ori(f, base_quat_wxyz) for f in fidx])

        if self.encoder_dim in (640, 641, 642, 650, 651, 652):
            has_root_z = self.encoder_dim in (650, 651, 652)
            if self.encoder_dim in (641, 642, 651, 652):
                parts.append(np.array([0.0]))  # single encoder selector, not padding.
            if self.encoder_dim in (642, 652):
                parts.append(np.array([1.0]))  # G1 one-hot for legacy single-encoder export.
            parts.append(command_multi_future)
            if has_root_z:
                parts.append(command_z_multi_future)
            parts.append(anchor_mf)
            obs = np.concatenate(parts).astype(np.float32)
            assert obs.shape[0] == self.encoder_dim, (
                f"encoder dim {obs.shape[0]} != {self.encoder_dim}"
            )
            return obs[np.newaxis]
        if self.encoder_dim != 1762:
            raise ValueError(f"Unsupported SONIC encoder input dim: {self.encoder_dim}")

        # 1  encoder_mode_4  [mode_id, 0, 0, 0]  (4-D)  g1 mode_id=0
        parts.append(np.array([0.0, 0.0, 0.0, 0.0]))

        # 2  motion_joint_positions  10×29 = 290-D  (future, step=5)
        parts.append(self.ref_jpos_isab[fidx].ravel())

        # 3  motion_joint_velocities  290-D
        parts.append(self.ref_jvel_isab[fidx].ravel())

        # 4  motion_root_z_position_10frame_step5  10-D  (NOT required in g1 mode → zeros)
        parts.append(np.zeros(10))

        # 5  motion_root_z_position  1-D  (NOT required in g1 mode → zeros)
        parts.append(np.zeros(1))

        # 6  motion_anchor_orientation  6-D  (NOT required in g1 mode → zeros)
        parts.append(np.zeros(6))

        # 7  motion_anchor_orientation  10×6 = 60-D  (future, step=5)
        parts.append(np.concatenate(
            [self._anchor_ori(f, base_quat_wxyz) for f in fidx]
        ))

        # 8–14  unused modes → zeros
        parts.append(np.zeros(120))   # lowerbody jpos
        parts.append(np.zeros(120))   # lowerbody jvel
        parts.append(np.zeros(9))     # vr_3point target
        parts.append(np.zeros(12))    # vr_3point orn
        parts.append(np.zeros(720))   # smpl_joints
        parts.append(np.zeros(60))    # smpl_anchor_ori
        parts.append(np.zeros(60))    # wrist jpos

        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.encoder_dim, f"encoder dim {obs.shape[0]} != {self.encoder_dim}"
        return obs[np.newaxis]

    # ── decoder observation (994-D) ─────────────────────────────────────

    def build_decoder_obs(self, token_state: np.ndarray) -> np.ndarray:
        parts: list = []

        # 1  token_state  (64-D)
        parts.append(token_state.ravel()[:64])

        # 2  his_base_angular_velocity  10×3 = 30-D  (oldest first)
        parts.append(self.hist_ang_vel.ravel())

        # 3  his_body_joint_positions  10×29 = 290-D
        parts.append(self.hist_body_q.ravel())

        # 4  his_body_joint_velocities  290-D
        parts.append(self.hist_body_dq.ravel())

        # 5  his_last_actions  290-D
        parts.append(self.hist_last_action.ravel())

        # 6  his_gravity_dir  10×3 = 30-D
        parts.append(self.hist_gravity_dir.ravel())

        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == 994, f"decoder dim {obs.shape[0]} != 994"
        return obs[np.newaxis]        # (1, 994)

    # ── internal helpers ────────────────────────────────────────────────

    def _future_frames(self, cur: int, n: int, step: int) -> np.ndarray:
        frames = np.arange(n) * step + cur
        return np.clip(frames, 0, self.traj_len - 1)

    def _anchor_ori(self, frame: int, base_quat: np.ndarray) -> np.ndarray:
        """6-D anchor orientation: first 2 columns of rot-matrix, row-major."""
        ref_q = self.ref_root_quat[frame]
        new_ref = quat_mul(self.apply_delta, ref_q)
        base_to_ref = quat_mul(quat_conj(base_quat), new_ref)
        mat = _quat_to_rotmat(base_to_ref)          # (3,3)
        return mat[:, :2].ravel()                     # (6,)


# ═══════════════════════════════════════════════════════════════════════════
#   MUJOCO SIMULATION WRAPPER FOR SONIC
# ═══════════════════════════════════════════════════════════════════════════

class SonicMjSim(MJSim):
    """MuJoCo simulation with GEAR-SONIC PD gains and torque limits."""

    def __init__(
        self,
        init_qpos: np.ndarray,
        headless: bool = True,
        xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    ):
        super().__init__(
            xml_path=xml_path, ctrl_dt=CTRL_DT, sim_dt=SIM_DT, headless=headless
        )
        self.kps = SONIC_KPS.astype(np.float32)
        self.kds = SONIC_KDS.astype(np.float32)
        self.torque_limit = SONIC_TORQUE_LIMIT.astype(np.float32)
        self.init_qpos = init_qpos.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#   SONIC ONNX INFERENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def _preload_cudnn():
    """Preload libcudnn.so.9 from nvidia-cudnn-cu12 pip package.

    Must be called AFTER importing torch (so torch's CUDA runtime is already
    loaded and won't be affected) and BEFORE creating ORT CUDA sessions.
    """
    import ctypes

    cudnn_dir = list(
        __import__("nvidia.cudnn", fromlist=["cudnn"]).__path__
    )[0] + "/lib"
    ctypes.CDLL(cudnn_dir + "/libcudnn.so.9", mode=ctypes.RTLD_GLOBAL)


class SonicOnnxPolicy:
    def __init__(self, encoder_path: str, decoder_path: str, device: str = "cpu"):
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.startswith("cuda"):
            # IMPORTANT: Import torch FIRST so it loads its own bundled CUDA
            # runtime, THEN preload cuDNN for ORT.  If cuDNN is loaded before
            # torch, the CUDA 12.8 runtime from nvidia-cuda-runtime-cu12 pip
            # package will be exposed globally and conflict with the driver
            # (which may only support CUDA ≤12.2).
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for the SONIC policy")
            _preload_cudnn()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is unavailable for the SONIC policy")
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Unsupported SONIC device: {device}")

        session_options = ort.SessionOptions()
        if providers[0] != "CPUExecutionProvider":
            # Refuse a silent downgrade to CPU when a GPU provider was asked for. ORT rejects
            # this setting when the CPU EP is itself the requested provider, so it is only
            # meaningful in the GPU case.
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.encoder = ort.InferenceSession(
            encoder_path, sess_options=session_options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            decoder_path, sess_options=session_options, providers=providers
        )

        # Verify which provider is actually used
        enc_prov = self.encoder.get_providers()
        dec_prov = self.decoder.get_providers()
        print(f"[Policy] Encoder providers: {enc_prov}")
        print(f"[Policy] Decoder providers: {dec_prov}")
        if device.startswith("cuda") and (
            enc_prov[0] != "CUDAExecutionProvider" or dec_prov[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError("SONIC ONNX sessions did not bind to CUDAExecutionProvider")

        # cache I/O names
        enc_input = self.encoder.get_inputs()[0]
        dec_input = self.decoder.get_inputs()[0]
        self.enc_in_name = enc_input.name
        self.enc_out_name = self.encoder.get_outputs()[0].name
        self.dec_in_name = dec_input.name
        self.dec_out_name = self.decoder.get_outputs()[0].name
        self.encoder_input_dim = self._last_static_dim(enc_input.shape)
        self.decoder_input_dim = self._last_static_dim(dec_input.shape)
        print(f"[Policy] Encoder input dim: {self.encoder_input_dim}")
        print(f"[Policy] Decoder input dim: {self.decoder_input_dim}")

    @staticmethod
    def _last_static_dim(shape) -> int:
        dim = shape[-1]
        if isinstance(dim, str) or dim is None:
            raise ValueError(f"ONNX input has dynamic last dimension: {shape}")
        return int(dim)

    def encode(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Encode one SONIC observation into a token."""
        return self.encoder.run(
            [self.enc_out_name], {self.enc_in_name: encoder_obs}
        )[0]

    def decode(self, decoder_obs: np.ndarray) -> np.ndarray:
        """decoder_obs: (1, 994) → action (1, 29) in IsaacLab order"""
        return self.decoder.run(
            [self.dec_out_name], {self.dec_in_name: decoder_obs}
        )[0]


# ═══════════════════════════════════════════════════════════════════════════
#   ACTION → MOTOR TARGET CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def action_to_motor_targets(nn_action_isab: np.ndarray) -> np.ndarray:
    """Convert raw NN output (IsaacLab order, 29-D) to MuJoCo joint target (29-D).

    In MuJoCo order:
        q_target[mj] = default[mj] + nn_action[isab] * action_scale[mj]
    where isab = MUJ_TO_ISAB[mj]
    """
    q_target = DEFAULT_ANGLES_MUJ.copy()
    q_target += nn_action_isab[MUJ_TO_ISAB] * SONIC_ACTION_SCALE_MUJ
    return q_target


# ═══════════════════════════════════════════════════════════════════════════
#   SINGLE TRAJECTORY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_trajectory(
    traj_id: int,
    ref_traj: Dict,
    file_name: str,
    policy: SonicOnnxPolicy,
    reward_model=None,
    xml_path: str = str(ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"),
    verbose: bool = True,
    record_video: bool = False,
    video_path: Optional[str] = None,
    rollout_dir: Optional[str] = None,
    rollout_tracker: str = "sonic",
    rollout_run_id: Optional[str] = None,
    rollout_group: str = "rlhf_rollout",
    source_path: Optional[str] = None,
    category: Optional[str] = None,
    video_width: int = 3840,
    video_height: int = 2160,
    render_only: bool = False,
    termination_metric: str = "trunk",
) -> Dict:
    """Run one trajectory and return metrics dict."""
    # ── init sim ──
    init_qpos = ref_traj["qpos"][0].copy()
    init_qpos[:2] = 0.0                       # reset xy
    mj_sim = SonicMjSim(init_qpos=init_qpos, headless=True, xml_path=xml_path)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    # ── optional video renderer ──
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
    obs_builder = SonicObsBuilder(ref_traj, encoder_dim=policy.encoder_input_dim)
    base_quat = state.mj_data.qpos[3:7].copy()
    obs_builder.reset(base_quat)

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
        # ── reference for this frame ──
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

        # body_q (IsaacLab order, q - default)
        joint_pos_muj = qpos[7:]
        body_q_isab = joint_pos_muj[ISAB_TO_MUJ] - DEFAULT_ANGLES_ISAB
        body_dq_isab = qvel[6:][ISAB_TO_MUJ]

        # update history BEFORE building obs (so current state is included)
        obs_builder.update_history(ang_vel, body_q_isab, body_dq_isab,
                                   last_nn_action, base_quat)

        # ── encoder ──
        enc_obs = obs_builder.build_encoder_obs(step, base_quat)
        token = policy.encode(enc_obs)           # (1, 64)

        # ── decoder ──
        dec_obs = obs_builder.build_decoder_obs(token)
        nn_action = policy.decode(dec_obs)[0]    # (29,) IsaacLab order
        last_nn_action = nn_action.copy()

        # ── action → motor target ──
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

    # reward model
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
    ("--encoder", {
        "default": "thirdparty/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
        "help": "GEAR-SONIC encoder ONNX",
    }),
    ("--decoder", {
        "default": "thirdparty/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
        "help": "GEAR-SONIC decoder ONNX",
    }),
)


def validate(args) -> None:
    required_file(args.encoder)
    required_file(args.decoder)


def build_context(args, xml_path: str) -> Dict:
    return {
        "args": args,
        "xml_path": xml_path,
        "policy": SonicOnnxPolicy(
            required_file(args.encoder), required_file(args.decoder), args.device
        ),
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
        "load_ref_traj": load_ref_traj,
        "simulate": evaluate_single_trajectory,
    }


evaluate = evaluate_uniform
