"""Humanoid-GPT policy and simulation backend for ``eval_parallel_tracker``."""

import os
import sys
import torch
import pickle
import hashlib
import numpy as np
import cv2
from absl import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from xml.etree import ElementTree

import mujoco
from jax import tree_util as jtu

# <repo>/src/humantracker/eval/backends/<this file>
ROOT = Path(__file__).resolve().parents[4]
HGPT_ROOT = ROOT / "thirdparty" / "Humanoid-GPT"
if not HGPT_ROOT.is_dir():
    raise FileNotFoundError(f"Humanoid-GPT checkout not found: {HGPT_ROOT}")
# Humanoid-GPT pins numpy/jax/mujoco versions that conflict with the other
# trackers, so it is vendored rather than installed; its `tracking` and `utils`
# packages are resolved from the checkout.
sys.path.insert(0, str(HGPT_ROOT))

# Imported for its side effects: it installs the console handler and raises absl's
# verbosity to INFO, the level this backend's tables are logged at.
from utils.logger import LOGGER  # noqa: F401
from tracking.convert_qpos2kpt import qpos2kpt
from tracking.policy import Args
from tracking.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config
from humantracker.eval.core.geometry import render_frame
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
# Two protocol names come straight from core; the two printers are wrapped below so
# that they log rather than print.
from humantracker.eval.core.summary import (
    compute_category_summary,
    compute_overall_summary,
    print_category_summary as _print_category_table,
    print_overall_summary as _print_overall_table,
)
from humantracker.eval.core.termination_metrics import calculate_trajectory_length
from humantracker.eval.core.tracking_errors import (
    calculate_joint_tracking_error,
    calculate_kpt_mae_error,
    calculate_root_tracking_error,
)
from humantracker.eval.backends import load_ref_traj
from humantracker.eval.paths import required_file


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"


@dataclass
class EvalHGPTArgs(Args):
    """The subset of upstream's argument object that its rollout code reads.

    ``build_context`` fills every field from the evaluation command line, which is
    defined in ``eval_parallel_tracker`` and not restated here.
    """

    load_path: str = "thirdparty/Humanoid-GPT/storage/ckpts/pns_wo_priv216.onnx"
    device: str = "auto"
    privileged: bool = False
    sim_xml_path: str = "storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml"
    termination_metric: str = "trunk"
    video_width: int = 3840
    video_height: int = 2160
    rollout_dir: str = ""
    rollout_tracker: str = ""
    rollout_run_id: str = ""
    rollout_group: str = "rlhf_rollout"


def _convert_traj_to_kpt(data: Dict, mj_model: mujoco.MjModel, freq_tgt: int) -> Dict:
    freq_src = float(data["frequency"]) if "frequency" in data else float(freq_tgt)
    qpos_src = np.asarray(data["qpos"], dtype=np.float32)
    return qpos2kpt(
        mj_model,
        qpos_src=qpos_src,
        freq_src=freq_src,
        freq_tgt=freq_tgt,
        interp_sec=0.0,
        end_default_sec=0.0,
        debug=False,
        foot_contact_est=False,
        height_clip_mode=None,
        video_path=None,
    )


# --------------- Cache utilities ---------------

def _file_hash(file_path: Path) -> str:
    stat = file_path.stat()
    key = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_kpt_cached(
    file_path: Path,
    convert_mj_model: mujoco.MjModel,
    freq_tgt: int,
    cache_dir: Path,
) -> Dict:
    kpt_cache_dir = cache_dir / "kpt"
    kpt_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{file_path.stem}_{_file_hash(file_path)}_f{freq_tgt}"
    cache_path = kpt_cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    kpt_data = _convert_traj_to_kpt(load_ref_traj(file_path), convert_mj_model, freq_tgt)
    with open(cache_path, "wb") as f:
        pickle.dump(kpt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return kpt_data


def _onnx_providers_for_device(device: str):
    import onnxruntime as ort

    device_name = str(device).lower()
    if device_name.startswith("cuda"):
        device_id = 0
        if ":" in device_name:
            device_id = int(device_name.split(":", 1)[1])
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is unavailable for the HGPT policy")
        return [("CUDAExecutionProvider", {"device_id": device_id})]
    if device_name == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"Unsupported HGPT device: {device}")


def _onnx_session(load_path: str, providers):
    import onnxruntime as ort

    requested = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    session_options = ort.SessionOptions()
    if requested != "CPUExecutionProvider":
        # Refuse a silent downgrade to CPU when a GPU provider was asked for. ORT rejects
        # this setting when the CPU EP is itself the requested provider, so it is only
        # meaningful in the GPU case.
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(load_path, sess_options=session_options, providers=providers)
    if session.get_providers()[0] != requested:
        raise RuntimeError(f"HGPT ONNX session did not bind to {requested}")
    return session


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


class _EvalMLPPolicyONNX:
    def __init__(self, load_path: str, device: str):
        providers = _onnx_providers_for_device(device)
        self.onnx_model = _onnx_session(load_path, providers)
        print(f"[ONNX] Loaded {load_path}  providers={self.onnx_model.get_providers()}")

    def infer(self, obs) -> np.ndarray:
        return self.onnx_model.run(["continuous_actions"], {"obs": _to_numpy(obs)})[0]


class _EvalTransformerPolicyONNX:
    def __init__(self, load_path: str, device: str, max_seq_len: int, sample_steps: int):
        providers = _onnx_providers_for_device(device)
        self.onnx_model = _onnx_session(load_path, providers)
        print(f"[ONNX] Loaded {load_path}  providers={self.onnx_model.get_providers()}")
        self.max_seq_len = max_seq_len
        self.sample_steps = sample_steps
        self.obs_list = []

    def infer(self, obs) -> np.ndarray:
        self.obs_list.append(_to_numpy(obs))
        self.obs_list = self.obs_list[-self.max_seq_len:]
        obs_list = np.asarray(self.obs_list, dtype=np.float32).transpose(1, 0, 2)
        return self.onnx_model.run(["continuous_actions"], {"obs": obs_list})[0][:, -1, :]


def _build_policy(args: EvalHGPTArgs):
    if args.load_path.endswith(".onnx"):
        import onnxruntime as ort
        # Shape probing does not need CUDA. Keeping this CPU-only avoids
        # initializing CUDA in the parent process before rollout workers start.
        sess = ort.InferenceSession(args.load_path, providers=["CPUExecutionProvider"])
        input_shape = sess.get_inputs()[0].shape
        expected_rank = len(input_shape)
        if expected_rank == 2:
            logging.info(f"[Policy] Detected MLP ONNX model (input rank={expected_rank})")
            return _EvalMLPPolicyONNX(args.load_path, args.device)
        else:
            logging.info(f"[Policy] Detected Transformer ONNX model (input rank={expected_rank})")
            return _EvalTransformerPolicyONNX(
                args.load_path,
                args.device,
                max_seq_len=int(args.max_seq_len),
                sample_steps=int(args.sample_steps),
            )
    raise ValueError(f"Humanoid-GPT policy must be a .onnx file: {args.load_path}")


def _evaluate_single_traj(
    traj_id: int,
    ref_traj: Dict,
    file_name: str,
    args: EvalHGPTArgs,
    env_cfg,
    policy=None,
    collect_rm_features: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    source_path: Optional[str] = None,
    category: Optional[str] = None,
):
    if policy is None:
        raise ValueError("HGPT evaluation requires an initialized policy")
    local_policy = policy
    _init_qpos = ref_traj["qpos"][0].copy()
    _init_qpos[:2] = 0.0
    mj_sim = G1TrackMjSim(init_qpos=_init_qpos, headless=True, xml_path=args.sim_xml_path)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, local_policy, privileged=args.privileged)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    # optional video renderer
    renderer = None
    free_cam = None
    video_writer = None
    if record_video:
        renderer = mujoco.Renderer(
            mj_sim.mj_model,
            height=args.video_height,
            width=args.video_width,
        )
        free_cam = mujoco.MjvCamera()
        if video_path:
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                50,
                (args.video_width, args.video_height),
            )
            if not video_writer.isOpened():
                renderer.close()
                raise RuntimeError(f"Failed to open video writer: {video_path}")

    traj_metrics = {
        "kpt_pos_errors": [], "kpt_rot_errors": [],
        "joint_pos_errors": [], "joint_vel_errors": [],
        "root_pos_errors": [], "root_vel_errors": [], "root_yaw_errors": [],
        "state_history": [],
        "motor_target_history": [],
        "feature_history": [],
    }

    traj_len = len(ref_traj["qpos"])
    for track_step in range(traj_len):
        ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
        track_step_next = np.clip(track_step + 1, 0, traj_len - 1)
        ref_next = jtu.tree_map(lambda x: x[track_step_next][None], ref_traj)

        action = infer_fn.infer_onnx(state, {"ref_curr": ref_curr, "ref_next": ref_next})

        state = mj_sim.step(state, action)

        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(state, ref_curr, ref_next, mj_sim.mj_model)
        joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(state, ref_curr)
        root_pos_err_mm, root_vel_err_mms, root_yaw_err = calculate_root_tracking_error(state, ref_curr)

        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
        traj_metrics["joint_pos_errors"].append(joint_pos_mae)
        traj_metrics["joint_vel_errors"].append(joint_vel_mae)
        traj_metrics["root_pos_errors"].append(root_pos_err_mm)
        traj_metrics["root_vel_errors"].append(root_vel_err_mms)
        traj_metrics["root_yaw_errors"].append(root_yaw_err)
        traj_metrics["state_history"].append({
            "qpos": state.mj_data.qpos.copy(),
            "qvel": state.mj_data.qvel.copy(),
            "xpos": state.mj_data.xpos.copy(),
            "xmat": state.mj_data.xmat.copy(),
            "site_xpos": state.mj_data.site_xpos.copy(),
        })
        mt = infer_fn.info["motor_targets"][0]
        mt = np.asarray(mt).reshape(-1)
        traj_metrics["motor_target_history"].append(mt.copy())

        # Real per-frame RM features (always, so rollouts are training-ready).
        nn_action_h = np.asarray(infer_fn.info["nn_action"][0]).reshape(-1)
        traj_metrics["feature_history"].append(
            extract_frame_fields(
                mj_sim.mj_model, state.mj_data, ref_curr, ref_next, nn_action_h, mt
            )
        )

        # render frame for video
        if renderer is not None:
            frame = render_frame(renderer, state.mj_data, free_cam)
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # --- Aggregate (works with partial data from early termination) ---
    steps_completed = len(traj_metrics["kpt_pos_errors"])
    if steps_completed > 0:
        traj_length_ratio, termination_step = calculate_trajectory_length(
            traj_metrics["state_history"], ref_traj, mj_sim.mj_model,
            args.termination_metric,
        )
        avg_kpt_pos_error = float(np.mean(traj_metrics["kpt_pos_errors"]))
        avg_kpt_rot_error = float(np.mean(traj_metrics["kpt_rot_errors"]))
        avg_joint_pos_error = float(np.mean(traj_metrics["joint_pos_errors"]))
        avg_joint_vel_error = float(np.mean(traj_metrics["joint_vel_errors"]))
        avg_root_pos_error = float(np.mean(traj_metrics["root_pos_errors"]))
        avg_root_vel_error = float(np.mean(traj_metrics["root_vel_errors"]))
        avg_root_yaw_error = float(np.mean(traj_metrics["root_yaw_errors"]))
    else:
        traj_length_ratio, termination_step = 0.0, 0
        avg_kpt_pos_error = avg_kpt_rot_error = float("inf")
        avg_joint_pos_error = avg_joint_vel_error = float("inf")
        avg_root_pos_error = avg_root_vel_error = avg_root_yaw_error = float("inf")

    result = {
        "traj_id": traj_id,
        "file_name": file_name,
        "length_ratio": traj_length_ratio,
        "termination_step": termination_step,
        "total_frames": traj_len,
        "kpt_pos_mae": avg_kpt_pos_error,
        "kpt_rot_mae": avg_kpt_rot_error,
        "joint_pos_mae": avg_joint_pos_error,
        "joint_vel_mae": avg_joint_vel_error,
        "root_pos_err_mm": avg_root_pos_error,
        "root_vel_err_mms": avg_root_vel_error,
        "root_yaw_err": avg_root_yaw_error,
        "score": None,
    }

    # ── kinematic smoothness (acceleration / jerk) and contact consistency ──
    if steps_completed > 0:
        sm = compute_smoothness_metrics(
            traj_metrics["state_history"],
            ctrl_dt=env_cfg.ctrl_dt,
            action_history=traj_metrics["motor_target_history"],
        )
        result.update(sm)
        cc = compute_contact_consistency_from_height(
            traj_metrics["state_history"], ref_traj, mj_sim.mj_model,
        )
        result.update(cc)
    else:
        result.update({
            "joint_acc_mean": float("inf"),
            "joint_jerk_mean": float("inf"),
            "joint_acc_rms": float("inf"),
            "joint_jerk_rms": float("inf"),
            "action_jerk_mean": float("inf"),
            "foot_contact_acc": float("inf"),
            "foot_contact_iou": float("inf"),
        })

    if args.rollout_dir and traj_metrics["state_history"]:
        rollout_path, rollout_meta_path = save_tool_rollout_npz(
            output_root=args.rollout_dir,
            tracker_name=args.rollout_tracker,
            source_path=source_path or file_name,
            ref_traj=ref_traj,
            state_history=traj_metrics["state_history"],
            feature_history=traj_metrics["feature_history"],
            metrics=result,
            category=category,
            fps=int(round(1.0 / float(env_cfg.ctrl_dt))),
            ref_start_index=0,
            run_id=args.rollout_run_id or None,
            group=args.rollout_group,
        )
        result["rollout_path"] = str(rollout_path)
        result["rollout_meta_path"] = str(rollout_meta_path)

    # Finish the streaming video before releasing the renderer.
    if video_writer is not None:
        video_writer.release()
    if renderer is not None:
        renderer.close()

    if collect_rm_features:
        prompt_features, trajectory_features = sequence_features_from_history(
            traj_metrics["feature_history"],
            fps=int(round(1.0 / float(env_cfg.ctrl_dt))),
        )
        result["_prompt_features"] = prompt_features
        result["_traj_features"] = trajectory_features
    return result


def print_category_summary(category: str, metrics: List[Dict]) -> None:
    """Route the table through the Humanoid-GPT logger, where this backend's output goes."""
    _print_category_table(category, metrics, sink=logging.info)


def print_overall_summary(metrics: List[Dict]) -> None:
    """Failures keep their warning level; the rest of the table is informational."""
    _print_overall_table(metrics, sink=logging.info, alert=logging.warning)


# ═══════════════════════════════════════════════════════════════════════════
#   EVALUATION BACKEND PROTOCOL
#   See humantracker/eval/backends/__init__.py for what the runner expects.
# ═══════════════════════════════════════════════════════════════════════════

OPTIONS = (
    ("--policy", {
        "default": "thirdparty/Humanoid-GPT/storage/ckpts/pns_wo_priv216.onnx",
        "help": "Humanoid-GPT policy ONNX",
    }),
    ("--robot_xml", {
        "default": "storage/assets/unitree_g1_5010/g1_mjx_track_papergray.xml",
        "help": "robot model the evaluation scene must include",
    }),
    ("--privileged", {
        "action": "store_true",
        "help": "run the privileged policy variant",
    }),
    ("--convert", {
        "action": "store_true",
        "help": "convert reference motions to keypoints before evaluating",
    }),
    ("--convert_freq_tgt", {
        "type": int,
        "default": 50,
        "help": "target frequency of the keypoint conversion",
    }),
    ("--cache_dir", {
        "default": "storage/.cache",
        "help": "where converted keypoints are cached",
    }),
    ("--no_cache", {
        "action": "store_true",
        "help": "convert on every run instead of reading the cache",
    }),
)


def validate(args) -> None:
    if os.environ.get("G1_VERSION") != "5010":
        raise RuntimeError("Humanoid-GPT evaluation requires G1_VERSION=5010")
    if args.videos_only:
        raise ValueError("This backend collects reward-model features and cannot render only")
    required_file(args.policy)
    _validate_scene(args.xml_path, args.robot_xml)


def _validate_scene(scene_value: str, robot_value: str) -> None:
    """Require the evaluation scene to include exactly this backend's robot model."""
    scene_path = Path(required_file(scene_value))
    robot_path = Path(required_file(robot_value)).resolve()
    included = {
        (scene_path.parent / element.attrib["file"]).resolve()
        for element in ElementTree.parse(scene_path).getroot().findall("include")
    }
    if included != {robot_path}:
        raise ValueError(
            f"Humanoid-GPT scene must include exactly {robot_path}; "
            f"got {sorted(map(str, included))}"
        )


def build_context(args, xml_path: str) -> Dict:
    upstream = EvalHGPTArgs()
    upstream.load_path = required_file(args.policy)
    upstream.device = args.device
    upstream.privileged = bool(args.privileged)
    upstream.sim_xml_path = xml_path
    upstream.termination_metric = args.termination_metric
    upstream.video_width = int(args.video_width)
    upstream.video_height = int(args.video_height)
    upstream.rollout_dir = args.rollout_dir
    upstream.rollout_tracker = args.rollout_tracker or args.tracker
    upstream.rollout_run_id = args.rollout_run_id
    upstream.rollout_group = args.rollout_group
    return {
        "args": args,
        "upstream_args": upstream,
        # Upstream's inference config, not its training config: the training one lives
        # in Humanoid-GPT's unreleased train_utils, and the only fields read downstream
        # (ctrl_dt, action_scale, soft_joint_pos_limit_factor) are equal in both, for
        # either setting of `privileged`.
        "env_config": g1_infer_env_config(),
        "policy": _build_policy(upstream),
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
        "convert_model": mujoco.MjModel.from_xml_path(required_file(args.robot_xml))
        if args.convert
        else None,
    }


def _reference_motion(context: Dict, file_path: Path) -> Dict:
    args = context["args"]
    if not args.convert:
        return load_ref_traj(file_path)
    if args.no_cache:
        return _convert_traj_to_kpt(
            load_ref_traj(file_path), context["convert_model"], int(args.convert_freq_tgt)
        )
    return _load_kpt_cached(
        file_path,
        context["convert_model"],
        int(args.convert_freq_tgt),
        Path(args.cache_dir),
    )


def evaluate(context: Dict, task) -> Dict:
    traj_id, path, category, video_path = task
    file_path = Path(path)
    metric = _evaluate_single_traj(
        traj_id=traj_id,
        ref_traj=_reference_motion(context, file_path),
        file_name=file_path.name,
        args=context["upstream_args"],
        env_cfg=context["env_config"],
        policy=context["policy"],
        collect_rm_features=True,
        record_video=bool(video_path),
        video_path=video_path or None,
        source_path=path,
        category=category,
    )
    prompt_features = metric.pop("_prompt_features")
    traj_features = metric.pop("_traj_features")
    metric.update(score_reward_model(context["reward_model"], prompt_features, traj_features))
    return metric
