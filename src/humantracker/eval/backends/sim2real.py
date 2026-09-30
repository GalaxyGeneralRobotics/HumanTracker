"""sim2real backend: released G1 policies run through sim2real's own policy runtime.

Every tracker in :data:`~humantracker.eval.backends.SIM2REAL_POLICIES` ships as a
deploy ``policy.yaml`` + ONNX pair for `sim2real <https://github.com/EGalahad/sim2real>`_.
The upstream ``IntegratedPolicyRuntime`` owns everything policy-specific: observations,
joint order, action scale and PD gains all come from the YAML and are not restated
here. This module supplies only what every HumanTracker backend shares -- the
paper-gray scene, the reference motion, the metrics and HumanScore -- and steps MuJoCo
the way upstream's integrated sim2sim loop does, with sim2real's G1 torque limits.

sim2real requires Python 3.10 and runs from its own environment, so its packages are
imported inside the functions that need them: the module still imports, and
``--help`` still works, from the HumanTracker environment. sim2real reads reference
motions through an any4hdmi dataset, so ``--motion_view`` points at a manifest over
the test motions; see :mod:`humantracker.eval.sim2real.prepare_motion_view`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import mujoco
import numpy as np
import yaml

from humantracker.eval.backends import SIM2REAL_POLICIES, load_ref_traj
from humantracker.eval.core.mj_sim import State
from humantracker.eval.core.rm_feature_extractor import (
    extract_frame_fields,
    sequence_features_from_history,
)
from humantracker.eval.core.rm_scorer import load_reward_model, score_reward_model
from humantracker.eval.core.rollout_export import save_tool_rollout_npz
from humantracker.eval.core.smoothness_metrics import (
    compute_contact_consistency_from_height,
    compute_smoothness_metrics,
)
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
from humantracker.eval.paths import repo_path, required_dir, required_file

CTRL_DT = 0.02  # 50 Hz, the control rate of every released policy
FPS = int(round(1.0 / CTRL_DT))

# YAML keys that name a robot model the runtime loads from disk. TeleopIT's points
# outside its checkpoint directory, at a file sim2real does not ship (see sim2real/setup.sh).
_MODEL_PATH_KEYS = {"xml_path", "mjcf_path"}
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


# ═══════════════════════════════════════════════════════════════════════════
#   POLICY FILES
# ═══════════════════════════════════════════════════════════════════════════

def _policy_yaml(args) -> Path:
    return required_dir(args.sim2real_root) / "checkpoints" / SIM2REAL_POLICIES[args.tracker] / "policy.yaml"


def _local_model_paths(node, base: Path) -> Iterator[Path]:
    """Yield every on-disk robot model the policy YAML refers to, resolved against it."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _MODEL_PATH_KEYS and isinstance(value, str) and not _URI.match(value):
                yield base / value
            else:
                yield from _local_model_paths(value, base)
    elif isinstance(node, list):
        for value in node:
            yield from _local_model_paths(value, base)


def _view_file(args, file_path: Path) -> Path:
    """The motion view's hard link to ``file_path``, at the same dataset-relative path."""
    return Path(args.motion_view) / "motions" / file_path.relative_to(repo_path(args.mocap_path))


# ═══════════════════════════════════════════════════════════════════════════
#   UPSTREAM RUNTIME
# ═══════════════════════════════════════════════════════════════════════════

def _runtime_for_motion(context: Dict, motion_path: Path):
    """Return the worker's ``IntegratedPolicyRuntime``, reset to the start of ``motion_path``.

    The runtime cannot be built without a motion, and building it creates the ONNX
    session (1.6 GB for HoloMotion), so a worker builds it on its first trajectory and
    afterwards swaps the motion in place. The swap mutates the existing state processor
    rather than replacing it, because every observation term holds a reference to it.
    """
    from sim2real.rl_policy.utils.motion import MotionDataset, motion_dataset_first_motion
    from sim2real.sim_env.integrated_sim2sim import IntegratedPolicyRuntime, IntegratedSim2SimArgs

    required_file(motion_path)
    policy = context["policy"]
    if policy is None:
        args = context["args"]
        policy = IntegratedPolicyRuntime(
            args=IntegratedSim2SimArgs(
                policy_config=str(_policy_yaml(args)),
                motion_path=str(motion_path),
                robot="g1",
                env_dt=CTRL_DT,
                sim_dt=args.sim_dt,
                initial_pause_s=0.0,
                inference_backend=args.inference_backend,
                headless=True,
            ),
            robot_cfg=context["robot_cfg"],
        )
        context["policy"] = policy
    else:
        # Mirrors the motion half of upstream's IntegratedMotionState.__init__.
        state = policy.state_processor
        motion = motion_dataset_first_motion(MotionDataset.create_from_path(
            str(motion_path),
            robot_cfg=context["robot_cfg"],
            mjcf_path=state.motion_config.get("mjcf_path"),
        ))
        state.motion_dataset = motion
        state.motion_length = motion.num_steps
        state.motion_joint_names = list(motion.joint_names)
        state.motion_body_names = list(motion.body_names)
        state.motion_config["motion_path"] = str(motion_path)

    policy.total_inference_cnt = 0
    policy.state_dict = {
        "action": np.zeros(policy.num_actions, dtype=np.float32),
        "paused": False,
        "control_mode": "policy",
    }
    policy.state_processor.reset()
    return policy


def _sync_state(policy, data: mujoco.MjData) -> None:
    """Copy the simulated robot into the runtime's state processor.

    The scene's joint order equals sim2real's G1 order (checked in ``build_context``),
    so this is the index-free form of upstream's ``sync_policy_state``.
    """
    state = policy.state_processor
    state.root_pos_w[:] = data.qpos[:3]
    state.root_quat_w[:] = data.qpos[3:7]
    state.root_lin_vel_w[:] = data.qvel[:3]
    state.root_ang_vel_b[:] = data.qvel[3:6]
    state.joint_pos[:] = data.qpos[7:]
    state.joint_vel[:] = data.qvel[6:]
    state.joint_torque[:] = data.actuator_force
    state.low_state_tick = int(data.time * 1000)


def _apply_command(context: Dict, data: mujoco.MjData, command) -> None:
    """Hold one policy command for a control period: PD plus feedforward, clipped."""
    target, target_vel, feedforward, kp, kd = command
    model = context["model"]
    limit = context["effort_limit"]
    for _ in range(round(CTRL_DT / context["args"].sim_dt)):
        torque = feedforward + kp * (target - data.qpos[7:]) + kd * (target_vel - data.qvel[6:])
        data.ctrl[:] = np.clip(torque, -limit, limit)
        mujoco.mj_step(model, data)


def _check_stepped(data: mujoco.MjData, expected_time: float, step: int) -> None:
    """Fail the trajectory if the last control period diverged.

    On divergence MuJoCo warns and resets the state -- clock included -- instead of
    raising, so an unchecked rollout would silently restart from the initial pose and
    be scored as if it were still tracking.
    """
    if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, expected_time, rtol=0, atol=1e-8)):
        raise FloatingPointError(f"Nonfinite state or MuJoCo auto-reset at step {step}")


def _ref_frames(ref: Dict, step: int) -> Tuple[Dict, Dict]:
    """The reference at ``step`` and at the following frame, each with a batch axis."""
    length = len(ref["qpos"])

    def frame(index: int) -> Dict:
        return {key: value[index][None] for key, value in ref.items()
                if isinstance(value, np.ndarray) and len(value) == length}

    return frame(step), frame(min(step + 1, length - 1))


# ═══════════════════════════════════════════════════════════════════════════
#   SINGLE TRAJECTORY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(context: Dict, task: Tuple[int, str, str, str]) -> Dict:
    traj_id, path, category, _video_path = task
    args = context["args"]
    model = context["model"]
    file_path = Path(path)
    ref = load_ref_traj(file_path)
    if args.seed is not None:
        np.random.seed(args.seed + traj_id)

    policy = _runtime_for_motion(context, _view_file(args, file_path))
    data = mujoco.MjData(model)
    data.qpos[:] = ref["qpos"][0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _sync_state(policy, data)
    policy.reset()
    policy.state_dict["paused"] = False
    state = State(mj_data=data)

    # ── metric accumulators ──
    kpt_pos_errs, kpt_rot_errs = [], []
    joint_pos_errs, joint_vel_errs = [], []
    root_pos_errs, root_vel_errs, root_yaw_errs = [], [], []
    state_history, motor_target_history, feature_history = [], [], []

    traj_len = len(ref["qpos"])
    for step in range(traj_len):
        ref_curr, ref_next = _ref_frames(ref, step)

        # ── infer & step sim ──
        _sync_state(policy, data)
        command = policy.step()
        if command is None:
            raise RuntimeError(f"Policy inference failed at frame {step}")
        # Upstream's loop advances this after every step; ScaleBFM's buffered
        # observations only refresh when it changes.
        policy.total_inference_cnt += 1
        motor_target = np.asarray(command[0])
        expected_time = data.time + CTRL_DT
        _apply_command(context, data, command)
        _check_stepped(data, expected_time, step)

        # ── metrics ──
        if "kpt2gv_pose" in ref:
            kpe, kre = calculate_kpt_mae_error(state, ref_curr, ref_next, model)
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
            name: getattr(data, name).copy()
            for name in ("qpos", "qvel", "xpos", "xmat", "site_xpos")
        })
        motor_target_history.append(motor_target.copy())
        feature_history.append(extract_frame_fields(
            model, data, ref_curr, ref_next, policy.state_dict["action"], motor_target
        ))

    # ── aggregate ──
    traj_len_ratio, term_step = calculate_trajectory_length(
        state_history, ref, model, args.termination_metric
    )
    result = {
        "traj_id": traj_id,
        "file_name": file_path.name,
        "length_ratio": traj_len_ratio,
        "termination_step": term_step,
        "total_frames": traj_len,
        "joint_pos_mae": float(np.mean(joint_pos_errs)),
        "joint_vel_mae": float(np.mean(joint_vel_errs)),
        "root_pos_err_mm": float(np.mean(root_pos_errs)),
        "root_vel_err_mms": float(np.mean(root_vel_errs)),
        "root_yaw_err": float(np.mean(root_yaw_errs)),
        "kpt_pos_mae": float(np.mean(kpt_pos_errs)) if kpt_pos_errs else float("inf"),
        "kpt_rot_mae": float(np.mean(kpt_rot_errs)) if kpt_rot_errs else float("inf"),
    }
    result.update(compute_smoothness_metrics(
        state_history, ctrl_dt=CTRL_DT, action_history=motor_target_history,
    ))
    # Jerk over the executed part only: after termination a fallen robot's jerk is
    # not tracking quality. Needs four frames for a third difference.
    prefix_frames = int(np.clip(term_step, 0, len(state_history)))
    result["jerk_prefix_frames"] = prefix_frames
    result["joint_jerk_until_termination"] = (
        compute_smoothness_metrics(state_history[:prefix_frames], ctrl_dt=CTRL_DT)["joint_jerk_mean"]
        if prefix_frames >= 4 else None
    )
    result.update(compute_contact_consistency_from_height(state_history, ref, model))
    score_prompt_feats, score_traj_feats = sequence_features_from_history(feature_history, fps=FPS)
    result.update(score_reward_model(context["reward_model"], score_prompt_feats, score_traj_feats))

    if args.rollout_dir:
        rollout_path, rollout_meta_path = save_tool_rollout_npz(
            output_root=args.rollout_dir,
            tracker_name=args.rollout_tracker or args.tracker,
            source_path=path,
            ref_traj=ref,
            state_history=state_history,
            feature_history=feature_history,
            metrics=result,
            category=category,
            fps=FPS,
            ref_start_index=0,
            run_id=args.rollout_run_id or None,
            group=args.rollout_group,
        )
        result["rollout_path"] = str(rollout_path)
        result["rollout_meta_path"] = str(rollout_meta_path)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#   EVALUATION BACKEND PROTOCOL
#   See humantracker/eval/backends/__init__.py for what the runner expects.
# ═══════════════════════════════════════════════════════════════════════════

OPTIONS = (
    ("--sim2real_root", {
        "required": True,
        "help": "patched sim2real checkout holding the released checkpoints/ "
                "(setup_thirdparty.sh clones it under thirdparty/)",
    }),
    ("--motion_view", {
        "required": True,
        "help": "any4hdmi view of the test motions (humantracker.eval.sim2real.prepare_motion_view)",
    }),
    ("--sim_dt", {
        "type": float,
        "default": 0.005,
        "help": "physics step; must divide the 0.02 s control period",
    }),
    ("--inference_backend", {"default": "onnx-cpu", "choices": ["onnx-cpu", "onnx-gpu"]}),
    ("--seed", {
        "type": int,
        "default": None,
        "help": "seed numpy with seed + traj_id before each trajectory",
    }),
)


def validate(args) -> None:
    if importlib.util.find_spec("sim2real") is None:
        raise ModuleNotFoundError(
            "The sim2real backend runs from sim2real's own environment: use "
            "<sim2real_root>/.venv/bin/python with PYTHONPATH=<HumanTracker>/src "
            "(see src/humantracker/eval/README.md)"
        )
    policy_yaml = Path(required_file(_policy_yaml(args)))
    with policy_yaml.open() as stream:
        policy_config = yaml.safe_load(stream)
    required_file(policy_yaml.parent / policy_config.get("model_path", "policy.onnx"))
    for model_path in _local_model_paths(policy_config, policy_yaml.parent):
        required_file(model_path)
    required_file(Path(args.motion_view) / "manifest.json")
    if args.sim_dt <= 0 or not np.isclose(CTRL_DT / args.sim_dt, round(CTRL_DT / args.sim_dt)):
        raise ValueError("--sim_dt must divide the 0.02 s control period")
    if args.video_interval:
        raise ValueError("The sim2real backend does not render videos")
    if args.ref_noise != "none":
        raise ValueError("The sim2real backend does not inject reference noise")


def build_context(args, xml_path: str) -> Dict:
    from sim2real.config.robots import get_robot_cfg

    robot_cfg = get_robot_cfg("g1")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = args.sim_dt
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    if joint_names != list(robot_cfg.joint_names):
        raise ValueError("HumanTracker scene joint order differs from sim2real's G1 config")
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    if actuator_names != joint_names:
        raise ValueError("HumanTracker scene must actuate each G1 joint once, in joint order")
    return {
        "args": args,
        "model": model,
        "robot_cfg": robot_cfg,
        "effort_limit": np.asarray([robot_cfg.joint_effort_limit[n] for n in joint_names]),
        "policy": None,  # built on the worker's first trajectory, see _runtime_for_motion
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
    }
