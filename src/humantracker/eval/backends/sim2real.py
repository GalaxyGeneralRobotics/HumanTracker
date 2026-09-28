"""Evaluate released sim2real G1 policies with HumanTracker's metrics.

The upstream policy runtime owns observations, action scaling, joint order and PD
gains. This adapter only supplies the HumanTracker scene, motion and scoring.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from humantracker.eval.backends import load_ref_traj
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
from humantracker.eval.paths import required_dir, required_file


# Names match the public leaderboard; the four existing HumanTracker backends stay separate.
POLICIES = {
    "heft": "heft/pmg",
    "holomotion": "holomotion/v1_4_0",
    "mimiclite-ppo": "mimic-lite/ppo",
    "mimiclite-roa": "mimic-lite/roa",
    "mimiclite-v1.1": "mimic-lite/v1_1",
    "scalebfm-m": "scalebfm/humanoid_transformer_m",
    "scalebfm-xl": "scalebfm/humanoid_transformer_xl",
    "grit-v0.0.1": "grit/v0_0_1",
    "teleopit": "teleopit",
}

OPTIONS = (
    ("--sim2real_root", {"required": True, "help": "sim2real checkout with released checkpoints/"}),
    ("--motion_view", {"required": True, "help": "any4hdmi manifest and hard-linked test motions"}),
    ("--sim_dt", {"type": float, "default": 0.005, "help": "upstream integrated sim2sim physics step"}),
    ("--inference_backend", {"default": "onnx-cpu", "choices": ["onnx-cpu", "onnx-gpu"]}),
    ("--seed", {"type": int, "default": None, "help": "per-trajectory observation RNG seed"}),
)


def _policy_file(args) -> Path:
    return Path(args.sim2real_root) / "checkpoints" / POLICIES[args.tracker] / "policy.yaml"


def _view_file(args, file_path: Path) -> Path:
    relative = file_path.relative_to(Path(args.mocap_path).resolve())
    return Path(args.motion_view) / "motions" / relative


def validate(args) -> None:
    required_dir(args.sim2real_root)
    policy_file = required_file(_policy_file(args))
    with open(policy_file) as stream:
        policy_config = yaml.safe_load(stream)
    model_file = Path(policy_file).parent / policy_config.get("model_path", "policy.onnx")
    required_file(model_file)
    required_file(Path(args.motion_view) / "manifest.json")
    if args.tracker == "teleopit":
        required_file(Path(args.sim2real_root) / "legacy/data/robots/g1/g1-mjlab.xml")
    if args.sim_dt <= 0 or not np.isclose(0.02 / args.sim_dt, round(0.02 / args.sim_dt)):
        raise ValueError("--sim_dt must divide the 0.02 s control period")
    if args.video_interval:
        raise ValueError("sim2real backend does not render videos")


def build_context(args, xml_path: str) -> dict:
    from sim2real.config.robots import get_robot_cfg

    robot_cfg = get_robot_cfg("g1")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = args.sim_dt
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    if joint_names != list(robot_cfg.joint_names):
        raise ValueError("HumanTracker scene joint order differs from sim2real G1 config")
    if model.nu != len(joint_names):
        raise ValueError("HumanTracker scene must have one actuator per G1 joint")
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    if actuator_names != joint_names:
        raise ValueError("HumanTracker scene actuator order differs from G1 joints")
    effort_limit = np.asarray([robot_cfg.joint_effort_limit[n] for n in joint_names])
    return {
        "args": args,
        "model": model,
        "robot_cfg": robot_cfg,
        "effort_limit": effort_limit,
        "policy": None,
        "reward_model": load_reward_model(args.rm_checkpoint, args.rm_device),
    }


def _set_motion(context: dict, motion_path: Path):
    from sim2real.rl_policy.utils.motion import MotionDataset, motion_dataset_first_motion
    from sim2real.sim_env.integrated_sim2sim import IntegratedPolicyRuntime, IntegratedSim2SimArgs

    args = context["args"]
    required_file(motion_path)
    policy = context["policy"]
    if policy is None:
        runtime_args = IntegratedSim2SimArgs(
            policy_config=str(_policy_file(args)),
            motion_path=str(motion_path),
            robot="g1",
            env_dt=0.02,
            sim_dt=args.sim_dt,
            initial_pause_s=0.0,
            inference_backend=args.inference_backend,
            headless=True,
        )
        policy = IntegratedPolicyRuntime(args=runtime_args, robot_cfg=context["robot_cfg"])
        context["policy"] = policy
    else:
        state = policy.state_processor
        motion = motion_dataset_first_motion(MotionDataset.create_from_path(
            str(motion_path), robot_cfg=context["robot_cfg"],
            mjcf_path=state.motion_config.get("mjcf_path"),
        ))
        state.motion_dataset = motion
        state.motion_length = motion.num_steps
        state.motion_joint_names = list(motion.joint_names)
        state.motion_body_names = list(motion.body_names)
        state.motion_config["motion_path"] = str(motion_path)

    state = policy.state_processor
    policy.total_inference_cnt = 0
    policy.state_dict = {
        "action": np.zeros(policy.num_actions, dtype=np.float32),
        "paused": False,
        "control_mode": "policy",
    }
    state.reset()
    return policy


def _sync_state(policy, data: mujoco.MjData) -> None:
    state = policy.state_processor
    state.root_pos_w[:] = data.qpos[:3]
    state.root_quat_w[:] = data.qpos[3:7]
    state.root_lin_vel_w[:] = data.qvel[:3]
    state.root_ang_vel_b[:] = data.qvel[3:6]
    state.joint_pos[:] = data.qpos[7:]
    state.joint_vel[:] = data.qvel[6:]
    state.joint_torque[:] = data.actuator_force
    state.low_state_tick = int(data.time * 1000)


def _step(context: dict, data: mujoco.MjData, command) -> None:
    target, target_vel, feedforward, kp, kd = command
    limit = context["effort_limit"]
    for _ in range(round(0.02 / context["args"].sim_dt)):
        torque = feedforward + kp * (target - data.qpos[7:]) + kd * (target_vel - data.qvel[6:])
        data.ctrl[:] = np.clip(torque, -limit, limit)
        mujoco.mj_step(context["model"], data)


def _frames(ref: dict, step: int):
    length = len(ref["qpos"])
    def frame(index):
        return {key: value[index][None] for key, value in ref.items()
                if isinstance(value, np.ndarray) and len(value) == length}
    return frame(step), frame(min(step + 1, length - 1))


def evaluate(context: dict, task: tuple) -> dict:
    traj_id, path, category, _video_path = task
    args = context["args"]
    file_path = Path(path)
    ref = load_ref_traj(file_path)
    if args.seed is not None:
        np.random.seed(args.seed + traj_id)
    policy = _set_motion(context, _view_file(args, file_path))
    model = context["model"]
    data = mujoco.MjData(model)
    data.qpos[:] = ref["qpos"][0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _sync_state(policy, data)
    policy.reset()
    policy.state_dict["paused"] = False
    state = State(mj_data=data)

    kpt_pos, kpt_rot, joint_pos, joint_vel = [], [], [], []
    root_pos, root_vel, root_yaw = [], [], []
    history, targets, features = [], [], []
    length = len(ref["qpos"])
    for step in range(length):
        ref_curr, ref_next = _frames(ref, step)
        _sync_state(policy, data)
        command = policy.step()
        # Upstream advances this counter so buffered observations refresh each step.
        policy.total_inference_cnt += 1
        if command is None:
            raise RuntimeError(f"Policy inference failed at frame {step}")
        target = np.asarray(command[0])
        _step(context, data, command)
        if "kpt2gv_pose" in ref:
            pos, rot = calculate_kpt_mae_error(state, ref_curr, ref_next, model)
            kpt_pos.append(pos)
            kpt_rot.append(rot)
        pos, vel = calculate_joint_tracking_error(state, ref_curr)
        joint_pos.append(pos)
        joint_vel.append(vel)
        pos, vel, yaw = calculate_root_tracking_error(state, ref_curr)
        root_pos.append(pos)
        root_vel.append(vel)
        root_yaw.append(yaw)
        history.append({name: getattr(data, name).copy() for name in
                        ("qpos", "qvel", "xpos", "xmat", "site_xpos")})
        targets.append(target.copy())
        features.append(extract_frame_fields(
            model, data, ref_curr, ref_next, policy.state_dict["action"], target
        ))

    ratio, term_step = calculate_trajectory_length(history, ref, model, args.termination_metric)
    result = {
        "traj_id": traj_id,
        "file_name": file_path.name,
        "length_ratio": ratio,
        "termination_step": term_step,
        "total_frames": length,
        "joint_pos_mae": float(np.mean(joint_pos)),
        "joint_vel_mae": float(np.mean(joint_vel)),
        "root_pos_err_mm": float(np.mean(root_pos)),
        "root_vel_err_mms": float(np.mean(root_vel)),
        "root_yaw_err": float(np.mean(root_yaw)),
        "kpt_pos_mae": float(np.mean(kpt_pos)) if kpt_pos else float("inf"),
        "kpt_rot_mae": float(np.mean(kpt_rot)) if kpt_rot else float("inf"),
    }
    result.update(compute_smoothness_metrics(history, ctrl_dt=0.02, action_history=targets))
    result.update(compute_contact_consistency_from_height(history, ref, model))
    prompt, trajectory = sequence_features_from_history(features, fps=50)
    result.update(score_reward_model(context["reward_model"], prompt, trajectory))
    if args.rollout_dir:
        rollout, meta = save_tool_rollout_npz(
            output_root=args.rollout_dir,
            tracker_name=args.rollout_tracker or args.tracker,
            source_path=path,
            ref_traj=ref,
            state_history=history,
            feature_history=features,
            metrics=result,
            category=category,
            fps=50,
            ref_start_index=0,
            run_id=args.rollout_run_id or None,
            group=args.rollout_group,
        )
        result.update(rollout_path=str(rollout), rollout_meta_path=str(meta))
    return result
