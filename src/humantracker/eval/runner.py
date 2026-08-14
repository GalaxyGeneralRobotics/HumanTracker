"""Dataset validation, worker setup and result output for the tracker evaluation.

Nothing here names a tracker. The selected backend arrives as a module implementing the
protocol in :mod:`humantracker.eval.backends` and is the only thing that knows how a
policy is built or a trajectory simulated; this module owns everything that is the same
whichever tracker is being measured.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from humantracker.eval.backends import load_backend
from humantracker.eval.core.summary import FAILED
from humantracker.eval.paths import repo_path, required_dir, required_file

# Every tracker is measured in the same paper-gray scene, on its own robot dynamics.
DEFAULT_SCENE_XML = "storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml"

DATASET_DIRECTORY = {
    "HighlyDynamic": "HighlyDynamic",
    "Daily": "Daily",
    "Ground": "Ground",
    "Interaction": "Interaction",
}

_CTX: Dict = {}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_traj_files(args: argparse.Namespace) -> Tuple[Path, Path, List[Path], Dict[str, str]]:
    data_path = required_dir(args.mocap_path)
    test_json = Path(required_file(args.test_json))

    traj_files: List[Path] = []
    traj_categories: Dict[str, str] = {}
    with test_json.open() as f:
        test_info = json.load(f)
    if not isinstance(test_info, list):
        raise TypeError(f"Test manifest must contain a JSON list: {test_json}")

    missing_files = []
    for index, item in enumerate(test_info):
        if not isinstance(item, dict) or not {"path", "category"} <= item.keys():
            raise ValueError(f"Invalid test manifest entry at index {index}: {item!r}")
        relative_path = Path(item["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts or len(relative_path.parts) < 2:
            raise ValueError(f"Invalid relative motion path at index {index}: {relative_path}")

        category = str(item["category"])
        if category not in DATASET_DIRECTORY:
            raise ValueError(f"Unknown category at index {index}: {item['category']}")
        path_category = relative_path.parts[0]
        if path_category != category:
            raise ValueError(
                f"Manifest category/path mismatch at index {index}: "
                f"{item['category']} != {relative_path.parts[0]}"
            )

        physical_path = data_path / DATASET_DIRECTORY[category] / Path(*relative_path.parts[1:])
        if not physical_path.is_file():
            missing_files.append(str(physical_path))
            continue
        if args.skip_flipped and "_flipped" in physical_path.name:
            continue
        traj_files.append(physical_path)
        traj_categories[str(physical_path)] = category

    if missing_files:
        preview = "\n".join(missing_files[:20])
        raise FileNotFoundError(
            f"Test manifest references {len(missing_files)} missing motions. First entries:\n{preview}"
        )
    if len(set(traj_files)) != len(traj_files):
        raise ValueError("Test manifest resolves to duplicate motion paths")

    print(
        f"[Data] Validated {len(traj_files)}/{len(test_info)} trajectories from {test_json}",
        flush=True,
    )

    if args.categories:
        wanted = set(args.categories)
        unknown = wanted - DATASET_DIRECTORY.keys()
        if unknown:
            raise ValueError(f"Unknown category filters: {sorted(unknown)}")
        traj_files = [f for f in traj_files if traj_categories[str(f)] in wanted]
    if args.max_trajs > 0:
        traj_files = traj_files[: args.max_trajs]
    if not traj_files:
        raise ValueError(f"No trajectories to evaluate under {data_path}")
    return data_path, test_json, traj_files, traj_categories


def make_tasks(args: argparse.Namespace, traj_files: List[Path], traj_categories: Dict[str, str]):
    video_dir = repo_path(args.video_dir)
    if args.video_interval > 0:
        video_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, fpath in enumerate(traj_files):
        category = traj_categories[str(fpath)]
        do_video = args.video_interval > 0 and idx % args.video_interval == 0
        task_dir = video_dir / f"{idx:05d}_{fpath.stem}"
        video_path = str(task_dir / "video.mp4") if do_video else ""
        tasks.append((idx, str(fpath), category, video_path))
    return tasks


def init_worker(tracker: str, args_dict: Dict) -> None:
    """Build this process's backend context once, before any trajectory is evaluated."""
    args = SimpleNamespace(**args_dict)
    backend = load_backend(tracker)
    context = backend.build_context(args, required_file(args.xml_path))
    if "reward_model" not in context:
        raise KeyError(f"Backend {tracker!r} built a context without a reward model")
    print(f"[Worker {os.getpid()}] {tracker} policy/RM loaded", flush=True)
    _CTX.clear()
    _CTX.update({"backend": backend, "context": context, "args": args})


def evaluate_task(task: Tuple[int, str, str, str]) -> Dict:
    """Evaluate one trajectory. Never raises: a single trajectory's crash (a diverged
    rollout hitting a geometric singularity, an OOM, ...) must not take down the worker
    process and, with it, every other trajectory already completed in the same
    :class:`ProcessPoolExecutor` batch. On failure this returns the same shape a
    successful call would -- ``joint_pos_mae`` is ``FAILED`` per ``core.summary``'s
    validity test, so the trajectory is excluded from averages exactly like any other
    early-terminated run, and the exception is preserved in ``error`` for triage.
    """
    args = _CTX["args"]
    traj_id, path, category, _video_path = task
    try:
        metric = _CTX["backend"].evaluate(_CTX["context"], task)
        if args.videos_only:
            return metric
        validate_reward_scores(metric)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(
            f"[Eval] Trajectory {traj_id} ({path}) failed: {exc!r}\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        return {
            "traj_id": traj_id,
            "file_name": Path(path).name,
            "length_ratio": 0.0,
            "joint_pos_mae": FAILED,
            "joint_vel_mae": FAILED,
            "root_pos_err_mm": FAILED,
            "root_vel_err_mms": FAILED,
            "root_yaw_err": FAILED,
            "kpt_pos_mae": FAILED,
            "kpt_rot_mae": FAILED,
            "score": None,
            "termination_metric": args.termination_metric,
            "category": category,
            "error": repr(exc),
        }
    metric["termination_metric"] = args.termination_metric
    metric["category"] = category
    return metric


def validate_reward_scores(metric: Dict) -> None:
    score = metric.get("score")
    if score is None or not np.isfinite(score):
        raise ValueError(f"Trajectory {metric['traj_id']} produced no finite score")
    frames = metric.get("score_frames")
    if type(frames) is not int or frames < 1:
        raise ValueError(f"Trajectory {metric['traj_id']} has invalid RM frame count")


def print_summaries(backend, all_metrics: List[Dict]) -> None:
    for cat in sorted({m["category"] for m in all_metrics}):
        cat_metrics = [m for m in all_metrics if m["category"] == cat]
        backend.print_category_summary(cat, cat_metrics)
    backend.print_overall_summary(all_metrics)


def save_json(args: argparse.Namespace, backend, all_metrics: List[Dict]) -> Path | None:
    if not args.output_json:
        return None

    cats_found = sorted({m["category"] for m in all_metrics})
    category_summaries = [
        backend.compute_category_summary(cat, [m for m in all_metrics if m["category"] == cat])
        for cat in cats_found
    ]
    overall_summary = backend.compute_overall_summary(all_metrics)

    out_path = repo_path(args.output_json)
    if args.timestamp_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path.with_name(f"{out_path.stem}_{timestamp}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "termination_metric": args.termination_metric,
        "overall_summary": json_safe(overall_summary),
        "category_summaries": json_safe(category_summaries),
        "per_trajectory": json_safe(all_metrics),
    }
    with out_path.open("w") as f:
        json.dump(output_data, f, indent=2)
    return out_path
