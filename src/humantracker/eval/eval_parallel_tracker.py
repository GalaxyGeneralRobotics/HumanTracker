"""Unified parallel evaluation entry for tracker policies.

This file owns the command line, the worker pool and progress reporting. Dataset
validation and result output live in :mod:`humantracker.eval.runner`; everything
policy-specific lives in a backend module under ``backends/``, which also declares its
own command-line flags.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional, Sequence, Tuple

from tqdm import tqdm

from humantracker.eval.backends import BACKEND_NAMES, load_backend
from humantracker.eval.core.termination_metrics import TERMINATION_METRICS
from humantracker.eval.paths import ROOT, required_file
from humantracker.eval.runner import (
    DEFAULT_SCENE_XML,
    evaluate_task,
    init_worker,
    load_traj_files,
    make_tasks,
    print_summaries,
    save_json,
)


def build_parser(
    backend_options: Sequence[Tuple[str, Dict]] = (),
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified parallel eval for tracker policies")
    parser.add_argument("--tracker", required=True, choices=BACKEND_NAMES)
    parser.add_argument("--mocap_path", required=True,
                        help="motion dataset root; relative paths resolve against the repository")
    parser.add_argument("--test_json", required=True,
                        help="evaluation manifest listing {path, category} entries")
    parser.add_argument(
        "--rm_checkpoint",
        default="storage/checkpoints/reward_model/best.pt",
        help="HumanScore checkpoint; defaults to where the released weights unpack",
    )
    parser.add_argument("--rm_device", default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--termination_metric",
        choices=TERMINATION_METRICS,
        required=True,
        help="termination rule used to score trajectory completion; "
             "'whole_body' applies SONIC's published terms to every tracker",
    )
    parser.add_argument(
        "--xml_path",
        default=DEFAULT_SCENE_XML,
        help="MuJoCo scene XML; the default is the paper-gray scene every tracker is measured in",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--multiprocessing_start_method", default="spawn")
    parser.add_argument("--max_trajs", type=int, default=0)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--skip_flipped", action="store_true")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--timestamp_output", action="store_true")
    parser.add_argument("--video_dir", default="eval_videos")
    parser.add_argument("--video_interval", type=int, default=0)
    parser.add_argument("--video_width", type=int, default=3840)
    parser.add_argument("--video_height", type=int, default=2160)
    parser.add_argument("--videos_only", action="store_true")
    parser.add_argument("--rollout_dir", default="")
    parser.add_argument("--rollout_tracker", default="",
                        help="tracker name recorded in exported rollouts; defaults to --tracker")
    parser.add_argument("--rollout_run_id", default="")
    parser.add_argument("--rollout_group", default="rlhf_rollout")

    for flag, options in backend_options:
        parser.add_argument(flag, **options)
    return parser


def selected_tracker() -> Optional[str]:
    """Read ``--tracker`` on its own, before the full command line is defined.

    Each backend declares its own flags, so the parser cannot be built until the tracker
    is known; and only the selected backend may be imported, because they pull in
    mutually incompatible upstream trees. Returns ``None`` when ``--tracker`` is absent,
    leaving the full parser to report that and to print help.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--tracker", choices=BACKEND_NAMES)
    return probe.parse_known_args()[0].tracker


def configure_rendering() -> None:
    """Select headless EGL before a backend, and with it mujoco, is imported.

    ``import mujoco`` creates its GL context eagerly, so these have to be in the
    environment by the time the backend module loads. Setting them afterwards leaves EGL
    searching the system vendor directories, where the ICD file this repository ships in
    ``egl_conf/`` is not found, and the import fails on a headless machine.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", str(ROOT / "egl_conf"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> None:
    configure_rendering()
    tracker = selected_tracker()
    backend_options = load_backend(tracker).OPTIONS if tracker else ()
    args = build_parser(backend_options).parse_args()

    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.max_trajs < 0:
        raise ValueError(f"--max_trajs must be non-negative, got {args.max_trajs}")
    if args.video_interval < 0:
        raise ValueError(f"--video_interval must be non-negative, got {args.video_interval}")
    if args.video_width < 1 or args.video_height < 1:
        raise ValueError("Video dimensions must be positive")
    if args.videos_only and args.video_interval < 1:
        raise ValueError("--videos_only requires a positive --video_interval")
    if args.videos_only and args.output_json:
        raise ValueError("--videos_only does not write --output_json")
    required_file(args.rm_checkpoint)

    backend = load_backend(args.tracker)
    backend.validate(args)

    data_path, test_json, traj_files, traj_categories = load_traj_files(args)
    tasks = make_tasks(args, traj_files, traj_categories)
    if args.videos_only:
        tasks = [task for task in tasks if task[3]]

    print(
        f"[Eval] tracker={args.tracker} workers={args.workers} "
        f"device={args.device} rm_device={args.rm_device} "
        f"termination_metric={args.termination_metric}"
    )
    print(f"[Eval] data_path={data_path}")
    print(f"[Eval] test_json={test_json}")
    print(f"[Eval] output_json={args.output_json}")

    args_dict = vars(args).copy()
    max_workers = int(args.workers)
    metrics_by_id: Dict[int, Dict] = {}

    if max_workers == 1:
        init_worker(args.tracker, args_dict)
        for task in tqdm(tasks, total=len(tasks), desc=f"{args.tracker} Eval"):
            metric = evaluate_task(task)
            metrics_by_id[metric["traj_id"]] = metric
    else:
        mp_context = mp.get_context(args.multiprocessing_start_method)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=init_worker,
            initargs=(args.tracker, args_dict),
        ) as executor:
            futures = {executor.submit(evaluate_task, task): task[0] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.tracker} Eval"):
                metric = future.result()
                metrics_by_id[metric["traj_id"]] = metric

    all_metrics = [metrics_by_id[task[0]] for task in tasks]
    if args.videos_only:
        print(f"[Eval] Rendered {len(all_metrics)} videos")
        return
    print_summaries(backend, all_metrics)
    out_path = save_json(args, backend, all_metrics)
    if out_path:
        print(f"[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
