"""Interchangeable tracker backends.

The module *is* the interface. The runner never names a tracker: it loads one of the
modules registered below and calls the protocol

    OPTIONS                        command-line flags only this backend needs, as
                                   ``(flag, argparse keyword arguments)`` pairs
    validate(args)                 reject an unusable configuration before any worker
                                   process is spawned
    build_context(args, xml_path)  per-worker state -- policy, reward model, loaders.
                                   The runner passes the result back untouched and only
                                   reads ``reward_model`` from it
    evaluate(context, task)        simulate one trajectory and return its metrics
    compute_category_summary(category, metrics)
    compute_overall_summary(metrics)
    print_category_summary(category, metrics)
    print_overall_summary(metrics)

Adding a tracker is a new module here plus one line in :data:`BACKEND_MODULES`.

Backends whose simulation entry takes the common signature share :func:`load_ref_traj`
and :func:`evaluate_uniform` rather than restating them. ``hgpt`` wraps upstream code
whose entry point differs, so it implements ``evaluate`` itself.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

BACKEND_MODULES = {
    "gmt": "humantracker.eval.backends.gmt",
    "hgpt": "humantracker.eval.backends.hgpt",
    "sonic": "humantracker.eval.backends.sonic",
    "twist2": "humantracker.eval.backends.twist2",
}

BACKEND_NAMES: Tuple[str, ...] = tuple(BACKEND_MODULES)

PROTOCOL = (
    "OPTIONS",
    "validate",
    "build_context",
    "evaluate",
    "compute_category_summary",
    "compute_overall_summary",
    "print_category_summary",
    "print_overall_summary",
)


def load_backend(name: str):
    """Import the backend registered under ``name``, checking it implements the protocol.

    Only the selected backend is imported: they pull in mutually incompatible upstream
    trees, so importing all four would break whichever environment is active.
    """
    module = importlib.import_module(BACKEND_MODULES[name])
    missing = [attr for attr in PROTOCOL if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"Backend {name!r} does not implement the protocol: {', '.join(missing)}"
        )
    return module


def load_ref_traj(file_path: Path) -> Dict:
    """Load a reference motion npz, deriving ``qpos`` from its parts when absent."""
    data = dict(np.load(file_path, allow_pickle=True))
    if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
        data["qpos"] = np.concatenate(
            [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
        )
    if "qpos" not in data:
        raise ValueError(f"{file_path} missing qpos field")
    return data


def evaluate_uniform(context: Dict, task: Tuple[int, str, str, str]) -> Dict:
    """Evaluate one trajectory through a backend's ``simulate`` entry.

    Shared by every backend whose simulation function takes the common signature, so
    that signature is written out once. ``context`` must carry ``simulate``,
    ``load_ref_traj``, ``policy``, ``reward_model``, ``xml_path`` and ``args``.
    """
    traj_id, path, category, video_path = task
    file_path = Path(path)
    args = context["args"]
    return context["simulate"](
        traj_id=traj_id,
        ref_traj=context["load_ref_traj"](file_path),
        file_name=file_path.name,
        policy=context["policy"],
        reward_model=context["reward_model"],
        xml_path=context["xml_path"],
        verbose=False,
        record_video=bool(video_path),
        video_path=video_path or None,
        video_width=args.video_width,
        video_height=args.video_height,
        rollout_dir=args.rollout_dir or None,
        rollout_tracker=args.rollout_tracker or args.tracker,
        rollout_run_id=args.rollout_run_id or None,
        rollout_group=args.rollout_group,
        source_path=path,
        category=category,
        render_only=args.videos_only,
        termination_metric=args.termination_metric,
    )
