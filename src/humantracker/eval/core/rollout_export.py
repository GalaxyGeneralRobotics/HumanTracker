"""Utilities for exporting tracker rollouts in motion_annotation tool format.

The motion_annotation preference UI expects each compared tracker to live in a
``<log_folder>/traj_csv/*.npz`` tree.  Files from different trackers are matched
by their stem after removing the leading ``YYYYMMDD_HHMMSS_`` prefix.

This module keeps the exported NPZ frame-only.  Scalar metadata is written to a
sidecar ``.meta.json`` so existing UI code that slices every NPZ value by frame
does not trip over zero-dimensional arrays.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from humantracker.reward_model.features import FEATURE_HEADS


DEFAULT_FPS = 50
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value: str, max_len: int = 180) -> str:
    """Return a filesystem-safe, stable name while keeping it human-readable."""
    text = str(value).replace(".npz", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        text = "motion"
    if len(text) > max_len:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        text = f"{text[: max_len - 11]}_{digest}"
    return text


def infer_motion_id(
    source_path: str | os.PathLike[str],
    category: Optional[str],
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return safe_name(explicit)
    if not category:
        raise ValueError("category is required when motion_id is not explicit")
    path = Path(str(source_path))
    if not path.stem:
        raise ValueError(f"source_path has no motion stem: {source_path}")
    return safe_name(f"{category}_{path.stem}")


def make_tool_log_folder(
    tracker_name: str,
    *,
    run_id: Optional[str] = None,
    group: str = "rlhf_rollout",
) -> str:
    """Create a log folder name compatible with ``npz_data_loader`` parsing."""
    rid = run_id or os.environ.get("HUMANTRACKER_ROLLOUT_RUN_ID") or _RUN_TIMESTAMP
    rid = safe_name(rid)
    if not re.match(r"^\d{8}_\d{6}$", rid):
        rid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rid}"
    tracker = safe_name(tracker_name)
    group_name = safe_name(group)
    return f"{rid}_{tracker}_['{group_name}']"


def _stack_state_history(state_history: Sequence[Dict[str, np.ndarray]], key: str) -> Optional[np.ndarray]:
    values = [s.get(key) for s in state_history if isinstance(s, dict) and key in s]
    if not values:
        return None
    return np.asarray(values)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _stack_feature_history(
    feature_history: Sequence[Dict[str, np.ndarray]], length: int
) -> Dict[str, np.ndarray]:
    """Stack a list of per-frame field dicts into (length, dim) float32 arrays.

    ``foot_acc`` is computed as the real temporal derivative of ``foot_vel``
    (np.gradient uses genuine one-sided differences at the endpoints, so there
    is no zero-padding). All other fields must be supplied by the extractor.
    """
    stacked: Dict[str, np.ndarray] = {}
    for key, dim in FEATURE_HEADS.items():
        if key == "foot_acc":
            continue
        frames = [np.asarray(f[key], dtype=np.float32).reshape(-1) for f in feature_history[:length]]
        arr = np.stack(frames, axis=0)
        if arr.shape[1] != dim:
            raise ValueError(f"feature '{key}' dim {arr.shape[1]} != expected {dim}")
        stacked[key] = arr
    return stacked


def _first_nonfinite_frame(
    rollout_qpos: np.ndarray, rollout_qvel: np.ndarray, feats: Dict[str, np.ndarray]
) -> Optional[int]:
    """The earliest frame at which ``qpos``, ``qvel``, or any stacked feature goes
    non-finite, or ``None`` if every frame is finite.

    A diverged rollout can go non-finite mid-sequence: MuJoCo's own autoreset absorbs a
    divergence inside ``mj_step`` (it silently resets ``mj_data`` to ``qpos0``, so
    ``qpos``/``qvel`` themselves are rarely the carrier -- see ``core/geometry.py``'s
    ``batch_base2navi``), but the raw policy output recorded into the per-frame features
    is captured *before* that scrubbing and does not benefit from it (see
    ``core/rm_scorer.py``). Call this before deriving ``foot_acc``: its gradient would
    smear a NaN backward into still-finite frames via the central-difference stencil.
    """
    finite = np.isfinite(rollout_qpos).all(axis=-1) & np.isfinite(rollout_qvel).all(axis=-1)
    for arr in feats.values():
        finite &= np.isfinite(arr).all(axis=-1)
    if finite.all():
        return None
    return int(np.argmax(~finite))


def save_tool_rollout_npz(
    *,
    output_root: str | os.PathLike[str],
    tracker_name: str,
    source_path: str | os.PathLike[str],
    ref_traj: Dict[str, Any],
    state_history: Sequence[Dict[str, np.ndarray]],
    feature_history: Sequence[Dict[str, np.ndarray]],
    metrics: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    fps: int = DEFAULT_FPS,
    ref_start_index: int = 0,
    run_id: Optional[str] = None,
    group: str = "rlhf_rollout",
    motion_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Save a full rollout as a motion_annotation-compatible NPZ plus metadata.

    Every field required by the reward-model dataset is written from the real
    per-frame features in ``feature_history`` (see
    ``eval/rm_feature_extractor.py``).  No zero-padding or fallback values are
    produced; missing inputs raise instead.
    """
    if not state_history:
        raise ValueError("state_history is empty; cannot export rollout")
    if not feature_history:
        raise ValueError(
            "feature_history is empty; real per-frame RM features are required "
            "(no fallback). Use eval.rm_feature_extractor.extract_frame_fields."
        )
    if "qpos" not in ref_traj:
        raise KeyError("ref_traj must contain qpos")

    rollout_qpos = _stack_state_history(state_history, "qpos")
    rollout_qvel = _stack_state_history(state_history, "qvel")
    if rollout_qpos is None:
        raise KeyError("state_history items must contain qpos")
    if rollout_qvel is None:
        raise KeyError("state_history items must contain real qvel (no finite-difference fallback)")

    length = min(len(rollout_qpos), len(feature_history))
    if length <= 0:
        raise ValueError("no frames to export")

    rollout_qpos = rollout_qpos[:length].astype(np.float32, copy=False)
    rollout_qvel = rollout_qvel[:length].astype(np.float32, copy=False)

    # All RM fields from real per-frame features.
    feats = _stack_feature_history(feature_history, length)

    # A diverged rollout must not export non-finite frames into training data: truncate
    # to the finite prefix before deriving foot_acc, whose central-difference stencil
    # would otherwise smear a NaN backward into frames that were themselves still finite.
    first_bad = _first_nonfinite_frame(rollout_qpos, rollout_qvel, feats)
    if first_bad is not None:
        if first_bad == 0:
            raise ValueError(
                f"{source_path}: first recorded frame is already non-finite; "
                "no finite prefix to export"
            )
        length = first_bad
        rollout_qpos = rollout_qpos[:length]
        rollout_qvel = rollout_qvel[:length]
        feats = {key: arr[:length] for key, arr in feats.items()}

    # foot_acc: real temporal derivative of foot_vel (one-sided at the ends).
    dt = 1.0 / float(fps)
    feats["foot_acc"] = np.gradient(feats["foot_vel"], dt, axis=0).astype(np.float32)

    mid = infer_motion_id(source_path, category, motion_id)
    log_folder = make_tool_log_folder(tracker_name, run_id=run_id, group=group)
    output_dir = Path(output_root) / log_folder / "traj_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = (run_id or os.environ.get("HUMANTRACKER_ROLLOUT_RUN_ID") or _RUN_TIMESTAMP)
    file_prefix = safe_name(file_prefix)
    if not re.match(r"^\d{8}_\d{6}$", file_prefix):
        file_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = output_dir / f"{file_prefix}_{mid}.npz"

    # qpos/qvel keep the full simulator state for the labeling-tool renderer.
    save_arrays = {key: feats[key] for key in FEATURE_HEADS}
    save_arrays["qpos"] = rollout_qpos
    save_arrays["qvel"] = rollout_qvel
    np.savez_compressed(npz_path, **save_arrays)

    meta = {
        "motion_id": mid,
        "tracker": str(tracker_name),
        "category": category,
        "source_motion_path": str(source_path),
        "rollout_npz_path": str(npz_path),
        "rollout_meta_path": str(npz_path.with_suffix(".meta.json")),
        "log_folder": log_folder,
        "npz_name": npz_path.name,
        "fps": int(fps),
        "num_frames": int(length),
        "duration_sec": float(length / float(fps)),
        "source_start_frame": int(ref_start_index),
        "source_end_frame": int(ref_start_index + length),
        "qpos_dim": int(rollout_qpos.shape[1]),
        "ref_qpos_dim": int(np.asarray(ref_traj["qpos"]).shape[1]),
        "fields": list(save_arrays.keys()),
        "metrics": _json_safe(metrics or {}),
        "extra": _json_safe(extra or {}),
        "truncated_nonfinite_at": first_bad,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = npz_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    manifest_path = Path(output_root) / "rollouts.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return npz_path, meta_path
