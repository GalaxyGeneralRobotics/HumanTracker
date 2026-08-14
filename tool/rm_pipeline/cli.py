from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .annotations import aggregate_annotations, export_rm_parquet, validate_pair_record


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def make_run_id(value: str = "auto") -> str:
    if not value or value.lower() in {"auto", "now"}:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    if not RUN_ID_RE.match(value):
        raise ValueError("--run-id must be YYYYMMDD_HHMMSS, or use auto")
    return value


def latest_jsonl(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    if not root.exists():
        return None
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_jsonl_input(raw: str, kind: str, pipeline_root: Path, patterns: Sequence[str]) -> Path:
    if not raw or raw.lower() in {"auto", "latest"}:
        latest = latest_jsonl(pipeline_root, patterns)
        if latest is None:
            raise FileNotFoundError(f"{kind} latest requested, but no matching JSONL was found under {pipeline_root}")
        print(f"[{kind}] using latest: {latest}")
        return latest

    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(parts: Sequence[Any], length: int = 12) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def safe_name(value: str, max_len: int = 180) -> str:
    text = str(value).replace(".npz", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.+=@-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        text = "item"
    if len(text) > max_len:
        text = f"{text[: max_len - 11]}_{stable_hash([text], 10)}"
    return text


def load_meta_for_npz(npz_path: Path) -> Dict[str, Any]:
    meta_path = npz_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "motion_id",
        "tracker",
        "category",
        "source_motion_path",
        "rollout_npz_path",
        "log_folder",
        "fps",
        "num_frames",
        "source_start_frame",
        "source_end_frame",
    }
    missing = sorted(required - meta.keys())
    if missing:
        raise KeyError(f"{meta_path}: missing fields {missing}")
    if Path(meta["rollout_npz_path"]).resolve() != npz_path.resolve():
        raise ValueError(f"{meta_path}: rollout_npz_path does not match {npz_path}")
    if not Path(meta["source_motion_path"]).is_file():
        raise FileNotFoundError(meta["source_motion_path"])
    return meta


def load_frame_arrays(npz_path: Path) -> tuple[int, Dict[str, np.ndarray]]:
    required = ("ref_pose", "ref_joint_pos", "imu_pose", "joint_pos")
    with np.load(npz_path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{npz_path}: missing frame arrays {missing}")
        lengths = {int(data[key].shape[0]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"{npz_path}: required frame arrays have different lengths")
        frame_count = lengths.pop()
        arrays: Dict[str, np.ndarray] = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == frame_count:
                arrays[key] = value
    return frame_count, arrays


def iter_rollout_npzs(root: Path, manifest: Optional[Path] = None) -> List[Dict[str, Any]]:
    if manifest is not None:
        entries = read_jsonl(manifest)
        for line_no, entry in enumerate(entries, 1):
            if "rollout_npz_path" not in entry:
                raise KeyError(f"{manifest}:{line_no}: rollout_npz_path is required")
            path = Path(entry["rollout_npz_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
        return entries

    paths = sorted(root.glob("*/traj_csv/*.npz"))
    return [load_meta_for_npz(path) for path in paths]


def clip_rollout(
    entry: Dict[str, Any],
    clips_root: Path,
    run_id: str,
    clip_seconds: float,
    keep_tail: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    npz_path = Path(entry["rollout_npz_path"])
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    meta = load_meta_for_npz(npz_path)
    tracker = str(meta["tracker"])
    motion_id = str(meta["motion_id"])
    fps = int(meta["fps"])
    clip_frames = int(round(fps * clip_seconds))
    if clip_frames <= 0:
        raise ValueError("clip length must be positive")
    n, frame_arrays = load_frame_arrays(npz_path)
    if n != int(meta["num_frames"]):
        raise ValueError(f"{npz_path}: num_frames does not match metadata")
    source_start = int(meta["source_start_frame"])
    if int(meta["source_end_frame"]) - source_start != n:
        raise ValueError(f"{npz_path}: source frame range does not match num_frames")
    log_folder = str(meta["log_folder"])
    out_dir = clips_root / log_folder / "traj_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_chunks = n // clip_frames
    chunk_count = full_chunks + (1 if keep_tail and n % clip_frames else 0)
    for clip_index in range(chunk_count):
        local_start = clip_index * clip_frames
        local_end = min(n, local_start + clip_frames)
        if local_end - local_start < clip_frames and not keep_tail:
            continue
        global_start = source_start + local_start
        global_end = source_start + local_end
        clip_id = f"{motion_id}__clip_{global_start:06d}_{global_end:06d}"
        clip_path = out_dir / f"{run_id}_{safe_name(clip_id)}.npz"
        if clip_path.exists() or clip_path.with_suffix(".meta.json").exists():
            raise FileExistsError(clip_path)
        np.savez_compressed(
            clip_path,
            **{
                key: value[local_start:local_end]
                for key, value in frame_arrays.items()
            },
        )
        clip_meta = {
            "clip_id": clip_id,
            "motion_id": motion_id,
            "tracker": tracker,
            "category": meta["category"],
            "traj_path": str(clip_path),
            "clip_npz_path": str(clip_path),
            "source_rollout_path": str(npz_path),
            "source_motion_path": meta["source_motion_path"],
            "log_folder": log_folder,
            "fps": fps,
            "num_frames": int(local_end - local_start),
            "duration_sec": float((local_end - local_start) / fps),
            "source_start_frame": int(global_start),
            "source_end_frame": int(global_end),
            "local_start_frame": 0,
            "local_end_frame": int(local_end - local_start),
            "clip_index": int(clip_index),
            "drop_tail": not keep_tail,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with clip_path.with_suffix(".meta.json").open("x", encoding="utf-8") as f:
            json.dump(clip_meta, f, indent=2, ensure_ascii=False)
        rows.append(clip_meta)
    return rows


def command_clip_rollouts(args: argparse.Namespace) -> None:
    rollout_root = Path(args.rollout_root)
    run_id = make_run_id(args.run_id)
    pipeline_root = Path(args.pipeline_root)
    clips_root = Path(args.clips_root) if args.clips_root else pipeline_root / run_id / "clips"
    manifest = resolve_jsonl_input(
        args.manifest,
        "rollout manifest",
        pipeline_root,
        ("**/rollouts.jsonl",),
    ) if args.manifest else None
    entries = iter_rollout_npzs(rollout_root, manifest)
    if not entries:
        raise ValueError(f"no rollout npz files found under {rollout_root}")

    worker = partial(
        clip_rollout,
        clips_root=clips_root,
        run_id=run_id,
        clip_seconds=float(args.clip_seconds),
        keep_tail=bool(args.keep_tail),
    )
    out_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        for rows in executor.map(worker, entries):
            out_rows.extend(rows)

    manifest_path = Path(args.output_manifest) if args.output_manifest else clips_root / "clips.jsonl"
    write_jsonl(manifest_path, out_rows)
    print(f"[clip-rollouts] run_id={run_id}")
    print(f"[clip-rollouts] wrote {len(out_rows)} clips -> {manifest_path}")


def command_aggregate(args: argparse.Namespace) -> None:
    aggregates = aggregate_annotations(
        args.inputs,
        Path(args.output),
        int(args.min_annotations),
        int(args.min_agreement),
    )
    print(f"[aggregate] wrote {len(aggregates)} aggregates -> {args.output}")


def command_export_rm_parquet(args: argparse.Namespace) -> None:
    count = export_rm_parquet(Path(args.aggregates), Path(args.output))
    print(f"[export-rm-parquet] wrote {count} rows -> {args.output}")


def command_validate_pairs(args: argparse.Namespace) -> None:
    pairs = read_jsonl(Path(args.pairs))
    for pair in pairs:
        validate_pair_record(pair)
    print(f"[validate-pairs] ok: {len(pairs)} pairs")


def command_summarize_tool_format(_: argparse.Namespace) -> None:
    summary = {
        "tool_npz_required_arrays": {
            "ref_pose": "(T, 7), reference root pose [x, y, z, qw, qx, qy, qz]",
            "ref_joint_pos": "(T, 29), reference G1 joint positions",
            "imu_pose": "(T, 7), tracker rollout root pose",
            "joint_pos": "(T, 29), tracker rollout joint positions",
        },
        "tool_npz_optional_arrays": ["qpos", "qvel", "ref_joint_vel", "joint_vel", "ref_foot_contact", "foot_contact", "motor_target"],
        "tool_pair_record": "Use the canonical preference-pair JSONL with two indexed candidates.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tool.rm_pipeline",
        description="HumanTracker preference-data pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clip-rollouts", help="Slice full rollout NPZ files into fixed-length clips")
    p.add_argument("--rollout-root", required=True)
    p.add_argument("--clips-root", default="", help="Output clip directory. Defaults to <pipeline-root>/<run-id>/clips.")
    p.add_argument("--pipeline-root", default="storage/preference_pair/preference_pipeline")
    p.add_argument("--run-id", default="auto", help="YYYYMMDD_HHMMSS. Defaults to current system time.")
    p.add_argument("--manifest", default="")
    p.add_argument("--output-manifest", default="")
    p.add_argument("--clip-seconds", type=float, default=5.0)
    p.add_argument("--keep-tail", action="store_true", help="Keep final shorter clip; default drops it")
    p.add_argument("--workers", type=int, default=32)
    p.set_defaults(func=command_clip_rollouts)


    p = sub.add_parser("aggregate", help="Aggregate annotation parquet/jsonl records by majority vote")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-annotations", type=int, default=3)
    p.add_argument("--min-agreement", type=int, default=2)
    p.set_defaults(func=command_aggregate)

    p = sub.add_parser("export-rm-parquet", help="Export valid aggregate pairs as RM training parquet")
    p.add_argument("--aggregates", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=command_export_rm_parquet)

    p = sub.add_parser("validate-pairs", help="Validate pair manifest paths and alignment")
    p.add_argument("--pairs", required=True)
    p.set_defaults(func=command_validate_pairs)

    p = sub.add_parser("summarize-tool-format", help="Print the motion_annotation data format summary")
    p.set_defaults(func=command_summarize_tool_format)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
