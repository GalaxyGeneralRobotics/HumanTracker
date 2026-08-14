"""Fully pre-render a motion_annotation pair pool into the web cache.

The clip stage deduplicates candidate trajectories, shards them across one EGL
process per GPU, and writes the exact cache filenames used by the live server.
After every source clip is ready, the combined stage builds both presentation
orders for every pair.  Existing non-empty outputs are skipped, so the command
is safe to stop and resume.

MuJoCo must be imported only after each worker selects its EGL device.  Keep
``web_labeler.video_cache`` imports inside functions in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from web_labeler.combined_video import ENCODE_ARGS, FILTER_GRAPH, cache_filename


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_XML = (
    REPO_ROOT / "storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml"
)


@dataclass(frozen=True)
class ClipJob:
    key: str
    npz_path: str
    start: int
    end: int
    side_info: dict[str, Any]


@dataclass(frozen=True)
class CombinedJob:
    filename: str
    left_key: str
    right_key: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scene_cache_key(scene_xml: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(scene_xml).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(scene_xml.read_bytes())
    return f"{scene_xml.stem}-{hasher.hexdigest()[:12]}"


def _combined_filename(scene_key: str, left_key: str, right_key: str) -> str:
    return cache_filename(scene_key, left_key, right_key)


def _load_jobs(task_file: Path, scene_key: str) -> tuple[list[ClipJob], list[CombinedJob]]:
    from web_labeler.task_manager import get_frame_range, load_pairs

    # cache_filename imports MuJoCo in the parent only. Child renderers use the
    # spawn start method and import it again after selecting their EGL device.
    from web_labeler.video_cache import cache_filename

    records = load_pairs(task_file)
    clips_by_key: dict[str, ClipJob] = {}
    combined_by_name: dict[str, CombinedJob] = {}

    for record in records:
        candidates = record["candidates"]
        start, end = get_frame_range(record)
        keys: list[str] = []
        for candidate in candidates:
            npz_path = Path(candidate["traj_path"]).resolve()
            key = cache_filename(npz_path, start, end)
            keys.append(key)
            clips_by_key.setdefault(
                key,
                ClipJob(
                    key=key,
                    npz_path=str(npz_path),
                    start=start,
                    end=end,
                    side_info=dict(candidate),
                ),
            )
        for left_key, right_key in (keys, list(reversed(keys))):
            filename = _combined_filename(scene_key, left_key, right_key)
            combined_by_name.setdefault(
                filename,
                CombinedJob(filename, left_key, right_key),
            )

    return list(clips_by_key.values()), list(combined_by_name.values())


def _render_worker(
    worker_id: int,
    gpu_id: int,
    jobs: list[ClipJob],
    cache_dir: str,
    scene_xml: str,
    events: mp.Queue,
) -> None:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)

    # Import after setting MUJOCO_EGL_DEVICE_ID. Each process owns one context.
    from web_labeler.video_cache import get_or_render

    events.put(("worker_started", worker_id, gpu_id, len(jobs)))
    for job in jobs:
        target = Path(cache_dir) / job.key
        existed = target.is_file() and target.stat().st_size > 0
        started = time.monotonic()
        events.put(("job_started", worker_id, job.key))
        try:
            get_or_render(
                job.npz_path,
                job.start,
                job.end,
                cache_dir,
                scene_xml,
                side_info=job.side_info,
            )
            events.put(
                (
                    "job_done",
                    worker_id,
                    job.key,
                    time.monotonic() - started,
                    existed,
                )
            )
        except Exception as exc:  # keep the shard moving after a bad clip
            events.put(("job_failed", worker_id, job.key, repr(exc)))
    events.put(("worker_done", worker_id))


def _write_progress(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_clip_stage(
    jobs: list[ClipJob],
    cache_dir: Path,
    scene_xml: Path,
    gpu_ids: list[int],
    progress_path: Path,
    state: dict[str, Any],
) -> int:
    ready_at_start = sum(
        1 for job in jobs
        if (cache_dir / job.key).is_file() and (cache_dir / job.key).stat().st_size > 0
    )
    pending = [
        job for job in jobs
        if not (cache_dir / job.key).is_file() or (cache_dir / job.key).stat().st_size <= 0
    ]
    state["clip_stage"] = {
        "status": "running" if pending else "complete",
        "total": len(jobs),
        "ready": ready_at_start,
        "ready_at_start": ready_at_start,
        "rendered_this_run": 0,
        "failed": 0,
        "failures": {},
        "workers": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "active": {},
    }
    _write_progress(progress_path, state)
    if not pending:
        return 0

    ctx = mp.get_context("spawn")
    events: mp.Queue = ctx.Queue()
    shards = [pending[index::len(gpu_ids)] for index in range(len(gpu_ids))]
    processes = [
        ctx.Process(
            target=_render_worker,
            args=(worker_id, gpu_id, shards[worker_id], str(cache_dir), str(scene_xml), events),
            name=f"robotvis-render-gpu{gpu_id}",
        )
        for worker_id, gpu_id in enumerate(gpu_ids)
    ]
    for process in processes:
        process.start()

    completed_workers = 0
    last_write = 0.0
    clip_state = state["clip_stage"]
    while completed_workers < len(processes):
        try:
            event = events.get(timeout=2.0)
        except queue.Empty:
            event = None

        if event:
            kind = event[0]
            if kind == "worker_started":
                _, worker_id, gpu_id, count = event
                print(f"[clip-worker] worker={worker_id} gpu={gpu_id} jobs={count}", flush=True)
            elif kind == "job_started":
                _, worker_id, key = event
                clip_state["active"][str(worker_id)] = key
            elif kind == "job_done":
                _, worker_id, key, elapsed, existed = event
                clip_state["active"].pop(str(worker_id), None)
                if not existed:
                    clip_state["rendered_this_run"] += 1
                clip_state["ready"] += 1
                print(
                    f"[clip] {clip_state['ready']}/{len(jobs)} worker={worker_id} "
                    f"{key} ({elapsed:.2f}s)",
                    flush=True,
                )
            elif kind == "job_failed":
                _, worker_id, key, error = event
                clip_state["active"].pop(str(worker_id), None)
                clip_state["failed"] += 1
                clip_state["failures"][key] = error
                print(f"[clip] FAILED worker={worker_id} {key}: {error}", flush=True)
            elif kind == "worker_done":
                completed_workers += 1

        now = time.monotonic()
        if now - last_write >= 2.0:
            _write_progress(progress_path, state)
            last_write = now

    for process in processes:
        process.join()
        if process.exitcode:
            clip_state["failed"] += 1
            clip_state["failures"][f"worker_{process.name}"] = f"exitcode={process.exitcode}"

    clip_state["active"] = {}
    clip_state["status"] = "complete" if clip_state["failed"] == 0 else "complete_with_errors"
    _write_progress(progress_path, state)
    return int(clip_state["failed"])


def _build_combined(
    job: CombinedJob,
    cache_dir: Path,
    combined_dir: Path,
    ffmpeg: str,
) -> tuple[str, float, bool]:
    out = combined_dir / job.filename
    if out.is_file() and out.stat().st_size > 0:
        return job.filename, 0.0, True

    left_src = cache_dir / job.left_key
    right_src = cache_dir / job.right_key
    if not left_src.is_file() or not right_src.is_file():
        raise FileNotFoundError(f"Missing source for {job.filename}")

    tmp = out.with_name(
        f".{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
    )
    tmp.unlink(missing_ok=True)
    command = [
        ffmpeg, "-y",
        "-i", str(left_src),
        "-i", str(right_src),
        "-filter_complex",
        FILTER_GRAPH,
        "-map", "[v]",
        "-an",
        *ENCODE_ARGS,
        str(tmp),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join((result.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg rc={result.returncode}: {tail}")
    tmp.replace(out)
    return job.filename, time.monotonic() - started, False


def _run_combined_stage(
    jobs: list[CombinedJob],
    cache_dir: Path,
    combined_dir: Path,
    workers: int,
    progress_path: Path,
    state: dict[str, Any],
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    combined_dir.mkdir(parents=True, exist_ok=True)
    ready_at_start = sum(
        1 for job in jobs
        if (combined_dir / job.filename).is_file()
        and (combined_dir / job.filename).stat().st_size > 0
    )
    pending = [
        job for job in jobs
        if not (combined_dir / job.filename).is_file()
        or (combined_dir / job.filename).stat().st_size <= 0
    ]
    state["combined_stage"] = {
        "status": "running" if pending else "complete",
        "total": len(jobs),
        "ready": ready_at_start,
        "ready_at_start": ready_at_start,
        "built_this_run": 0,
        "failed": 0,
        "failures": {},
        "workers": workers,
    }
    _write_progress(progress_path, state)
    if not pending:
        return 0

    combined_state = state["combined_stage"]
    last_write = 0.0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="combined") as pool:
        futures = {
            pool.submit(_build_combined, job, cache_dir, combined_dir, ffmpeg): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                filename, elapsed, existed = future.result()
                combined_state["ready"] += 1
                if not existed:
                    combined_state["built_this_run"] += 1
                print(
                    f"[combined] {combined_state['ready']}/{len(jobs)} "
                    f"{filename} ({elapsed:.2f}s)",
                    flush=True,
                )
            except Exception as exc:
                combined_state["failed"] += 1
                combined_state["failures"][job.filename] = repr(exc)
                print(f"[combined] FAILED {job.filename}: {exc!r}", flush=True)

            now = time.monotonic()
            if now - last_write >= 2.0:
                _write_progress(progress_path, state)
                last_write = now

    combined_state["status"] = (
        "complete" if combined_state["failed"] == 0 else "complete_with_errors"
    )
    _write_progress(progress_path, state)
    return int(combined_state["failed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-render every source and combined video used by motion_annotation."
    )
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--cache-root", type=Path, default=Path(".video_cache"))
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated EGL device ids; one render process is created per id.",
    )
    parser.add_argument("--combine-workers", type=int, default=8)
    parser.add_argument("--clips-only", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print deduplicated counts without rendering.",
    )
    parser.add_argument("--progress-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_file = args.task_file.expanduser().resolve()
    scene_xml = args.scene_xml.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    if not task_file.is_file():
        raise FileNotFoundError(task_file)
    if not scene_xml.is_file():
        raise FileNotFoundError(scene_xml)

    scene_key = _scene_cache_key(scene_xml)
    cache_dir = cache_root / "scenes" / scene_key
    combined_dir = cache_dir.parent / f"{cache_dir.name}_combined"
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        args.progress_file.expanduser().resolve()
        if args.progress_file
        else cache_dir.parent / f"{cache_dir.name}_prerender_progress.json"
    )

    clips, combined = _load_jobs(task_file, scene_key)
    state: dict[str, Any] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "task_file": str(task_file),
        "scene_xml": str(scene_xml),
        "scene_cache_key": scene_key,
        "cache_dir": str(cache_dir),
        "combined_cache_dir": str(combined_dir),
        "progress_file": str(progress_path),
    }
    print(
        f"[config] clips={len(clips)} combined={len(combined)} "
        f"gpus={gpu_ids} cache={cache_dir}",
        flush=True,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "task_file": str(task_file),
                    "scene_xml": str(scene_xml),
                    "scene_cache_key": scene_key,
                    "unique_clips": len(clips),
                    "combined_videos": len(combined),
                    "gpu_ids": gpu_ids,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    _write_progress(progress_path, state)

    clip_failures = _run_clip_stage(
        clips, cache_dir, scene_xml, gpu_ids, progress_path, state
    )
    combined_failures = 0
    if not args.clips_only:
        combined_failures = _run_combined_stage(
            combined,
            cache_dir,
            combined_dir,
            max(1, int(args.combine_workers)),
            progress_path,
            state,
        )

    state["status"] = (
        "complete" if clip_failures == 0 and combined_failures == 0
        else "complete_with_errors"
    )
    state["finished_at_utc"] = _utc_now()
    _write_progress(progress_path, state)
    print(
        f"[done] status={state['status']} clip_failures={clip_failures} "
        f"combined_failures={combined_failures}",
        flush=True,
    )
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
