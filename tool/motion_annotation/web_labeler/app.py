"""FastAPI web UI for pairwise robot motion labeling.

Serves pre-rendered MP4 videos directly via static files for instant playback.
Uses native HTML5 <video> with synchronized playback.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import queue
import shutil
import threading
import time
import subprocess
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from web_labeler.combined_video import (
    ENCODE_ARGS,
    FILTER_GRAPH,
    cache_filename as combined_filename,
)
from web_labeler.video_cache import cache_filename
from web_labeler.render_manager import RenderManager, RenderJob
from web_labeler.task_manager import DynamicTaskManager, UNDO_LIMIT
from utils.hf_proto import (
    HFRecord, HFMeta, HFPreference, HFVideoSide, HFFlags, HFComparison,
)
from utils.data_collector.hf_recorder import HFRecorder

REPO_ROOT = Path(__file__).resolve().parents[3]
PREFERENCE_ROOT = REPO_ROOT / "storage" / "preference_pair"
PIPELINE_ROOT = PREFERENCE_ROOT / "preference_pipeline"
DEFAULT_SCENE_XML = REPO_ROOT / "storage" / "assets" / "unitree_g1_5010" / "scene_mjx_track_papergray.xml"


def _find_latest_file(parent: Path, patterns: tuple[str, ...]) -> Path:
    candidates = []
    for pattern in patterns:
        candidates.extend(parent.glob(pattern))
    candidates = sorted([p for p in candidates if p.is_file()], key=lambda p: str(p))
    if not candidates:
        raise FileNotFoundError(f"No files found in {parent} matching {patterns}")
    return candidates[-1]


def resolve_task_file(task_file: str) -> Path:
    if task_file == "latest":
        path = _find_latest_file(PIPELINE_ROOT, ("**/pairs.jsonl", "**/pairs_*.jsonl"))
        print(f"[auto-resolve] pairs-file -> {path}")
        return path
    path = Path(task_file)
    path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_hf_logs_dir(hf_logs_dir: str, task_path: Path) -> Path:
    if hf_logs_dir != "auto":
        path = Path(hf_logs_dir)
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    run_id = task_path.parent.name
    if task_path.stem.startswith("pairs_"):
        run_id = f"{run_id}_{task_path.stem.removeprefix('pairs_')}"
    output = PREFERENCE_ROOT / f"hf_preference_{run_id}_dynamic"
    print(f"[auto-resolve] hf-logs-dir -> {output}")
    return output


def _safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value.strip()) or "default"


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{name} is required")
    return value.strip()


def _lease_call(operation, *args):
    try:
        return operation(*args)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def build_record(
    task_mgr: DynamicTaskManager,
    idx: int,
    label: str,
    annotator_id: str,
    client_id: str,
    lease_id: str,
) -> HFRecord:
    left_info, right_info, record, display_order = task_mgr.get_pair_for_lease(
        lease_id, annotator_id, client_id
    )
    if int(record["pair_idx"]) != idx:
        raise ValueError(f"lease pair index {record['pair_idx']} does not match request {idx}")
    start, end = task_mgr.get_frame_range(record)
    preferred_candidate_idx = task_mgr.resolve_preferred_candidate_idx(label, display_order)
    choice_type = "preference" if label in ("left", "right") else label
    display_choice_idx = 0 if label == "left" else 1 if label == "right" else None

    ts = time.strftime("%Y%m%d_%H%M%S")
    pair_id = str(record["pair_id"])
    record_id = f"{pair_id}__{_safe_id(annotator_id)}__{lease_id}"

    meta = HFMeta(
        record_id=record_id, annotator_id=annotator_id, timestamp=ts,
        tool="web_labeler", task="G1-TrackMJ", scene="flat_ground",
        fps=int(record["fps"]),
    )
    left_npz = task_mgr.get_npz_path(left_info)
    right_npz = task_mgr.get_npz_path(right_info)


    def side_extra(side_info, npz_path):
        return {
            "traj_path": str(npz_path),
            "pair_id": record["pair_id"],
            "motion_id": record["motion_id"],
            "clip_uid": record["clip_uid"],
            "source_start_frame": record["source_start_frame"],
            "source_end_frame": record["source_end_frame"],
            "fps": record["fps"],
            "tracker": side_info["tracker"],
            "candidate_idx": side_info["candidate_idx"],
            "candidate_uid": side_info["candidate_uid"],
        }

    video_left = HFVideoSide(
        npz_name=left_npz.name, start_frame=start, end_frame=end,
        policy=left_info["tracker"],
        extra=side_extra(left_info, left_npz),
    )
    video_right = HFVideoSide(
        npz_name=right_npz.name, start_frame=start, end_frame=end,
        policy=right_info["tracker"],
        extra=side_extra(right_info, right_npz),
    )
    preference = HFPreference(winner=label, confidence=1.0)
    flags = HFFlags(invalid=(label == "bad_traj"))
    comparison = HFComparison(
        type="pairwise", play_mode="synchronized", camera="fixed_front",
        length_frames=end - start,
        extra={
            "pair_idx": record["pair_idx"],
            "pair_id": record["pair_id"],
            "motion_idx": record["motion_idx"],
            "motion_id": record["motion_id"],
            "category": record["category"],
            "tracker_pair_key": record["tracker_pair_key"],
            "clip_uid": record["clip_uid"],
            "source_start_frame": record["source_start_frame"],
            "source_end_frame": record["source_end_frame"],
            "fps": record["fps"],
            "lease_id": lease_id,
            "annotator_id": annotator_id,
            "client_id": client_id,
            "human_preference": {
                "choice_type": choice_type,
                "preferred_candidate_idx": preferred_candidate_idx,
            },
            "display_order": display_order,
            "display_choice_idx": display_choice_idx,
        },
    )
    return HFRecord(meta=meta, video_left=video_left, video_right=video_right,
                    preference=preference, flags=flags, comparison=comparison)


def parse_args():
    ap = argparse.ArgumentParser(description="Motion Labeler")
    ap.add_argument("--task-file", required=True,
                help="pairs JSONL from rm_pipeline, or 'latest' for the newest one")
    ap.add_argument("--hf-logs-dir", default="auto")
    ap.add_argument("--cache-dir", default=".video_cache")
    ap.add_argument(
        "--scene-xml",
        default=str(DEFAULT_SCENE_XML),
        help=f"MuJoCo scene XML (default: {DEFAULT_SCENE_XML})",
    )
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--annotator-id", default=None, help="Default annotator id shown in the browser.")
    ap.add_argument("--annotators", nargs="*", default=None, help="Optional annotator suggestions for the browser datalist.")
    ap.add_argument("--annotations-per-pair", type=int, default=1, help="Dynamic target annotations collected per pair.")
    ap.add_argument(
        "--lease-seconds",
        type=int,
        default=120,
        help="Inactivity timeout renewed by the active browser heartbeat.",
    )
    ap.add_argument("--state-file", default="", help="Dynamic state JSON path. Defaults to <hf-logs-dir>/dynamic_state.json.")
    ap.add_argument(
        "--external-prerender",
        action="store_true",
        help="Let an external multi-GPU job fill the cache; the web worker renders only browser-priority clips.",
    )
    return ap.parse_args()


HTML_PATH = Path(__file__).with_name("static") / "index.html"
HTML_PAGE = HTML_PATH.read_text(encoding="utf-8")


def create_server(args):
    task_path = resolve_task_file(args.task_file)
    args.task_file = str(task_path)
    hf_logs_dir = resolve_hf_logs_dir(args.hf_logs_dir, task_path)
    args.hf_logs_dir = str(hf_logs_dir)

    scene_xml_path = Path(args.scene_xml).expanduser().resolve()
    if not scene_xml_path.is_file():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml_path}")
    args.scene_xml = str(scene_xml_path)
    scene_hasher = hashlib.sha256()
    scene_hasher.update(str(scene_xml_path).encode("utf-8"))
    scene_hasher.update(b"\0")
    scene_hasher.update(scene_xml_path.read_bytes())
    scene_cache_key = f"{scene_xml_path.stem}-{scene_hasher.hexdigest()[:12]}"

    cache_root = Path(args.cache_dir)
    cache_dir = cache_root / "scenes" / scene_cache_key

    print(f"[config] task-file:   {args.task_file}")
    print(f"[config] hf-logs-dir: {args.hf_logs_dir}")
    print(f"[config] cache-root:  {cache_root}")
    print(f"[config] cache-dir:   {cache_dir}")
    print(f"[config] scene-xml:   {args.scene_xml}")
    print(f"[config] scene-cache-key: {scene_cache_key}")
    print(f"[config] port:        {args.port}")
    print(f"[config] annotations-per-pair: {args.annotations_per_pair}")
    print(f"[config] lease-seconds:        {args.lease_seconds}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    compat_cache_dir = cache_dir.parent / f"{cache_dir.name}_compat"
    compat_cache_dir.mkdir(parents=True, exist_ok=True)
    combined_cache_dir = cache_dir.parent / f"{cache_dir.name}_combined"
    combined_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_logs_dir = Path(args.hf_logs_dir)
    hf_logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] compat-cache-dir: {compat_cache_dir}")
    print(f"[config] combined-cache-dir: {combined_cache_dir}")

    state_file = Path(args.state_file) if args.state_file else hf_logs_dir / "dynamic_state.json"
    if not state_file.is_absolute():
        state_file = REPO_ROOT / state_file
    print(f"[config] state-file:  {state_file}")

    task_mgr = DynamicTaskManager(
        args.task_file,
        state_file=state_file,
        annotations_per_pair=args.annotations_per_pair,
        lease_seconds=args.lease_seconds,
    )
    recorder = HFRecorder(hf_logs_dir, undo_limit=UNDO_LIMIT)
    annotation_lock = threading.Lock()

    # ---- Background pre-rendering -----------------------------------------
    # Resolve every clip up-front and let a worker thread render them all into
    # the disk cache. The pair the annotator is viewing (and its neighbours)
    # gets pushed to the front, so navigation is effectively instant.
    manager = RenderManager(
        cache_dir,
        args.scene_xml,
        prefetch_all=not args.external_prerender,
    )

    def _pair_meta(idx: int, display_order: list[int] | None = None):
        """Return (left_url, right_url, left_key, right_key) for a pair index."""
        left_info, right_info, record, _ = task_mgr.get_pair_for_display_order(
            idx, [0, 1] if display_order is None else display_order
        )
        start, end = task_mgr.get_frame_range(record)
        left_npz = task_mgr.get_npz_path(left_info)
        right_npz = task_mgr.get_npz_path(right_info)
        lkey = cache_filename(left_npz, start, end)
        rkey = cache_filename(right_npz, start, end)
        scene_query = quote(scene_cache_key)
        return (
            f"/videos_compat/{lkey}?scene={scene_query}",
            f"/videos_compat/{rkey}?scene={scene_query}",
            lkey,
            rkey,
        )

    jobs = []
    for i in range(task_mgr.total_pairs):
        left_info, right_info, record, _ = task_mgr.get_pair(i)
        start, end = task_mgr.get_frame_range(record)
        jobs.append(RenderJob(task_mgr.get_npz_path(left_info), start, end, left_info))
        jobs.append(RenderJob(task_mgr.get_npz_path(right_info), start, end, right_info))
    manager.add_jobs(jobs)
    manager.start()
    print(f"[render] queued {len(jobs)} clips for background pre-rendering")
    print(
        "[render] mode: "
        + ("browser-priority only (external pre-render active)" if args.external_prerender else "full background scan")
    )

    app = FastAPI()
    compat_lock = threading.Lock()
    combined_lock = threading.Lock()
    combined_stop = threading.Event()
    combined_total = 0
    combined_done = 0
    combined_failed = 0
    combined_current = ""
    combined_status_lock = threading.Lock()
    combined_priority_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()
    combined_priority_seen: set[str] = set()
    combined_priority_lock = threading.Lock()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required")

    def _safe_video_path(directory: Path, filename: str) -> Path:
        if Path(filename).name != filename or not filename.endswith(".mp4"):
            raise HTTPException(status_code=404, detail="invalid video name")
        path = directory / filename
        if not path.exists() or path.stat().st_size <= 0:
            raise HTTPException(status_code=404, detail="video not found")
        return path

    def _ensure_compat_video(filename: str) -> Path:
        src = _safe_video_path(cache_dir, filename)
        out = compat_cache_dir / filename
        if out.exists() and out.stat().st_size > 0:
            return out
        tmp = out.with_suffix(".tmp.mp4")
        with compat_lock:
            if out.exists() and out.stat().st_size > 0:
                return out
            tmp.unlink(missing_ok=True)
            command = [
                ffmpeg, "-y", "-i", str(src), "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
                "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "21",
                "-bf", "0", "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
                "-movflags", "+faststart", str(tmp),
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise RuntimeError(f"ffmpeg produced an empty video: {tmp}")
            tmp.replace(out)
            return out


    def _combined_filename(left_key: str, right_key: str) -> str:
        return combined_filename(scene_cache_key, left_key, right_key)

    def _combined_url(left_key: str, right_key: str) -> str:
        filename = _combined_filename(left_key, right_key)
        return (
            f"/videos_combined/{filename}?scene={quote(scene_cache_key)}"
            f"&left={quote(left_key)}&right={quote(right_key)}"
        )

    def _combined_path(filename: str) -> Path:
        return combined_cache_dir / filename

    def _combined_exists(filename: str) -> bool:
        path = _combined_path(filename)
        return path.exists() and path.stat().st_size > 0

    def _schedule_combined_build(filename: str, left_key: str, right_key: str) -> None:
        if _combined_exists(filename):
            return
        with combined_priority_lock:
            if filename in combined_priority_seen:
                return
            combined_priority_seen.add(filename)
        combined_priority_queue.put((filename, left_key, right_key))

    def _ensure_combined_video(filename: str, left_key: str, right_key: str) -> Path:
        if filename != _combined_filename(left_key, right_key):
            raise HTTPException(status_code=404, detail="combined video key mismatch")
        left_src = _safe_video_path(cache_dir, left_key)
        right_src = _safe_video_path(cache_dir, right_key)
        out = _combined_path(filename)
        if out.exists() and out.stat().st_size > 0:
            return out

        tmp = out.with_suffix(".tmp.mp4")
        with combined_lock:
            if out.exists() and out.stat().st_size > 0:
                return out
            tmp.unlink(missing_ok=True)
            cmd = [
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
            t0 = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
                stderr = (result.stderr or "").strip().splitlines()
                tail = "\n".join(stderr[-8:])
                tmp.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail=f"ffmpeg hstack failed: {tail}")
            tmp.replace(out)
            print(f"[video-combined] {filename} ({time.time() - t0:.2f}s)")
            return out

    def _build_combined_when_ready(filename: str, left_key: str, right_key: str) -> None:
        lready = manager.is_ready(left_key)
        rready = manager.is_ready(right_key)
        while not (lready and rready):
            if combined_stop.is_set():
                return
            if manager.error_for(left_key) or manager.error_for(right_key):
                raise RuntimeError("source render failed")
            time.sleep(0.25)
            lready = manager.is_ready(left_key)
            rready = manager.is_ready(right_key)
        if combined_stop.is_set():
            return
        _ensure_combined_video(filename, left_key, right_key)

    def _combined_priority_worker() -> None:
        nonlocal combined_current
        while not combined_stop.is_set():
            try:
                filename, left_key, right_key = combined_priority_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            with combined_status_lock:
                combined_current = f"priority:{filename}"
            requeued = False
            try:
                source_error = manager.error_for(left_key) or manager.error_for(right_key)
                if source_error:
                    raise RuntimeError(f"source render failed: {source_error}")
                if not (manager.is_ready(left_key) and manager.is_ready(right_key)):
                    # A missing source must not block every ready pair behind it.
                    # Keep this request in the queue and prioritize its sources,
                    # then give another combined request a chance to run.
                    manager.prioritize([left_key, right_key])
                    combined_priority_queue.put((filename, left_key, right_key))
                    requeued = True
                    time.sleep(0.1)
                else:
                    _ensure_combined_video(filename, left_key, right_key)
            except Exception as e:  # noqa: BLE001 - keep priority builder alive
                print(f"[video-combined] priority FAILED {filename}: {e}")
            finally:
                if not requeued:
                    with combined_priority_lock:
                        combined_priority_seen.discard(filename)
                with combined_status_lock:
                    combined_current = ""
                combined_priority_queue.task_done()

    combined_jobs: list[tuple[str, str, str]] = []
    seen_combined: set[str] = set()
    for i in range(task_mgr.total_pairs):
        for order in ([0, 1], [1, 0]):
            _, _, lkey, rkey = _pair_meta(i, display_order=order)
            name = _combined_filename(lkey, rkey)
            if name in seen_combined:
                continue
            seen_combined.add(name)
            combined_jobs.append((name, lkey, rkey))
    combined_total = len(combined_jobs)

    def _combined_prebuild_worker() -> None:
        nonlocal combined_done, combined_failed, combined_current
        print(f"[video-combined] queued {combined_total} pair videos for background prebuild")
        for name, lkey, rkey in combined_jobs:
            if combined_stop.is_set():
                break
            while not combined_stop.is_set() and not combined_priority_queue.empty():
                time.sleep(0.1)
            if combined_stop.is_set():
                break
            with combined_status_lock:
                combined_current = name
            try:
                _build_combined_when_ready(name, lkey, rkey)
                if combined_stop.is_set():
                    break
                with combined_status_lock:
                    combined_done += 1
            except Exception as e:  # noqa: BLE001 - keep prebuild moving
                with combined_status_lock:
                    combined_failed += 1
                print(f"[video-combined] FAILED {name}: {e}")
        with combined_status_lock:
            combined_current = ""
        print(f"[video-combined] prebuild stopped/done: done={combined_done} failed={combined_failed} total={combined_total}")

    if not args.external_prerender:
        combined_worker = threading.Thread(
            target=_combined_prebuild_worker,
            name="combined-prebuild-worker",
            daemon=True,
        )
        combined_worker.start()
    combined_priority_worker = threading.Thread(
        target=_combined_priority_worker,
        name="combined-priority-worker",
        daemon=True,
    )
    combined_priority_worker.start()

    def _prioritize_pair(idx: int, display_order: list[int]) -> None:
        _, _, left_key, right_key = _pair_meta(idx, display_order=display_order)
        manager.prioritize([left_key, right_key])
        combined_name = _combined_filename(left_key, right_key)
        if not _combined_exists(combined_name):
            _schedule_combined_build(combined_name, left_key, right_key)

    def _pair_payload(
        idx: int,
        *,
        lease: dict | None,
        display_order: list[int] | None,
        prioritize: bool,
        progress: dict,
        stats: dict | None = None,
        undo_depth: int | None = None,
    ):
        if idx < 0 or idx >= task_mgr.total_pairs:
            raise HTTPException(status_code=404, detail=f"pair index out of range: {idx}")
        if (lease is None) == (display_order is None):
            raise ValueError("exactly one of lease and display_order is required")
        order = lease["display_order"] if lease is not None else display_order
        left_url, right_url, lkey, rkey = _pair_meta(idx, display_order=order)
        if prioritize:
            _prioritize_pair(idx, order)
        err = manager.error_for(lkey) or manager.error_for(rkey)
        ready = manager.is_ready(lkey) and manager.is_ready(rkey)
        combined_name = _combined_filename(lkey, rkey)
        combined_ready = _combined_exists(combined_name)
        _, _, record, _ = task_mgr.get_pair_for_display_order(idx, order)
        payload = {
            "idx": idx,
            "total": task_mgr.total_pairs,
            "pair_id": record["pair_id"],
            "motion_id": record["motion_id"],
            "clip_uid": record["clip_uid"],
            "left_url": left_url,
            "right_url": right_url,
            "combined_url": _combined_url(lkey, rkey),
            "combined_ready": combined_ready,
            "ready": ready,
            "error": err,
            "display_order": order,
            "cache_done": progress["done"],
            "cache_total": progress["total"],
        }
        if stats is not None:
            payload["stats"] = stats
        if lease:
            if undo_depth is None:
                raise ValueError("undo_depth is required for a leased task")
            payload.update({
                "lease_id": lease["lease_id"],
                "lease_expires_at": lease["expires_at"],
                "annotator_id": lease["annotator_id"],
                "client_id": lease["client_id"],
                "undo_depth": undo_depth,
            })
        return payload

    def _next_response(annotator: str, client_id: str):
        lease = _lease_call(task_mgr.lease_next, annotator, client_id)
        stats = task_mgr.stats()
        undo_depth = task_mgr.undo_depth(annotator)
        progress = manager.progress()
        if lease is None:
            return {
                "done": True,
                "stats": stats,
                "undo_depth": undo_depth,
                "cache_done": progress["done"],
                "cache_total": progress["total"],
            }
        return _pair_payload(
            int(lease["idx"]),
            lease=lease,
            display_order=None,
            prioritize=True,
            progress=progress,
            stats=stats,
            undo_depth=undo_depth,
        )

    def _prefetch_response(annotator: str, limit: int):
        previews, stats = _lease_call(
            task_mgr.prefetch_snapshot, annotator, limit
        )
        progress = manager.progress()
        items = [
            _pair_payload(
                idx,
                lease=None,
                display_order=display_order,
                prioritize=False,
                progress=progress,
            )
            for idx, display_order in previews
        ]
        return {"items": items, "stats": stats}

    def _pair_response(idx: int):
        return _pair_payload(
            idx,
            lease=None,
            display_order=[0, 1],
            prioritize=True,
            progress=manager.progress(),
            stats=task_mgr.stats(),
        )

    def _label_response(
        idx: int,
        label: str,
        annotator: str,
        client_id: str,
        lease_id: str,
    ):
        _lease_call(task_mgr.renew_lease, lease_id, annotator, client_id)
        record = _lease_call(
            build_record,
            task_mgr,
            idx,
            label,
            annotator,
            client_id,
            lease_id,
        )
        with annotation_lock:
            output_path = recorder.append(record)
            pending_count = recorder.pending_count
            transition = _lease_call(
                task_mgr.complete_and_lease_next,
                lease_id,
                annotator,
                client_id,
                label,
                record.meta.record_id,
            )
        storage_status = (
            f"wrote {output_path.name}"
            if output_path is not None
            else f"journaled {pending_count}/{recorder.shard_size} pending records"
        )
        response = {
            "status": f"Saved {record.meta.record_id} ({label}); {storage_status}",
            "stats": transition["stats"],
            "undo_depth": transition["undo_depth"],
            "done": transition["lease"] is None,
        }
        if transition["lease"] is not None:
            response["next_task"] = _pair_payload(
                int(transition["lease"]["idx"]),
                lease=transition["lease"],
                display_order=None,
                prioritize=True,
                progress=manager.progress(),
                stats=transition["stats"],
                undo_depth=transition["undo_depth"],
            )
        return response

    def _undo_response(annotator: str, client_id: str):
        with annotation_lock:
            target = _lease_call(task_mgr.peek_undo, annotator)
            _lease_call(
                recorder.undo,
                target["record_id"],
                annotator,
                target["pair_id"],
            )
            lease = _lease_call(
                task_mgr.undo_last,
                annotator,
                client_id,
                target["record_id"],
            )
        stats = task_mgr.stats()
        undo_depth = task_mgr.undo_depth(annotator)
        payload = _pair_payload(
            int(lease["idx"]),
            lease=lease,
            display_order=None,
            prioritize=True,
            progress=manager.progress(),
            stats=stats,
            undo_depth=undo_depth,
        )
        return {
            "status": f"Undid {target['pair_id']}; label it again.",
            "undone_record_id": target["record_id"],
            "undo_depth": undo_depth,
            "task": payload,
            "stats": stats,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index():
        annotators = args.annotators or task_mgr.get_annotators() or []
        options = "".join(f'<option value="{a}"></option>' for a in annotators)
        default_annotator = json.dumps(args.annotator_id or (annotators[0] if annotators else "default"))
        content = (
            HTML_PAGE
            .replace("ANNOTATOR_OPTIONS", options)
            .replace("DEFAULT_ANNOTATOR", default_annotator)
        )
        return HTMLResponse(content=content, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/next")
    async def get_next(annotator: str, client_id: str):
        annotator = _required_string(annotator, "annotator")
        client_id = _required_string(client_id, "client_id")
        return await asyncio.to_thread(_next_response, annotator, client_id)


    @app.get("/api/prefetch")
    async def get_prefetch(annotator: str, limit: int = 4):
        annotator = _required_string(annotator, "annotator")
        if limit < 1 or limit > 24:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 24")
        return await asyncio.to_thread(_prefetch_response, annotator, limit)


    @app.get("/api/pair/{idx}")
    async def get_pair(idx: int):
        return await asyncio.to_thread(_pair_response, idx)


    @app.get("/api/progress")
    async def get_progress():
        def response():
            progress = manager.progress()
            compat_done = sum(1 for _ in compat_cache_dir.glob("*.mp4"))
            combined_cache_done = sum(
                _combined_exists(filename) for filename, _, _ in combined_jobs
            )
            with combined_status_lock:
                prebuild = {
                    "total": combined_total,
                    "done": combined_done,
                    "failed": combined_failed,
                    "current": combined_current,
                    "priority_queue": combined_priority_queue.qsize(),
                }
            return {
                **progress,
                "compat_done": compat_done,
                "combined_done": combined_cache_done,
                "combined_prebuild": prebuild,
            }

        return await asyncio.to_thread(response)

    @app.get("/videos_compat/{filename}")
    async def get_compat_video(filename: str):
        if not manager.has_job(filename):
            raise HTTPException(status_code=404, detail="unknown video")
        render_error = manager.error_for(filename)
        if render_error:
            raise HTTPException(status_code=500, detail=f"source render failed: {render_error}")
        if not manager.is_ready(filename):
            manager.prioritize([filename])
            return Response(
                status_code=202,
                headers={"Retry-After": "1", "Cache-Control": "no-store", "X-Video-State": "rendering"},
            )
        path = await asyncio.to_thread(_ensure_compat_video, filename)
        return FileResponse(
            str(path),
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=604800, immutable"},
        )

    @app.get("/videos_combined/{filename}")
    async def get_combined_video(filename: str, left: str, right: str):
        if filename != _combined_filename(left, right):
            raise HTTPException(status_code=404, detail="combined video key mismatch")
        if not manager.has_job(left) or not manager.has_job(right):
            raise HTTPException(status_code=404, detail="unknown source video")
        render_error = manager.error_for(left) or manager.error_for(right)
        if render_error:
            raise HTTPException(status_code=500, detail=f"source render failed: {render_error}")
        if _combined_exists(filename):
            return FileResponse(
                str(_combined_path(filename)),
                media_type="video/mp4",
                headers={"Cache-Control": "public, max-age=604800, immutable"},
            )
        manager.prioritize([left, right])
        _schedule_combined_build(filename, left, right)
        return Response(
            status_code=202,
            headers={"Retry-After": "1", "Cache-Control": "no-store", "X-Video-State": "rendering"},
        )

    @app.post("/api/lease/heartbeat")
    async def post_lease_heartbeat(request: Request):
        body = await request.json()
        annotator = _required_string(body.get("annotator"), "annotator")
        client_id = _required_string(body.get("client_id"), "client_id")
        lease_id = _required_string(body.get("lease_id"), "lease_id")
        lease = await asyncio.to_thread(
            _lease_call, task_mgr.renew_lease, lease_id, annotator, client_id
        )
        return {"lease_expires_at": lease["expires_at"]}

    @app.post("/api/lease/release")
    async def post_lease_release(request: Request):
        body = await request.json()
        annotator = _required_string(body.get("annotator"), "annotator")
        client_id = _required_string(body.get("client_id"), "client_id")
        lease_id = _required_string(body.get("lease_id"), "lease_id")
        released = await asyncio.to_thread(
            _lease_call, task_mgr.release_lease, lease_id, annotator, client_id
        )
        return {"released": released}

    @app.post("/api/label")
    async def post_label(request: Request):
        body = await request.json()
        idx = int(body["idx"])
        label = str(body["label"])
        annotator = _required_string(body.get("annotator"), "annotator")
        client_id = _required_string(body.get("client_id"), "client_id")
        lease_id = _required_string(body.get("lease_id"), "lease_id")
        return await asyncio.to_thread(
            _label_response, idx, label, annotator, client_id, lease_id
        )

    @app.post("/api/undo")
    async def post_undo(request: Request):
        body = await request.json()
        annotator = _required_string(body.get("annotator"), "annotator")
        client_id = _required_string(body.get("client_id"), "client_id")
        return await asyncio.to_thread(_undo_response, annotator, client_id)


    # Mount static files AFTER routes so it doesn't shadow them
    app.mount("/videos", StaticFiles(directory=str(cache_dir)), name="videos")

    @app.on_event("shutdown")
    async def _shutdown_combined_worker():
        combined_stop.set()

    return app, recorder


def main():
    args = parse_args()
    app, recorder = create_server(args)
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
