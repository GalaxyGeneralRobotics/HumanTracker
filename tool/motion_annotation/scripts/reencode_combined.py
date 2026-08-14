"""Re-encode existing combined pair videos with the current ENCODE_ARGS.

The original combined cache was encoded with baseline profile, no B-frames and
a 0.5 s GOP, which produced ~4 MB per 5 s pair video -- far too heavy for the
thin tunnels annotators use. This tool rebuilds every combined video from its
two cached source clips (so there is no extra generation loss) using the
current ``web_labeler.combined_video.ENCODE_ARGS`` and atomically replaces the
old file. The live labeler can keep serving while it runs.

Outputs already encoded with the current profile ("Main") are skipped, so the
command is safe to stop and resume.

Run from tool/motion_annotation with the h-gpt environment:

    python -m scripts.reencode_combined \
      --task-file .../pairs.jsonl \
      --cache-root .video_cache/20260724_093027 \
      --workers 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from web_labeler.combined_video import ENCODE_ARGS, FILTER_GRAPH
from web_labeler.prerender_all import (
    DEFAULT_SCENE_XML,
    CombinedJob,
    _load_jobs,
    _scene_cache_key,
)


def _current_profile(ffprobe: str, path: Path) -> str:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=profile", "-of", "csv=p=0", str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return (result.stdout or "").strip()


def _reencode(
    job: CombinedJob,
    cache_dir: Path,
    combined_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    force: bool,
) -> tuple[str, str, int, int, float]:
    """Return (filename, action, old_size, new_size, seconds)."""
    out = combined_dir / job.filename
    if not out.is_file() or out.stat().st_size <= 0:
        return job.filename, "missing", 0, 0, 0.0
    old_size = out.stat().st_size
    if not force and _current_profile(ffprobe, out) == "Main":
        return job.filename, "skipped", old_size, old_size, 0.0

    left_src = cache_dir / job.left_key
    right_src = cache_dir / job.right_key
    if not left_src.is_file() or not right_src.is_file():
        raise FileNotFoundError(f"missing source clips for {job.filename}")

    tmp = out.with_name(f".{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
    tmp.unlink(missing_ok=True)
    command = [
        ffmpeg, "-y",
        "-i", str(left_src),
        "-i", str(right_src),
        "-filter_complex", FILTER_GRAPH,
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
    new_size = tmp.stat().st_size
    tmp.replace(out)
    return job.filename, "reencoded", old_size, new_size, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--cache-root", type=Path, default=Path(".video_cache"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N videos (for testing).")
    parser.add_argument("--force", action="store_true", help="Re-encode even if the file already uses the current profile.")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required in PATH")

    task_file = args.task_file.expanduser().resolve()
    scene_xml = args.scene_xml.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    scene_key = _scene_cache_key(scene_xml)
    cache_dir = cache_root / "scenes" / scene_key
    combined_dir = cache_dir.parent / f"{cache_dir.name}_combined"
    if not combined_dir.is_dir():
        raise FileNotFoundError(combined_dir)

    _, combined_jobs = _load_jobs(task_file, scene_key)
    if args.limit > 0:
        combined_jobs = combined_jobs[: args.limit]
    total = len(combined_jobs)
    print(f"[config] combined={total} workers={args.workers} dir={combined_dir}", flush=True)

    counts = {"reencoded": 0, "skipped": 0, "missing": 0, "failed": 0}
    old_total = new_total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_reencode, job, cache_dir, combined_dir, ffmpeg, ffprobe, args.force): job
            for job in combined_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            done += 1
            try:
                filename, action, old_size, new_size, elapsed = future.result()
                counts[action] += 1
                if action == "reencoded":
                    old_total += old_size
                    new_total += new_size
                    print(
                        f"[reencode] {done}/{total} {filename} "
                        f"{old_size / 1e6:.2f}MB -> {new_size / 1e6:.2f}MB ({elapsed:.1f}s)",
                        flush=True,
                    )
                elif done % 500 == 0:
                    print(f"[reencode] {done}/{total} ({action})", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep the batch moving
                counts["failed"] += 1
                print(f"[reencode] FAILED {job.filename}: {exc!r}", flush=True)

    if old_total:
        print(
            f"[done] reencoded={counts['reencoded']} skipped={counts['skipped']} "
            f"missing={counts['missing']} failed={counts['failed']} "
            f"size {old_total / 1e9:.2f}GB -> {new_total / 1e9:.2f}GB "
            f"({old_total / max(1, new_total):.1f}x smaller)",
            flush=True,
        )
    else:
        print(f"[done] {counts}", flush=True)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
