"""Compose side-by-side comparison videos from walk_viz_sweep outputs.

Looks up ``video.mp4`` under each tracker's ``videos_<tag>/<idx>_<stem>/``.
Writes ``<out>/tiles/<stem>.mp4``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TAGS = (
    ("h264", "videos_hgpt264_gmr_stream_s1.0_filtered"),
    ("h216", "videos_hgpt216_gmr_stream_s1.0_filtered"),
)


def find_clip(root: Path, stem: str) -> Path | None:
    matches = sorted(root.glob(f"*_{stem}/video.mp4"))
    return matches[0] if matches else None


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def tile(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # One labelled column per tracker. Shortest stream wins so a fall does not freeze.
    filt = "".join(
        f"[{i}:v]drawtext=text={label}:x=12:y=12:fontsize=28:fontcolor=white,"
        f"setpts=PTS-STARTPTS[v{i}];"
        for i, (label, _) in enumerate(TAGS)
    )
    filt += "".join(f"[v{i}]" for i in range(len(TAGS))) + f"hstack=inputs={len(TAGS)}[v]"
    cmd = ["ffmpeg", "-y"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", filt, "-map", "[v]", "-an", "-shortest", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(dest)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("viz_dir", nargs="?", default="outputs/eval_runs/walk_viz")
    args = parser.parse_args()
    root = Path(args.viz_dir)
    if not have_ffmpeg():
        print("ffmpeg not found; leave individual tracker videos as-is", file=sys.stderr)
        sys.exit(1)

    # Directory is 00000_Daily_198 — keep everything after the first underscore.
    stems: set[str] = set()
    for _, folder in TAGS:
        for mp4 in (root / folder).glob("*/video.mp4"):
            name = mp4.parent.name
            stems.add(name.split("_", 1)[1] if "_" in name else name)

    written = 0
    for stem in sorted(stems):
        paths = []
        missing = False
        for _, folder in TAGS:
            p = find_clip(root / folder, stem)
            if p is None:
                print(f"skip {stem}: missing {folder}")
                missing = True
                break
            paths.append(p)
        if missing:
            continue
        dest = root / "tiles" / f"{stem}.mp4"
        print(f"tile {stem} -> {dest}")
        tile(paths, dest)
        written += 1
    print(f"wrote {written} tiled videos")


if __name__ == "__main__":
    main()
