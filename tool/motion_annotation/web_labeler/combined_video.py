"""Shared encoding contract for side-by-side annotation videos."""

from __future__ import annotations

import hashlib


SIDE_WIDTH = 480
SIDE_HEIGHT = 360
CRF = 26
FILTER_GRAPH = (
    f"[0:v]fps=50,scale={SIDE_WIDTH}:{SIDE_HEIGHT}:flags=bicubic,setsar=1[left];"
    f"[1:v]fps=50,scale={SIDE_WIDTH}:{SIDE_HEIGHT}:flags=bicubic,setsar=1[right];"
    "[left][right]hstack=inputs=2[v]"
)

# Annotators mostly reach the labeler over thin tunnels (SSH port forwarding /
# public relays with well under 1 MB/s), so the encoded size of each pair video
# dominates end-to-end latency. Main profile with B-frames and a 2.5 s GOP cuts
# the file size roughly 4x versus the previous baseline/-bf 0/-g 25 settings at
# visually equivalent quality for motion comparison.
ENCODE_ARGS = [
    "-c:v", "libx264",
    "-profile:v", "main",
    "-level", "4.0",
    "-pix_fmt", "yuv420p",
    "-preset", "veryfast",
    "-crf", str(CRF),
    "-g", "125",
    "-keyint_min", "25",
    "-movflags", "+faststart",
]


def cache_filename(scene_key: str, left_key: str, right_key: str) -> str:
    encoding = f"{FILTER_GRAPH}|{' '.join(ENCODE_ARGS)}"
    digest = hashlib.md5(
        f"{encoding}|{scene_key}|{left_key}|{right_key}".encode()
    ).hexdigest()[:12]
    return f"combined_{digest}.mp4"
