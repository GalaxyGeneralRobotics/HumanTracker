"""Summarise a reference-noise sweep produced by ``tool/ref_noise_sweep.sh``.

Prints three tables, kept separate on purpose:

* the cross-tracker comparison, restricted to ``vel=clean`` because that is the
  only setting every tracker is equally exposed to;
* the velocity split, showing what differentiating a corrupted pose does to the
  two trackers that read reference velocity;
* the VLA chunking profile.

Usage: python tool/ref_noise_report.py [sweep_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

TRACKERS = {
    "sonic": "SONIC 1.0",
    "hgpt": "HGPT 264",
}
SCALES = ["0", "0.5", "1.0", "2.0"]


def load(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    rows: List[Dict] = data["per_trajectory"]

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "n": len(rows),
        # A clip counts as a success only if it ran to the end without tripping
        # the termination rule, which is what `length_ratio == 1` encodes.
        "success": 100.0 * sum(r["length_ratio"] >= 1.0 for r in rows) / len(rows),
        "hs": 100.0 * mean("score"),
        "mpjpe": 1000.0 * mean("kpt_pos_mae"),
        "mpjve": mean("joint_vel_mae"),
        "ajerk": mean("action_jerk_mean"),
        "noise": data.get("ref_noise", {}),
    }


def cell(row: Optional[Dict]) -> str:
    if row is None:
        return f"{'--':>21}"
    return f"{row['success']:5.0f}% {row['hs']:5.1f} {row['mpjpe']:6.1f} {row['ajerk']:7.0f}"


def table(title: str, note: str, names: List[str], columns: List[tuple]) -> None:
    print(f"\n{title}\n{note}")
    head = f"{'tracker':18}" + "".join(f"  {label:^21}" for label, _ in columns)
    print(head)
    print(f"{'':18}" + "".join(f"  {'succ    HS  MPJPE   ajerk':>21}" for _ in columns))
    print("-" * len(head))
    for key in names:
        line = f"{TRACKERS[key]:18}"
        for _, resolve in columns:
            line += "  " + cell(resolve(key))
        print(line)


def main() -> None:
    sweep = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_runs/noise_sweep_v2")
    cache: Dict[str, Optional[Dict]] = {}

    def get(tracker: str, profile: str, scale: str, vel: str) -> Optional[Dict]:
        name = f"{tracker}_clean" if scale == "0" else f"{tracker}_{profile}_s{scale}_{vel}"
        if name not in cache:
            cache[name] = load(sweep / f"{name}.json")
        return cache[name]

    counts = {
        r["n"] for r in cache.values() if r
    } or {  # nothing cached yet; peek at one file
        r["n"] for r in [get(t, "online_tracking", "0", "clean") for t in TRACKERS] if r
    }
    print(f"sweep: {sweep}   clips per run: {sorted(counts)}")
    ref = get("sonic", "online_tracking", "1.0", "clean")
    if ref:
        n = ref["noise"]
        print(
            f"scale=1 -> latency {n['timing']['latency_ms']:.0f} ms, "
            f"hold {n['timing']['hold_hz']:.0f} Hz, "
            f"bias {n['bias']['joint_deg']:.1f} deg / {n['bias']['root_pos_m']*100:.1f} cm, "
            f"jitter {n['jitter']['joint_deg']:.1f} deg (tau {n['jitter']['tau_s']:.2f} s)"
        )

    table(
        "Online-tracking robustness (reference velocity left clean)",
        "The comparable setting: only pose and timing corruption is applied, so\n"
        "trackers that read reference velocity gain nothing from it.",
        list(TRACKERS),
        [(f"scale {s}", lambda t, s=s: get(t, "online_tracking", s, "clean")) for s in SCALES],
    )

    table(
        "Velocity split at scale 1.0",
        "Same pose corruption; only how the command's velocity is derived changes.\n"
        "'raw diff' is the earlier behaviour, differentiated at the control rate.",
        ["sonic", "hgpt"],
        [
            ("clean vel", lambda t: get(t, "online_tracking", "1.0", "clean")),
            ("filtered vel", lambda t: get(t, "online_tracking", "1.0", "filtered")),
            ("raw diff", lambda t: get(t, "online_tracking", "1.0", "raw_diff")),
        ],
    )

    table(
        "VLA action-chunk profile at scale 1.0",
        "Per-chunk offsets that jump at the 10 Hz boundary, versus clean.",
        list(TRACKERS),
        [
            ("clean", lambda t: get(t, "online_tracking", "0", "clean")),
            ("vla_chunk", lambda t: get(t, "vla_chunk", "1.0", "clean")),
        ],
    )

    table(
        "Sparse-anchor / mid-link conflict (after-FK)",
        "Endpoints stay on the delayed skeleton; elbow/knee/shoulder/hip/torso\n"
        "get an independent heading-frame residual.",
        ["hgpt"],
        [(f"scale {s}", lambda t, s=s: get(t, "sparse_anchor", s, "clean")) for s in SCALES],
    )

    table(
        "VLA fill-in at scale 1.0",
        "Same mid-link conflict, plus 10 Hz chunk jumps on the filled-in links.",
        ["hgpt"],
        [
            ("clean", lambda t: get(t, "online_tracking", "0", "clean")),
            ("vla_fill", lambda t: get(t, "vla_fill", "1.0", "clean")),
        ],
    )
    print()


if __name__ == "__main__":
    main()
