"""Summarise the deploy-faithful sweep (``tool/deploy_faithful_sweep.sh``).

Prints one table per reference condition for the Humanoid-GPT policies:

1. Clean G1 reference.
2. ``xsens_human`` -- ScaleBridge-style human-proportion 14-cloud.
3. ``gmr_stream`` -- GMR + EMA + delay, the HGPT live input.
4. Each policy on its live input.
5. Command-following (error vs the *dirty* reference).

Then the clean -> live HumanScore drop per policy.

Usage: python tool/deploy_faithful_report.py [sweep_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

TRACKERS = {
    "hgpt264": "HGPT 264",
    "hgpt216": "HGPT 216",
}


def load(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    rows: List[Dict] = data["per_trajectory"]

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def median(key: str) -> float:
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        if not vals:
            return float("nan")
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])

    return {
        "n": len(rows),
        "success": 100.0 * sum(r["length_ratio"] >= 1.0 for r in rows) / len(rows),
        "hs": 100.0 * mean("score"),
        "mpjpe": 1000.0 * mean("kpt_pos_mae"),
        # Mean action jerk is dominated by a handful of HGPT-216 explosions;
        # the tables use the median so those clips do not set the column.
        "ajerk": median("action_jerk_mean"),
        "ajerk_mean": mean("action_jerk_mean"),
        "cmd_kpt": 1000.0 * mean("cmd_kpt_pos_mae"),
        "cmd_jnt": mean("cmd_joint_pos_mae"),
        "noise": data.get("ref_noise", {}),
    }


def cell(row: Optional[Dict], cmd: bool = False) -> str:
    if row is None:
        return f"{'--':>23}"
    if cmd:
        return f"{row['success']:5.0f}% {row['hs']:5.1f} {row['cmd_kpt']:6.1f} {row['cmd_jnt']:6.3f}"
    return f"{row['success']:5.0f}% {row['hs']:5.1f} {row['mpjpe']:6.1f} {row['ajerk']:7.0f}"


def table(title: str, note: str, names: List[str], columns: List[tuple], cmd: bool = False) -> None:
    print(f"\n{title}\n{note}")
    head = f"{'tracker':16}" + "".join(f"  {label:^23}" for label, _ in columns)
    print(head)
    units = "succ    HS  cmdK   cmdJ" if cmd else "succ    HS  MPJPE   ajerk"
    print(f"{'':16}" + "".join(f"  {units:>23}" for _ in columns))
    print("-" * len(head))
    for key in names:
        line = f"{TRACKERS[key]:16}"
        for _, resolve in columns:
            line += "  " + cell(resolve(key), cmd=cmd)
        print(line)


def _rank(rows: List[tuple]) -> str:
    present = [(n, r) for n, r in rows if r is not None]
    present.sort(key=lambda x: (-x[1]["success"], -x[1]["hs"]))
    return " > ".join(n for n, _ in present) if present else "(missing)"


def main() -> None:
    sweep = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_runs/noise_sweep_deploy")
    cache: Dict[str, Optional[Dict]] = {}

    def get(tracker: str, name: str) -> Optional[Dict]:
        if name not in cache:
            cache[name] = load(sweep / f"{name}.json")
        return cache[name]

    def clean(t: str) -> Optional[Dict]:
        return get(t, f"{t}_clean")

    def xsens(t: str, s: str) -> Optional[Dict]:
        return get(t, f"{t}_xsens_human_s{s}_clean")

    def gmr(t: str, s: str, vel: str = "filtered") -> Optional[Dict]:
        return get(t, f"{t}_gmr_stream_s{s}_{vel}")

    counts = {r["n"] for r in [clean(t) for t in TRACKERS] if r}
    print(f"sweep: {sweep}   clips per run: {sorted(counts) or ['(none yet)']}")

    table(
        "1. Clean G1 reference (HumanTracker domain)",
        "The training/eval domain for Humanoid-GPT.",
        list(TRACKERS),
        [("clean", lambda t: clean(t))],
    )

    table(
        "2. xsens_human -- ScaleBridge 14-segment human cloud",
        "All 14 links re-proportioned off G1 (per-clip arm/leg/torso scales) plus\n"
        "the same small mocap jitter.",
        list(TRACKERS),
        [(f"scale {s}", lambda t, s=s: xsens(t, s)) for s in ("0.5", "1.0", "2.0")],
    )

    table(
        "3. gmr_stream -- HGPT live IK + EMA + delay",
        "Kinematically consistent qpos, tau=38.6 ms EMA, ~65 ms latency, mild IK\n"
        "residual (wrists/ankles tighter).",
        list(TRACKERS),
        [(f"scale {s}", lambda t, s=s: gmr(t, s)) for s in ("0.5", "1.0", "2.0")],
    )

    table(
        "3b. HGPT velocity split on gmr_stream x1",
        "filtered = LiveRefConverter default; raw_diff = differentiate the noisy qpos.",
        ["hgpt264", "hgpt216"],
        [
            ("filtered", lambda t: gmr(t, "1.0", "filtered")),
            ("raw_diff", lambda t: gmr(t, "1.0", "raw_diff")),
        ],
    )

    table(
        "4. Live input -- gmr_stream x1, filtered velocity",
        "What each policy receives on the robot.",
        list(TRACKERS),
        [
            ("own live x1", lambda t: gmr(t, "1.0")),
        ],
    )

    table(
        "5. Command following on own live input (error vs dirty ref)",
        "cmdK = MPJPE vs the corrupted command (mm); cmdJ = joint MAE vs dirty qpos.\n"
        "Separates 'stable but ignoring the command' from 'faithfully tracking a bad one'.",
        list(TRACKERS),
        [
            ("own live x1", lambda t: gmr(t, "1.0")),
        ],
        cmd=True,
    )

    print("\n--- clean -> gmr_stream x1 HumanScore drop (larger = more damage) ---")
    for t in TRACKERS:
        c, g = clean(t), gmr(t, "1.0")
        print(f"{TRACKERS[t]:16} " + (f"{c['hs'] - g['hs']:+.1f}" if c and g else "incomplete"))
    print()


if __name__ == "__main__":
    main()
