"""Daily / walk-like slice of the deploy-faithful sweep.

The 100-clip mean mixes Interaction, kicks and ground work. Real-robot checks
are simple daily motions (walking). This report keeps only the Daily clips,
then the subset whose *reference* is actually locomotion: travelled distance,
walking-range planar speed, and a real single-support fraction.

Robot-state columns (what you see on the machine) come first; GT HumanScore
is kept as a secondary column, not the headline.

Usage:
  python tool/daily_walk_report.py [sweep_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MOTION_ROOT = Path("storage/dataset/HumanTracker/motions")
MANIFEST = Path("outputs/eval_runs/ref_noise_sweep_clips.json")

# Reference-side walk gate. Tuned on the 39 Daily clips in the 100-clip
# manifest so "walk" means the robot is going somewhere with a gait, not
# standing, fidgeting in place, or spinning.
WALK_SPEED_MIN = 0.25
WALK_SPEED_MAX = 1.60
WALK_TRAVEL_MIN = 1.50
WALK_SINGLE_MIN = 0.20
WALK_DUTY = (0.25, 0.85)
WALK_YAW_MAX = 1.20
WALK_ANKLE_STD_MIN = 0.02

TRACKERS_CLEAN = [
    ("hgpt264", "HGPT 264", "hgpt264_clean"),
    ("hgpt216", "HGPT 216", "hgpt216_clean"),
]
TRACKERS_OWN = [
    ("hgpt264", "HGPT 264", "hgpt264_gmr_stream_s1.0_filtered"),
    ("hgpt216", "HGPT 216", "hgpt216_gmr_stream_s1.0_filtered"),
]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def walk_features(npz_path: Path) -> Dict[str, float]:
    import numpy as np

    z = np.load(npz_path)
    qpos = z["qpos"].astype(np.float64)
    gv = z["gv_vel"].astype(np.float64)
    fc = z["foot_contact"].astype(np.float64)
    kpt = z["kpt2gv_pose"]
    travel = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    speed = float(np.mean(np.linalg.norm(gv[:, :2], axis=1)))
    yaw = float(np.mean(np.abs(gv[:, 2])))
    duty = fc.mean(axis=0)
    single = float(np.mean(fc.sum(axis=1) == 1))
    ankle = float(0.5 * (kpt[:, 3, 2, 3].std() + kpt[:, 6, 2, 3].std()))
    return {
        "frames": float(len(qpos)),
        "travel_m": travel,
        "speed_mps": speed,
        "yaw_rate": yaw,
        "duty_l": float(duty[0]),
        "duty_r": float(duty[1]),
        "single_support": single,
        "ankle_std_m": ankle,
    }


def is_walk(feat: Dict[str, float]) -> bool:
    lo, hi = WALK_DUTY
    return (
        WALK_SPEED_MIN <= feat["speed_mps"] <= WALK_SPEED_MAX
        and feat["travel_m"] >= WALK_TRAVEL_MIN
        and feat["single_support"] >= WALK_SINGLE_MIN
        and lo <= feat["duty_l"] <= hi
        and lo <= feat["duty_r"] <= hi
        and feat["yaw_rate"] < WALK_YAW_MAX
        and feat["ankle_std_m"] >= WALK_ANKLE_STD_MIN
    )


def classify_daily(motion_root: Path, manifest: Path) -> List[Dict]:
    entries = json.loads(manifest.read_text())
    out = []
    for e in entries:
        if e.get("category") != "Daily":
            continue
        feat = walk_features(motion_root / e["path"])
        feat["path"] = e["path"]
        feat["file_name"] = Path(e["path"]).name
        feat["walk"] = is_walk(feat)
        out.append(feat)
    return out


def load_run(path: Path) -> Optional[Dict[str, Dict]]:
    if not path.is_file():
        return None
    rows = {}
    for r in json.loads(path.read_text())["per_trajectory"]:
        rows[r["file_name"]] = r
    return rows


def clip_row(r: Dict) -> Dict[str, float]:
    return {
        "ok": 1.0 if r["length_ratio"] >= 1.0 else 0.0,
        "hs": 100.0 * r["score"],
        "mpjpe": 1000.0 * r["kpt_pos_mae"],
        "fc_acc": r.get("foot_contact_acc", float("nan")),
        "fc_iou": r.get("foot_contact_iou", float("nan")),
        "jerk": r.get("joint_jerk_mean", float("nan")),
        "ajerk": r.get("action_jerk_mean", float("nan")),
        "root_yaw": r.get("root_yaw_err", float("nan")),
        "root_vel": r.get("root_vel_err_mms", float("nan")),
        "root_pos": r.get("root_pos_err_mm", float("nan")),
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    def col(key: str, finite: bool = True) -> List[float]:
        vals = [r[key] for r in rows]
        if finite:
            vals = [v for v in vals if v == v and v not in (float("inf"), float("-inf"))]
        return vals

    hs = col("hs")
    return {
        "n": float(len(rows)),
        "succ": 100.0 * _mean(col("ok", finite=False)),
        "hs": _mean(hs),
        "hs_med": _median(hs),
        "hs_top": _mean(sorted(hs, reverse=True)[: min(8, len(hs))]) if hs else float("nan"),
        "mpjpe": _mean(col("mpjpe")),
        "fc_acc": _mean(col("fc_acc")),
        "fc_iou": _mean(col("fc_iou")),
        "jerk": _median(col("jerk")),
        "ajerk": _median(col("ajerk")),
        "root_yaw": _mean(col("root_yaw")),
        "root_vel": _mean(col("root_vel")),
    }


def cell(a: Optional[Dict[str, float]]) -> str:
    if a is None:
        return f"{'--':>44}"
    return (
        f"{a['succ']:4.0f}% {a['hs']:5.1f} {a['hs_top']:5.1f} "
        f"{a['fc_acc']:4.2f} {a['jerk']:6.0f} {a['ajerk']:6.0f} {a['root_yaw']:.3f}"
    )


def table(title: str, note: str, names: List[str], labels: Dict[str, str], columns: List) -> None:
    print(f"\n{title}\n{note}")
    head = f"{'tracker':16}" + "".join(f"  {lab:^44}" for lab, _ in columns)
    print(head)
    units = "succ    HS  top8  foot  jJerk  aJerk   yaw"
    print(f"{'':16}" + "".join(f"  {units:>44}" for _ in columns))
    print("-" * len(head))
    for key in names:
        line = f"{labels[key]:16}"
        for _, resolve in columns:
            line += "  " + cell(resolve(key))
        print(line)


def main() -> None:
    sweep = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_runs/noise_sweep_deploy")
    repo = Path(__file__).resolve().parents[1]
    motion_root = repo / MOTION_ROOT
    manifest = repo / MANIFEST
    daily = classify_daily(motion_root, manifest)
    walk = [d for d in daily if d["walk"]]
    walk_names = {d["file_name"] for d in walk}
    daily_names = {d["file_name"] for d in daily}

    listing = {
        "gate": {
            "speed_mps": [WALK_SPEED_MIN, WALK_SPEED_MAX],
            "travel_m_min": WALK_TRAVEL_MIN,
            "single_support_min": WALK_SINGLE_MIN,
            "duty": list(WALK_DUTY),
            "yaw_rate_max": WALK_YAW_MAX,
            "ankle_std_m_min": WALK_ANKLE_STD_MIN,
        },
        "daily": daily,
        "walk_files": sorted(walk_names),
    }
    out_json = sweep / "daily_walk_clips.json"
    out_json.write_text(json.dumps(listing, indent=2))

    cache: Dict[str, Optional[Dict[str, Dict]]] = {}

    def run(stem: str) -> Optional[Dict[str, Dict]]:
        if stem not in cache:
            cache[stem] = load_run(sweep / f"{stem}.json")
        return cache[stem]

    def pick(stem: str, names: Sequence[str]) -> Optional[Dict[str, float]]:
        rows = run(stem)
        if rows is None:
            return None
        chosen = [clip_row(rows[n]) for n in names if n in rows]
        return aggregate(chosen) if chosen else None

    labels = {k: lab for k, lab, _ in TRACKERS_CLEAN}
    clean_stem = {k: stem for k, _, stem in TRACKERS_CLEAN}
    own_stem = {k: stem for k, _, stem in TRACKERS_OWN}
    keys = [k for k, _, _ in TRACKERS_CLEAN]
    daily_list = sorted(daily_names)
    walk_list = sorted(walk_names)

    print(f"sweep: {sweep}")
    print(
        f"Daily clips: {len(daily_list)}    walk-like: {len(walk_list)}  "
        f"(travel≥{WALK_TRAVEL_MIN}m, {WALK_SPEED_MIN}–{WALK_SPEED_MAX} m/s, "
        f"single-support≥{WALK_SINGLE_MIN})"
    )
    print("walk files: " + ", ".join(walk_list))
    print(
        "columns: succ | HS mean | HS of each tracker's own best 8 | "
        "foot-contact acc vs GT | median joint jerk | median action jerk | mean |root yaw| (rad)"
    )

    table(
        "1. All Daily (39)",
        "Closest category to 'simple real-robot motions', still mixes walk and in-place.",
        keys,
        labels,
        [
            ("clean", lambda t: pick(clean_stem[t], daily_list)),
            ("own live ×1", lambda t: pick(own_stem[t], daily_list)),
        ],
    )
    table(
        "2. Walk-like Daily (locomotion gate on the reference)",
        "The robot actually goes somewhere with a gait. This is the walking set.",
        keys,
        labels,
        [
            ("clean", lambda t: pick(clean_stem[t], walk_list)),
            ("own live ×1", lambda t: pick(own_stem[t], walk_list)),
        ],
    )

    shared = [
        n
        for n in walk_list
        if all(
            (run(own_stem[t]) or {}).get(n, {}).get("length_ratio", 0) >= 1.0
            for t in keys
        )
    ]
    table(
        "3. Walk-like, all four succeed on their own live input",
        f"Same clips for everyone (n={len(shared)}). Quality, not who falls less.",
        keys,
        labels,
        [("own live ×1", lambda t, s=shared: pick(own_stem[t], s))],
    )

    print("\n4. Per-clip walk table (own live ×1)  HS / foot-acc / joint-jerk / ok")
    print(
        f"{'clip':18} {'spd':>4} {'m':>4}"
        + "".join(f"  {labels[t]:>22}" for t in keys)
    )
    feats = {d["file_name"]: d for d in walk}
    for n in walk_list:
        f = feats[n]
        bits = [f"{n:18} {f['speed_mps']:4.2f} {f['travel_m']:4.1f}"]
        for t in keys:
            rows = run(own_stem[t]) or {}
            r = rows.get(n)
            if r is None:
                bits.append(f"{'--':>22}")
                continue
            ok = "Y" if r["length_ratio"] >= 1.0 else "N"
            bits.append(
                f"{100*r['score']:5.1f}/{r['foot_contact_acc']:.2f}/{r['joint_jerk_mean']:5.0f}/{ok}"
            )
        print(" ".join(bits))
    print()


if __name__ == "__main__":
    main()
