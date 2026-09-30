"""Build stratified walk / Daily manifests from the official test set.

The 100-clip sweep only kept 39 Daily (17 walk-like). Official test.json has
974 Daily files. This script classifies them with the same reference-side walk
gate as ``daily_walk_report.py`` and writes:

* ``outputs/eval_runs/all_daily_walk_inventory.json`` — every Daily clip
* ``outputs/eval_runs/walk_expand_clips.json`` — eval manifest (walk + contrast)
* ``outputs/eval_runs/walk_viz_clips.json`` — short video subset
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MOTION = REPO / "storage" / "dataset" / "HumanTracker" / "motions"
OLD_WALK = REPO / "outputs" / "eval_runs" / "noise_sweep_deploy" / "daily_walk_clips.json"

WALK_SPEED_MIN = 0.25
WALK_SPEED_MAX = 1.60
WALK_TRAVEL_MIN = 1.50
WALK_SINGLE_MIN = 0.20
WALK_DUTY = (0.25, 0.85)
WALK_YAW_MAX = 1.20
WALK_ANKLE_STD_MIN = 0.02

# Hard cap so four trackers × clean + own-live stay runnable on a laptop.
# All original 17 walk clips are always kept; extra slots are stratified by speed.
WALK_CAP = 80
CONTRAST_CAP = 24
SPEED_BINS = ((0.25, 0.50), (0.50, 0.80), (0.80, 1.20), (1.20, 1.60))

# Canonical clips for side-by-side video (must exist in Daily).
VIZ_PREFERRED = [
    "Daily_198.npz",  # 0.85 m/s; BFM walks, HGPT 216 falls, HS anti-aligned
    "Daily_121.npz",
    "Daily_130.npz",
    "Daily_136.npz",
    "Daily_401.npz",  # BFM5 fail on 17-set
    "Daily_613.npz",  # BFM14 fail on 17-set
    "Daily_490.npz",
    "Daily_823.npz",
]


def walk_features(npz_path: Path) -> Dict[str, float]:
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


def classify_daily() -> List[Dict]:
    test = json.loads((MOTION / "test.json").read_text())
    rows = []
    daily = [e for e in test if e.get("category") == "Daily"]
    for i, e in enumerate(daily):
        feat = walk_features(MOTION / e["path"])
        feat["path"] = e["path"]
        feat["file_name"] = Path(e["path"]).name
        feat["walk"] = is_walk(feat)
        rows.append(feat)
        if (i + 1) % 200 == 0:
            print(f"classified {i + 1}/{len(daily)}", flush=True)
    return rows


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def pick_walk(walk: List[Dict], keep: set[str]) -> List[Dict]:
    by_name = {r["file_name"]: r for r in walk}
    chosen: Dict[str, Dict] = {n: by_name[n] for n in keep if n in by_name}
    remaining = [r for r in walk if r["file_name"] not in chosen]
    if len(chosen) >= WALK_CAP:
        return [chosen[n] for n in sorted(chosen)]
    slots = WALK_CAP - len(chosen)
    per_bin = max(1, slots // len(SPEED_BINS))
    rng = _rng()
    extras: List[Dict] = []
    leftover: List[Dict] = []
    for lo, hi in SPEED_BINS:
        bucket = [r for r in remaining if lo <= r["speed_mps"] < hi]
        leftover.extend(bucket)
        if not bucket:
            continue
        take = min(per_bin, len(bucket), slots - len(extras))
        idx = rng.choice(len(bucket), size=take, replace=False)
        extras.extend(bucket[i] for i in sorted(idx.tolist()))
    if len(chosen) + len(extras) < WALK_CAP:
        used = {r["file_name"] for r in extras} | set(chosen)
        pool = [r for r in leftover if r["file_name"] not in used]
        need = WALK_CAP - len(chosen) - len(extras)
        if pool and need > 0:
            take = min(need, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            extras.extend(pool[i] for i in sorted(idx.tolist()))
    for r in extras:
        chosen[r["file_name"]] = r
    return [chosen[n] for n in sorted(chosen)]


def pick_contrast(daily: List[Dict], used: set[str]) -> List[Dict]:
    pool = [r for r in daily if not r["walk"] and r["file_name"] not in used]
    if not pool:
        return []
    rng = _rng()
    take = min(CONTRAST_CAP, len(pool))
    # Prefer in-place / slow Daily — the HS-high non-walks.
    pool = sorted(pool, key=lambda r: (r["travel_m"], r["speed_mps"]))
    short = pool[: max(take * 2, take)]
    idx = rng.choice(len(short), size=min(take, len(short)), replace=False)
    return [short[i] for i in sorted(idx.tolist())]


def as_manifest(rows: List[Dict]) -> List[Dict]:
    return [
        {"path": r["path"], "category": "Daily", "frames": int(r["frames"])}
        for r in sorted(rows, key=lambda r: r["file_name"])
    ]


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs" / "eval_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = classify_daily()
    walk = [r for r in daily if r["walk"]]
    keep: set[str] = set()
    if OLD_WALK.is_file():
        keep = set(json.loads(OLD_WALK.read_text())["walk_files"])

    selected_walk = pick_walk(walk, keep)
    contrast = pick_contrast(daily, {r["file_name"] for r in selected_walk})
    expand = selected_walk + contrast

    inventory = {
        "gate": {
            "speed_mps": [WALK_SPEED_MIN, WALK_SPEED_MAX],
            "travel_m_min": WALK_TRAVEL_MIN,
            "single_support_min": WALK_SINGLE_MIN,
            "duty": list(WALK_DUTY),
            "yaw_rate_max": WALK_YAW_MAX,
            "ankle_std_m_min": WALK_ANKLE_STD_MIN,
        },
        "n_daily": len(daily),
        "n_walk": len(walk),
        "n_expand_walk": len(selected_walk),
        "n_expand_contrast": len(contrast),
        "speed_bins": {
            f"{lo}-{hi}": sum(1 for r in walk if lo <= r["speed_mps"] < hi)
            for lo, hi in SPEED_BINS
        },
        "daily": daily,
    }
    (out_dir / "all_daily_walk_inventory.json").write_text(json.dumps(inventory, indent=2))
    (out_dir / "walk_expand_clips.json").write_text(json.dumps(as_manifest(expand), indent=2))

    viz_names = [n for n in VIZ_PREFERRED if any(r["file_name"] == n for r in daily)]
    viz_rows = [r for r in daily if r["file_name"] in set(viz_names)]
    (out_dir / "walk_viz_clips.json").write_text(json.dumps(as_manifest(viz_rows), indent=2))

    print(
        f"Daily={len(daily)} walk={len(walk)} expand_walk={len(selected_walk)} "
        f"contrast={len(contrast)} viz={len(viz_rows)}"
    )
    print("walk bins (all / selected):")
    sel_names = {r["file_name"] for r in selected_walk}
    for lo, hi in SPEED_BINS:
        n_all = sum(1 for r in walk if lo <= r["speed_mps"] < hi)
        n_sel = sum(1 for r in selected_walk if lo <= r["speed_mps"] < hi)
        print(f"  {lo:.2f}-{hi:.2f}: {n_all} / {n_sel}")
    print("kept original walk:", len(keep & sel_names), "/", len(keep))
    print(Counter("walk" if r in sel_names else "contrast" for r in [x["file_name"] for x in expand]))


if __name__ == "__main__":
    main()
