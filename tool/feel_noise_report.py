"""Feel-metric + noise report on an expand (or deploy) sweep directory.

Prints walk vs contrast slices, pairwise wins between the Humanoid-GPT
policies (the robot favours HGPT216 slightly over HGPT264), and noise-scale
curves for GT-free quantities.

Usage:
  python tool/feel_noise_report.py [sweep_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "outputs" / "eval_runs" / "all_daily_walk_inventory.json"

TRACKERS = [
    ("hgpt264", "HGPT 264"),
    ("hgpt216", "HGPT 216"),
]
OWN = {
    "hgpt264": "hgpt264_gmr_stream_s{s}_filtered",
    "hgpt216": "hgpt216_gmr_stream_s{s}_filtered",
}
CLEAN = {k: f"{k}_clean" for k, _ in TRACKERS}
LOWER_BETTER = {
    "ajerk", "ajerk_med", "ajerk_p95", "hf", "jerk", "sway", "bounce",
    "roll", "pitch", "heading", "duty_asym", "step_std", "mpjpe",
}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def load_run(path: Path) -> Optional[Dict[str, Dict]]:
    if not path.is_file():
        return None
    return {r["file_name"]: r for r in json.loads(path.read_text())["per_trajectory"]}


def _num(value, scale: float = 1.0) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if v != v or v in (float("inf"), float("-inf")):
        return float("nan")
    return scale * v


def clip_row(r: Dict) -> Dict[str, float]:
    return {
        "ok": 1.0 if (r.get("length_ratio") or 0) >= 1.0 else 0.0,
        "hs": _num(r.get("score"), 100.0),
        "mpjpe": _num(r.get("kpt_pos_mae"), 1000.0),
        "ajerk": _num(r.get("action_jerk_mean")),
        "ajerk_med": _num(r.get("action_jerk_med")),
        "ajerk_p95": _num(r.get("action_jerk_p95")),
        "hf": _num(r.get("action_hf_ratio")),
        "jerk": _num(r.get("joint_jerk_mean")),
        "sway": _num(r.get("root_lat_sway_m")),
        "bounce": _num(r.get("root_bounce_m")),
        "roll": _num(r.get("root_roll_std")),
        "pitch": _num(r.get("root_pitch_std")),
        "heading": _num(r.get("heading_rate_abs")),
        "duty_asym": _num(r.get("sim_duty_asym")),
        "single": _num(r.get("sim_single_support")),
        "step_std": _num(r.get("step_period_std_s")),
        "cadence": _num(r.get("cadence_hz")),
        "fc_acc": _num(r.get("foot_contact_acc")),
    }


def finite(xs: Sequence[float]) -> List[float]:
    return [v for v in xs if v == v and v not in (float("inf"), float("-inf"))]


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    def col(key: str) -> List[float]:
        return finite(r[key] for r in rows)

    ok = [r["ok"] for r in rows]
    finished = [r for r in rows if r["ok"] >= 1.0]
    out = {"n": float(len(rows)), "succ": 100.0 * _mean(ok)}
    for key in (
        "hs", "mpjpe", "ajerk", "ajerk_med", "ajerk_p95", "hf", "jerk",
        "sway", "bounce", "roll", "pitch", "heading", "duty_asym",
        "single", "step_std", "cadence", "fc_acc",
    ):
        vals = finite(r[key] for r in finished) if key != "hs" else col(key)
        out[key] = _median(vals) if key in (
            "ajerk", "ajerk_med", "ajerk_p95", "jerk", "hf"
        ) else _mean(vals)
        out[f"{key}_n"] = float(len(vals))
    return out


def pairwise_wins(
    a_rows: Dict[str, Dict],
    b_rows: Dict[str, Dict],
    names: Sequence[str],
    key: str,
    lower_better: bool,
    both_ok: bool,
) -> Tuple[int, int, float]:
    wins = 0
    n = 0
    deltas = []
    for name in names:
        if name not in a_rows or name not in b_rows:
            continue
        ra, rb = clip_row(a_rows[name]), clip_row(b_rows[name])
        if both_ok and (ra["ok"] < 1.0 or rb["ok"] < 1.0):
            continue
        va, vb = ra[key], rb[key]
        if va != va or vb != vb:
            continue
        n += 1
        delta = (vb - va) if lower_better else (va - vb)
        deltas.append(delta)
        if (va < vb) if lower_better else (va > vb):
            wins += 1
    return wins, n, _mean(deltas)


def fmt(a: Optional[Dict[str, float]], keys: Sequence[str]) -> str:
    if a is None:
        return "--"
    bits = []
    for k in keys:
        v = a[k]
        if k == "succ":
            bits.append(f"{v:4.0f}%")
        elif k in ("hs", "mpjpe"):
            bits.append(f"{v:5.1f}")
        elif k in ("ajerk", "ajerk_med", "ajerk_p95", "jerk"):
            bits.append(f"{v:6.0f}")
        else:
            bits.append(f"{v:5.3f}")
    return " ".join(bits)


def main() -> None:
    sweep = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_runs/noise_sweep_expand")
    inv = json.loads(INVENTORY.read_text()) if INVENTORY.is_file() else {"daily": []}
    walk = {r["file_name"] for r in inv.get("daily", []) if r.get("walk")}
    contrast = {r["file_name"] for r in inv.get("daily", []) if not r.get("walk")}

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

    # Discover names actually present.
    sample = None
    for t, _ in TRACKERS:
        sample = run(OWN[t].format(s="1.0")) or run(CLEAN[t])
        if sample:
            break
    if sample is None:
        print(f"no runs in {sweep}")
        return
    all_names = sorted(sample)
    walk_names = [n for n in all_names if n in walk] or all_names
    contrast_names = [n for n in all_names if n in contrast]

    print(f"sweep: {sweep}")
    print(f"clips in run: {len(all_names)}  walk-tagged: {len(walk_names)}  contrast: {len(contrast_names)}")
    print(
        "columns: succ | HS | MPJPE | aJerk | aJerk_med | hf | sway | bounce | "
        "duty_asym | step_std | cadence"
    )
    keys = ["succ", "hs", "mpjpe", "ajerk", "ajerk_med", "hf", "sway", "bounce",
            "duty_asym", "step_std", "cadence"]

    def table(title: str, names: Sequence[str], scale: str = "1.0", clean: bool = False) -> Dict[str, Dict[str, float]]:
        print(f"\n{title}  n={len(names)}")
        got = {}
        for t, lab in TRACKERS:
            stem = CLEAN[t] if clean else OWN[t].format(s=scale)
            agg = pick(stem, names)
            if agg:
                got[t] = agg
            print(f"  {lab:16}  {fmt(agg, keys)}")
        return got

    own_walk = table("1. Own-live ×1 · walk", walk_names)
    table("2. Own-live ×1 · contrast Daily", contrast_names)
    table("3. Clean · walk", walk_names, clean=True)

    print("\n4. Pairwise both-ok wins (walk, own-live ×1)  A vs B : wins/n  meanΔ")
    pairs = [("hgpt216", "hgpt264")]
    for key, lower in (("ajerk", True), ("hf", True), ("sway", True),
                       ("duty_asym", True), ("step_std", True), ("hs", False), ("ok", False)):
        print(f"  [{key}]")
        for a, b in pairs:
            ra, rb = run(OWN[a].format(s="1.0")), run(OWN[b].format(s="1.0"))
            if not ra or not rb:
                continue
            w, n, d = pairwise_wins(ra, rb, walk_names, key, lower, both_ok=(key != "ok"))
            print(f"    {a} vs {b}: {w}/{n}  Δ={d:.4f}")

    print("\n5. Noise scale (own profile) · walk succ / aJerk / sway / hf")
    for t, lab in TRACKERS:
        bits = [f"  {lab:16}"]
        for s in ("0.5", "1.0", "2.0"):
            agg = pick(OWN[t].format(s=s), walk_names)
            if agg is None:
                bits.append(f"  s{s}: --")
            else:
                bits.append(
                    f"  s{s}: {agg['succ']:4.0f}% aJ={agg['ajerk']:6.0f} "
                    f"sw={agg['sway']:.3f} hf={agg['hf']:.3f}"
                )
        print("".join(bits))

    payload = {
        "sweep": str(sweep),
        "n_run": len(all_names),
        "n_walk": len(walk_names),
        "walk_files": walk_names,
        "own_walk": own_walk,
    }
    (sweep / "feel_noise_report.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {sweep / 'feel_noise_report.json'}")


if __name__ == "__main__":
    main()
