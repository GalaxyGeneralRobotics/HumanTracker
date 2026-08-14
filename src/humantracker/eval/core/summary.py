"""The two summary tables every tracker reports, computed and printed in one place.

Each backend carried its own copy of these functions, and the copies had drifted: one dropped
non-finite reward-model scores while the others let them poison the mean, one derived the
failure count from the validity test while the others re-scanned for it, one printed the
failure line and another dropped it. A benchmark whose numbers depend on which backend
formatted them is not comparing trackers, so the tables live here.

A trajectory is *valid* when its mean joint position error is finite: a rollout that produced
no comparable frames reports ``inf`` rather than raising. MPJPE is weighted by each valid
trajectory's frame count; other trajectory metrics are averaged over valid rows, and the rest
are counted in ``error_count``. HumanScore is likewise weighted by the number of frames scored.
A metric the tracker does not record is absent from the summary rather than present and null.

Backends that log rather than print pass their own ``sink``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "compute_category_summary",
    "compute_overall_summary",
    "print_category_summary",
    "print_overall_summary",
]

Sink = Callable[[str], None]
Metrics = Sequence[Mapping[str, Any]]

# What a trajectory reports for its joint error when it produced no comparable frames.
FAILED = float("inf")

RULE = "=" * 60

# Averaged over the trajectories that report them; a tracker that records neither smoothness
# nor foot contact leaves these keys out of the summary altogether.
OPTIONAL_METRICS = (
    "joint_acc_mean",
    "joint_jerk_mean",
    "joint_acc_rms",
    "joint_jerk_rms",
    "action_jerk_mean",
    "foot_contact_acc",
    "foot_contact_iou",
)

# The subset of the above that the tables print, with each row's label and unit.
PRINTED_METRICS = (
    ("Joint Acc (mean):", "joint_acc_mean", " rad/s^2"),
    ("Joint Jerk (mean):", "joint_jerk_mean", " rad/s^3"),
    ("Action Jerk:", "action_jerk_mean", " rad/s^3"),
    ("Foot Contact Acc:", "foot_contact_acc", ""),
    ("Foot Contact IoU:", "foot_contact_iou", ""),
)

# Where a row's value starts, measured from the first character of its label.
CATEGORY_WIDTH = 18
OVERALL_WIDTH = 22


def _valid(metrics: Metrics) -> List[Mapping[str, Any]]:
    """The trajectories that produced comparable frames.

    Every backend writes ``joint_pos_mae`` for every trajectory, ``FAILED`` when the run
    ended before one frame could be compared against the reference. A row without the key
    is a backend bug and should surface as one rather than be dropped from the average.
    """
    return [
        m
        for m in metrics
        if m["joint_pos_mae"] < FAILED and m.get("nonfinite_frame") is None
    ]


def _mean(metrics: Metrics, key: str) -> float:
    """Mean over every row. A row missing ``key`` is a backend bug, not a gap to skip."""
    return float(np.mean([m[key] for m in metrics]))


def _weighted_mean(metrics: Metrics, key: str, weight_key: str) -> float:
    """Weighted mean reconstructed from each trajectory's mean and sample count."""
    frames = [m[weight_key] for m in metrics]
    if any(type(count) is not int or count < 1 for count in frames):
        raise ValueError(f"{weight_key} must contain positive integers")
    return float(
        sum(m[key] * count for m, count in zip(metrics, frames))
        / sum(frames)
    )


def _finite_mean(metrics: Metrics, key: str) -> Optional[float]:
    """Mean over the rows reporting a finite value, or ``None`` when none do."""
    values = [m[key] for m in metrics if m.get(key) is not None and np.isfinite(m[key])]
    return float(np.mean(values)) if values else None


def _records_keypoints(valid: Metrics) -> bool:
    """Whether keypoint errors were measured.

    Read from the first valid trajectory: the reference keypoint poses either come with the
    dataset, in which case every trajectory has them, or they do not. The key itself is
    always written, ``FAILED`` when they were not measured.
    """
    return valid[0]["kpt_pos_mae"] < FAILED


def _summary(metrics: Metrics, category: Optional[str]) -> Dict[str, Any]:
    """Build one summary. ``category`` labels it; ``None`` makes it the whole run's."""
    valid = _valid(metrics)
    total = len(metrics)
    summary: Dict[str, Any] = {"total": total, "valid": len(valid)}
    if category is not None:
        summary = {"category": category, **summary}
    if not valid:
        return summary

    success = sum(1 for m in valid if m["length_ratio"] >= 1.0)
    summary.update(
        {
            "success_rate": success / total,
            "success_count": success,
            "avg_traj_length": _mean(valid, "length_ratio"),
            "MPJPE": _weighted_mean(valid, "joint_pos_mae", "total_frames"),
            "MPJVE": _mean(valid, "joint_vel_mae"),
            "root_pos_err_mm": _mean(valid, "root_pos_err_mm"),
            "root_vel_err_mms": _mean(valid, "root_vel_err_mms"),
            "root_yaw_err": _mean(valid, "root_yaw_err"),
        }
    )
    if _records_keypoints(valid):
        summary["kpt_pos_mae"] = _mean(valid, "kpt_pos_mae")
        summary["kpt_rot_mae"] = _mean(valid, "kpt_rot_mae")
    for key in OPTIONAL_METRICS:
        value = _finite_mean(valid, key)
        if value is not None:
            summary[key] = value
    score_rows = [m for m in valid if m.get("score") is not None and np.isfinite(m["score"])]
    summary["score"] = (
        _weighted_mean(score_rows, "score", "score_frames") if score_rows else None
    )
    if category is None:
        summary["error_count"] = total - len(valid)
    return summary


def compute_category_summary(category: str, metrics: Metrics) -> Dict[str, Any]:
    """Summarise one motion category, carrying its name through into the result."""
    return _summary(metrics, category)


def compute_overall_summary(metrics: Metrics) -> Dict[str, Any]:
    """Summarise a whole run, adding the number of trajectories that produced no metrics."""
    return _summary(metrics, None)


def _report(
    metrics: Metrics,
    valid: Metrics,
    sink: Sink,
    *,
    heading: Sequence[str],
    empty: str,
    width: int,
    alert: Optional[Sink],
) -> None:
    """Print one table. ``alert`` is ``None`` for a category table, which leaves the failure
    count to the overall one."""
    total = len(metrics)
    if not valid:
        sink(empty)
        return

    def row(label: str, value: str, extra: int = 0) -> None:
        sink(f"  {label:<{width + extra}}{value}")

    for line in heading:
        sink(line)
    success = sum(1 for m in valid if m["length_ratio"] >= 1.0)
    row("Success Rate:", f"{success / total:.4f}  ({success}/{total})")
    row("Avg Traj Length:", f"{_mean(valid, 'length_ratio'):.4f}")
    row("MPJPE:", f"{_weighted_mean(valid, 'joint_pos_mae', 'total_frames'):.6f} rad")
    row("MPJVE:", f"{_mean(valid, 'joint_vel_mae'):.6f} rad/s")
    if _records_keypoints(valid):
        row("KPT Pos MAE:", f"{_mean(valid, 'kpt_pos_mae'):.6f} m")
        row("KPT Rot MAE:", f"{_mean(valid, 'kpt_rot_mae'):.6f} rad")
    row("Root Pos Err:", f"{_mean(valid, 'root_pos_err_mm'):.2f} mm")
    row("Root Vel Err:", f"{_mean(valid, 'root_vel_err_mms'):.2f} mm/s")
    row("Root Yaw Err:", f"{_mean(valid, 'root_yaw_err'):.6f} rad")
    for label, key, unit in PRINTED_METRICS:
        value = _finite_mean(valid, key)
        if value is not None:
            row(label, f"{value:.4f}{unit}")
    score_rows = [
        m
        for m in valid
        if m.get("score") is not None and np.isfinite(m["score"])
    ]
    score = _weighted_mean(score_rows, "score", "score_frames") if score_rows else None
    if score is None:
        row("Score:", "N/A")
    else:
        row("Score (frame mean):", f"{score:.4f}")
    if alert is not None and total > len(valid):
        alert(f"\n  {total - len(valid)} trajectories had errors.")


def print_category_summary(category: str, metrics: Metrics, *, sink: Sink = print) -> None:
    """Print one category's table."""
    _report(
        metrics,
        _valid(metrics),
        sink,
        heading=(f"\n{RULE}", f"  Category: {category}  ({len(metrics)} trajectories)", RULE),
        empty=f"\n=== {category}: {len(metrics)} trajs, all failed ===",
        width=CATEGORY_WIDTH,
        alert=None,
    )


def print_overall_summary(
    metrics: Metrics,
    *,
    sink: Sink = print,
    alert: Optional[Sink] = None,
) -> None:
    """Print the run's table. ``alert`` takes the failure count and defaults to ``sink``; a
    backend that logs passes its warning level so that failures keep it.
    """
    valid = _valid(metrics)
    _report(
        metrics,
        valid,
        sink,
        heading=(
            f"\n{RULE}",
            f"  OVERALL SUMMARY  ({len(metrics)} trajectories, {len(valid)} valid)",
            RULE,
        ),
        empty="\nNo valid results.",
        width=OVERALL_WIDTH,
        alert=sink if alert is None else alert,
    )
