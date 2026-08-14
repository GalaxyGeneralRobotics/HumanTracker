#!/usr/bin/env python3
"""Build 6,000 uniformly sampled one-comparison motion clips."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


TRACKERS = ("gmt", "hgpt", "sonic", "twist2")
CATEGORIES = ("HighlyDynamic", "Daily", "Ground", "Interaction")
PAIR_COUNT = 6000
SEED = 20260724


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniformly sample one balanced tracker pair per aligned clip"
    )
    parser.add_argument("clips_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-json", type=Path, required=True,
                        help="training manifest the clips were generated from")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="motion dataset root the manifest paths are relative to")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            yield row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object) -> str:
    value = "|".join(map(str, parts)).encode()
    return hashlib.sha256(value).hexdigest()[:20]


def uniform_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total < count:
        raise ValueError(f"cannot sample {count} unique positions from {total}")
    indices = [((2 * index + 1) * total) // (2 * count) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("uniform sampler produced duplicate positions")
    return indices


def tracker_pair_plan(count: int) -> list[tuple[tuple[str, str], bool]]:
    combinations = tuple(itertools.combinations(TRACKERS, 2))
    if count % len(combinations):
        raise ValueError(f"pair count must be divisible by {len(combinations)}")
    plan = []
    occurrences: Counter[tuple[str, str]] = Counter()
    for sample_index in range(count):
        combination = combinations[sample_index % len(combinations)]
        reverse = bool(occurrences[combination] % 2)
        occurrences[combination] += 1
        plan.append((combination, reverse))
    return plan


def load_train_index(train_json: Path, dataset_root: Path) -> dict[Path, dict[str, Any]]:
    rows = json.loads(train_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{train_json}: expected list")
    indexed: dict[Path, dict[str, Any]] = {}
    for motion_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "category", "frames"}:
            raise ValueError(f"{train_json}:{motion_index}: invalid row")
        relative_path = Path(str(row["path"]))
        category = str(row["category"])
        if category not in CATEGORIES or relative_path.parts[0] != category:
            raise ValueError(f"{train_json}:{motion_index}: invalid category/path")
        if "flipped" in relative_path.name.lower():
            continue
        source_path = (dataset_root / relative_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path in indexed:
            raise ValueError(f"duplicate source motion: {source_path}")
        indexed[source_path] = {
            "motion_idx": motion_index,
            "category": category,
            "source_motion_path": str(source_path),
        }
    return indexed


def require_clip(row: Mapping[str, Any], train_index: Mapping[Path, Mapping[str, Any]]) -> tuple:
    required = {
        "clip_id",
        "motion_id",
        "tracker",
        "category",
        "traj_path",
        "source_rollout_path",
        "source_motion_path",
        "source_start_frame",
        "source_end_frame",
        "local_start_frame",
        "local_end_frame",
        "num_frames",
        "fps",
    }
    missing = required - row.keys()
    if missing:
        raise KeyError(f"clip missing fields: {sorted(missing)}")
    tracker = str(row["tracker"])
    category = str(row["category"])
    if tracker not in TRACKERS or category not in CATEGORIES:
        raise ValueError(f"invalid tracker/category: {tracker}/{category}")
    source_path = Path(str(row["source_motion_path"])).resolve()
    if source_path not in train_index:
        raise ValueError(f"clip is not from non-flipped train split: {source_path}")
    if train_index[source_path]["category"] != category:
        raise ValueError(f"category mismatch: {source_path}")
    start = int(row["source_start_frame"])
    end = int(row["source_end_frame"])
    fps = int(row["fps"])
    frames = int(row["num_frames"])
    if start < 0 or end <= start or fps <= 0 or frames != end - start:
        raise ValueError(f"invalid clip window: {source_path}:{start}:{end}")
    for key in ("traj_path", "source_rollout_path", "source_motion_path"):
        if not Path(str(row[key])).is_file():
            raise FileNotFoundError(row[key])
    return source_path, start, end, tracker


def load_catalog(manifest: Path, train_json: Path, dataset_root: Path) -> list[dict[str, Any]]:
    train_index = load_train_index(train_json, dataset_root)
    grouped: dict[tuple[Path, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(manifest):
        source_path, start, end, tracker = require_clip(row, train_index)
        key = source_path, start, end
        if tracker in grouped[key]:
            raise ValueError(f"duplicate tracker clip: {key}/{tracker}")
        grouped[key][tracker] = row

    incomplete = [key for key, by_tracker in grouped.items() if set(by_tracker) != set(TRACKERS)]
    if incomplete:
        raise ValueError(f"{len(incomplete)} clip windows lack four trackers; first={incomplete[0]}")

    category_order = {category: index for index, category in enumerate(CATEGORIES)}
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            category_order[str(train_index[key[0]]["category"])],
            int(train_index[key[0]]["motion_idx"]),
            key[1],
            key[2],
        ),
    )
    source_clip_indices: Counter[Path] = Counter()
    catalog = []
    for clip_idx, (source_path, start, end) in enumerate(ordered_keys):
        by_tracker = grouped[(source_path, start, end)]
        first = by_tracker[TRACKERS[0]]
        shared = {
            (
                str(clip["motion_id"]),
                str(clip["category"]),
                int(clip["fps"]),
                int(clip["source_start_frame"]),
                int(clip["source_end_frame"]),
            )
            for clip in by_tracker.values()
        }
        if len(shared) != 1:
            raise ValueError(f"unaligned tracker clips: {source_path}:{start}:{end}")
        source_clip_idx = source_clip_indices[source_path]
        source_clip_indices[source_path] += 1
        catalog.append({
            "clip_idx": clip_idx,
            "clip_uid": "clip_" + stable_id(source_path, start, end),
            "source_clip_idx": source_clip_idx,
            "motion_idx": int(train_index[source_path]["motion_idx"]),
            "motion_id": str(first["motion_id"]),
            "category": str(first["category"]),
            "source_motion_path": str(source_path),
            "source_start_frame": start,
            "source_end_frame": end,
            "fps": int(first["fps"]),
            "duration_sec": (end - start) / int(first["fps"]),
            "tracker_clips": {tracker: by_tracker[tracker] for tracker in TRACKERS},
        })
    return catalog


def candidate(
    pair_id: str,
    candidate_idx: int,
    tracker: str,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    clip = selected["tracker_clips"][tracker]
    return {
        "candidate_idx": candidate_idx,
        "candidate_uid": f"{pair_id}::candidate_{candidate_idx}",
        "tracker": tracker,
        "motion_id": selected["motion_id"],
        "rollout_motion_id": str(clip["motion_id"]),
        "clip_id": str(clip["clip_id"]),
        "traj_path": str(clip["traj_path"]),
        "source_rollout_path": str(clip["source_rollout_path"]),
        "source_motion_path": selected["source_motion_path"],
        "source_start_frame": selected["source_start_frame"],
        "source_end_frame": selected["source_end_frame"],
        "local_start_frame": int(clip["local_start_frame"]),
        "local_end_frame": int(clip["local_end_frame"]),
        "num_frames": int(clip["num_frames"]),
        "fps": int(clip["fps"]),
    }


def build_pairs(catalog: list[dict[str, Any]], run_id: str) -> tuple[list[dict], list[dict]]:
    indices = uniform_indices(len(catalog), PAIR_COUNT)
    plan = tracker_pair_plan(PAIR_COUNT)
    selected_rows = []
    pairs = []
    for sample_idx, (clip_idx, (combination, reverse)) in enumerate(zip(indices, plan)):
        selected = catalog[clip_idx]
        tracker_a, tracker_b = combination
        combo_key = f"{tracker_a}|{tracker_b}"
        pair_id = "pair_" + stable_id(selected["clip_uid"], combo_key)
        order = (tracker_b, tracker_a) if reverse else (tracker_a, tracker_b)
        candidates = [
            candidate(pair_id, candidate_idx, tracker, selected)
            for candidate_idx, tracker in enumerate(order)
        ]
        selected_rows.append({
            "sample_idx": sample_idx,
            "tracker_pair_key": combo_key,
            **{key: value for key, value in selected.items() if key != "tracker_clips"},
        })
        pairs.append({
            "pair_idx": -1,
            "pair_id": pair_id,
            "motion_idx": selected["motion_idx"],
            "motion_id": selected["motion_id"],
            "clip_idx": selected["clip_idx"],
            "clip_uid": selected["clip_uid"],
            "source_clip_idx": selected["source_clip_idx"],
            "category": selected["category"],
            "split": "train",
            "tracker_pair_key": combo_key,
            "tracker_pair": list(combination),
            "candidate_count": 2,
            "candidates": candidates,
            "source_motion_path": selected["source_motion_path"],
            "source_start_frame": selected["source_start_frame"],
            "source_end_frame": selected["source_end_frame"],
            "fps": selected["fps"],
            "duration_sec": selected["duration_sec"],
            "annotation_status": "pending",
            "preferred_candidate_idx": None,
            "run_id": run_id,
            "seed": SEED,
        })

    random.Random(SEED).shuffle(pairs)
    for pair_idx, pair in enumerate(pairs):
        pair["pair_idx"] = pair_idx
    return selected_rows, pairs


def validate(catalog: list[dict], selected: list[dict], pairs: list[dict]) -> dict[str, Any]:
    combinations = tuple("|".join(pair) for pair in itertools.combinations(TRACKERS, 2))
    combination_counts = Counter(pair["tracker_pair_key"] for pair in pairs)
    tracker_counts: Counter[str] = Counter()
    tracker_positions: dict[str, Counter[int]] = defaultdict(Counter)
    for pair in pairs:
        if [candidate_row["candidate_idx"] for candidate_row in pair["candidates"]] != [0, 1]:
            raise ValueError(f"{pair['pair_id']}: invalid candidate indices")
        for candidate_row in pair["candidates"]:
            tracker = candidate_row["tracker"]
            tracker_counts[tracker] += 1
            tracker_positions[tracker][candidate_row["candidate_idx"]] += 1

    selected_indices = [row["clip_idx"] for row in selected]
    gaps = [right - left for left, right in zip(selected_indices, selected_indices[1:])]
    checks = {
        "pair_count": len(pairs) == PAIR_COUNT,
        "pair_indices_consecutive": [pair["pair_idx"] for pair in pairs] == list(range(PAIR_COUNT)),
        "pair_ids_unique": len({pair["pair_id"] for pair in pairs}) == PAIR_COUNT,
        "selected_clips_unique": len(set(selected_indices)) == PAIR_COUNT,
        "one_pair_per_selected_clip": Counter(pair["clip_idx"] for pair in pairs)
        == Counter({clip_idx: 1 for clip_idx in selected_indices}),
        "uniform_clip_sampling": max(gaps) - min(gaps) <= 1,
        "six_combinations_equal": combination_counts
        == Counter({combination: PAIR_COUNT // 6 for combination in combinations}),
        "tracker_appearances_equal": tracker_counts
        == Counter({tracker: PAIR_COUNT // 2 for tracker in TRACKERS}),
        "candidate_positions_equal": all(
            counts == Counter({0: PAIR_COUNT // 4, 1: PAIR_COUNT // 4})
            for counts in tracker_positions.values()
        ),
        "all_categories_present": set(pair["category"] for pair in pairs) == set(CATEGORIES),
        "annotation_order_shuffled": [pair["clip_idx"] for pair in pairs] != selected_indices,
    }
    if not all(checks.values()):
        raise ValueError(f"validation failed: {checks}")
    return {
        "checks": checks,
        "catalog_clip_count": len(catalog),
        "selected_clip_count": len(selected),
        "selected_motion_count": len({row["motion_id"] for row in selected}),
        "selected_category_counts": dict(Counter(row["category"] for row in selected)),
        "combination_counts": dict(combination_counts),
        "tracker_appearance_counts": dict(tracker_counts),
        "tracker_candidate_idx_counts": {
            tracker: dict(sorted(counts.items()))
            for tracker, counts in sorted(tracker_positions.items())
        },
        "clip_index_range": [selected_indices[0], selected_indices[-1]],
        "clip_index_gap_range": [min(gaps), max(gaps)],
    }


def write_parquet(path: Path, pairs: list[dict[str, Any]]) -> None:
    candidate_type = pa.struct([
        ("candidate_idx", pa.int8()),
        ("candidate_uid", pa.string()),
        ("tracker", pa.string()),
        ("motion_id", pa.string()),
        ("rollout_motion_id", pa.string()),
        ("clip_id", pa.string()),
        ("traj_path", pa.string()),
        ("source_rollout_path", pa.string()),
        ("source_motion_path", pa.string()),
        ("source_start_frame", pa.int32()),
        ("source_end_frame", pa.int32()),
        ("local_start_frame", pa.int32()),
        ("local_end_frame", pa.int32()),
        ("num_frames", pa.int32()),
        ("fps", pa.int16()),
    ])
    schema = pa.schema([
        ("pair_idx", pa.int64()),
        ("pair_id", pa.string()),
        ("motion_idx", pa.int64()),
        ("motion_id", pa.string()),
        ("clip_idx", pa.int64()),
        ("clip_uid", pa.string()),
        ("source_clip_idx", pa.int32()),
        ("category", pa.string()),
        ("split", pa.string()),
        ("tracker_pair_key", pa.string()),
        ("tracker_pair", pa.list_(pa.string(), 2)),
        ("candidate_count", pa.int8()),
        ("candidates", pa.list_(candidate_type, 2)),
        ("source_motion_path", pa.string()),
        ("source_start_frame", pa.int32()),
        ("source_end_frame", pa.int32()),
        ("fps", pa.int16()),
        ("duration_sec", pa.float32()),
        ("annotation_status", pa.string()),
        ("preferred_candidate_idx", pa.int8()),
        ("run_id", pa.string()),
        ("seed", pa.int64()),
    ])
    pq.write_table(pa.Table.from_pylist(pairs, schema=schema), path, compression="zstd")


def main() -> None:
    args = parse_args()
    manifest = args.clips_manifest.resolve()
    output_dir = args.output_dir.resolve()
    train_json = args.train_json.resolve()
    dataset_root = args.dataset_root.resolve()
    if not train_json.is_file():
        raise FileNotFoundError(train_json)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    catalog = load_catalog(manifest, train_json, dataset_root)
    selected, pairs = build_pairs(catalog, output_dir.name)
    validation = validate(catalog, selected, pairs)

    catalog_path = output_dir / "clip_catalog.jsonl"
    selected_path = output_dir / "selected_clips.jsonl"
    pairs_path = output_dir / "pairs.jsonl"
    parquet_path = output_dir / "pairs.parquet"
    write_jsonl(catalog_path, catalog)
    write_jsonl(selected_path, selected)
    write_jsonl(pairs_path, pairs)
    write_parquet(parquet_path, pairs)

    summary = {
        "seed": SEED,
        "inputs": {
            "train_json": str(train_json),
            "train_json_sha256": file_hash(train_json),
            "clips_manifest": str(manifest),
            "clips_manifest_sha256": file_hash(manifest),
            "source_policy": "non_flipped_train",
        },
        "selection": {
            "method": "midpoint_uniform_over_global_clip_index",
            "category_order": list(CATEGORIES),
            **validation,
        },
        "pairing": {
            "method": "one_unordered_tracker_combination_per_selected_clip",
            "trackers": list(TRACKERS),
            "target_pairs": PAIR_COUNT,
        },
        "outputs": {
            "clip_catalog": str(catalog_path),
            "clip_catalog_sha256": file_hash(catalog_path),
            "selected_clips": str(selected_path),
            "selected_clips_sha256": file_hash(selected_path),
            "pairs_jsonl": str(pairs_path),
            "pairs_jsonl_sha256": file_hash(pairs_path),
            "pairs_parquet": str(parquet_path),
            "pairs_parquet_sha256": file_hash(parquet_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
