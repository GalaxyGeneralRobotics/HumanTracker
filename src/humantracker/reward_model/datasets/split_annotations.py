#!/usr/bin/env python3
"""Create an exact, motion-disjoint 80/20 split for preference annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from humantracker.reward_model.datasets.parquet_reader import load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def record_features(annotation: dict) -> tuple[tuple[str, ...], ...]:
    length = clip_length(annotation)
    return (
        ("category", str(annotation["category"])),
        (
            "joint",
            str(annotation["category"]),
            str(annotation["tracker_pair_key"]),
            str(annotation["preference"]["choice_type"]),
        ),
        ("annotator", str(annotation["meta"]["annotator_id"])),
        ("length", "full" if length == 250 else "short"),
    )


def clip_length(annotation: dict) -> int:
    lengths = {
        int(candidate["end_frame"]) - int(candidate["start_frame"])
        for candidate in annotation["candidates"]
    }
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError(f"{annotation['record_id']}: invalid candidate lengths {lengths}")
    return next(iter(lengths))


def select_groups(annotations: list[dict], target_size: int, seed: int) -> set[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        motion_id = str(annotation["motion_id"])
        if not motion_id:
            raise ValueError("motion_id is required")
        grouped[motion_id].append(annotation)

    local_ratio = target_size / len(annotations)
    feature_totals = Counter(
        feature for annotation in annotations for feature in record_features(annotation)
    )
    feature_targets = {
        feature: count * local_ratio for feature, count in feature_totals.items()
    }
    groups = []
    for motion_id, records in sorted(grouped.items()):
        features = Counter(
            feature for annotation in records for feature in record_features(annotation)
        )
        features[("group_size", str(len(records)))] = len(records)
        groups.append((motion_id, len(records), features))
    for _, size, _ in groups:
        feature_totals[("group_size", str(size))] += size
    for feature, count in feature_totals.items():
        feature_targets.setdefault(feature, count * local_ratio)

    rng = random.Random(seed)
    random_rank = {motion_id: rng.random() for motion_id, _, _ in groups}
    selected: set[str] = set()
    selected_size = 0
    selected_features: Counter = Counter()
    remaining = list(groups)

    def delta(size: int, features: Counter) -> float:
        size_change = (
            (selected_size + size - target_size) ** 2
            - (selected_size - target_size) ** 2
        ) / target_size
        feature_change = 0.0
        for feature, increment in features.items():
            target = feature_targets[feature]
            current = selected_features[feature]
            feature_change += (
                (current + increment - target) ** 2 - (current - target) ** 2
            ) / target
        return size_change + feature_change

    while selected_size < target_size:
        capacity = target_size - selected_size
        eligible = [group for group in remaining if group[1] <= capacity]
        if not eligible:
            raise RuntimeError(f"cannot reach exact test size {target_size}")
        chosen = min(
            eligible,
            key=lambda group: (
                delta(group[1], group[2]),
                random_rank[group[0]],
                group[0],
            ),
        )
        selected.add(chosen[0])
        selected_size += chosen[1]
        selected_features.update(chosen[2])
        remaining.remove(chosen)
    return selected


def select_test_groups(
    annotations: list[dict], test_ratio: float, seed: int
) -> set[str]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between zero and one")
    by_category: dict[str, list[dict]] = defaultdict(list)
    motion_categories: dict[str, set[str]] = defaultdict(set)
    for annotation in annotations:
        category = str(annotation["category"])
        by_category[category].append(annotation)
        motion_categories[str(annotation["motion_id"])].add(category)
    if any(len(categories) != 1 for categories in motion_categories.values()):
        raise ValueError("a motion_id appears in multiple categories")

    total_target = round(len(annotations) * test_ratio)
    raw_targets = {
        category: len(records) * test_ratio
        for category, records in by_category.items()
    }
    targets = {category: int(value) for category, value in raw_targets.items()}
    remainder = total_target - sum(targets.values())
    order = sorted(raw_targets, key=lambda key: (-(raw_targets[key] % 1), key))
    for category in order[:remainder]:
        targets[category] += 1

    selected: set[str] = set()
    for offset, category in enumerate(sorted(by_category)):
        selected |= select_groups(by_category[category], targets[category], seed + offset)
    return selected


def split_stats(records: list[dict]) -> dict:
    strict_250 = [
        item
        for item in records
        if clip_length(item) == 250
        and item["preference"]["choice_type"] in {"preference", "similar"}
    ]
    return {
        "records": len(records),
        "motions": len({item["motion_id"] for item in records}),
        "choice_type": dict(
            sorted(Counter(item["preference"]["choice_type"] for item in records).items())
        ),
        "category": dict(sorted(Counter(item["category"] for item in records).items())),
        "tracker_pair": dict(
            sorted(Counter(item["tracker_pair_key"] for item in records).items())
        ),
        "annotator": dict(
            sorted(Counter(item["meta"]["annotator_id"] for item in records).items())
        ),
        "strict_250_no_mask": {
            "records": len(strict_250),
            "choice_type": dict(
                sorted(
                    Counter(
                        item["preference"]["choice_type"] for item in strict_250
                    ).items()
                )
            ),
        },
    }


def write_manifest(
    path: Path,
    split: str,
    records: list[dict],
    data_dir: Path,
    source_fingerprint: str,
    test_ratio: float,
    seed: int,
) -> None:
    manifest = {
        "split": split,
        "source_dir": str(data_dir.resolve()),
        "source_fingerprint": source_fingerprint,
        "seed": seed,
        "test_ratio": test_ratio,
        "group_key": "motion_id",
        "stratify_keys": [
            "category",
            "tracker_pair_key",
            "preference.choice_type",
            "meta.annotator_id",
            "clip_length",
            "motion_group_size",
        ],
        "stats": split_stats(records),
        "record_ids": [item["record_id"] for item in records],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    annotations = load_annotations(data_dir)
    test_groups = select_test_groups(annotations, args.test_ratio, args.seed)
    train = [item for item in annotations if item["motion_id"] not in test_groups]
    test = [item for item in annotations if item["motion_id"] in test_groups]

    if len(test) != round(len(annotations) * args.test_ratio):
        raise RuntimeError("test split does not have the requested size")
    train_groups = {item["motion_id"] for item in train}
    if train_groups & test_groups:
        raise RuntimeError("motion leakage between train and test")
    if {item["record_id"] for item in train + test} != {
        item["record_id"] for item in annotations
    }:
        raise RuntimeError("split coverage mismatch")

    fingerprint = hashlib.sha256(
        "\n".join(item["record_id"] for item in annotations).encode()
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir / "train.json",
        "train",
        train,
        data_dir,
        fingerprint,
        args.test_ratio,
        args.seed,
    )
    write_manifest(
        output_dir / "test.json",
        "test",
        test,
        data_dir,
        fingerprint,
        args.test_ratio,
        args.seed,
    )
    print(json.dumps({"train": split_stats(train), "test": split_stats(test)}, indent=2))


if __name__ == "__main__":
    main()
