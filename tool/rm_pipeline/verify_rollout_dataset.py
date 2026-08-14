#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TRACKERS = ("gmt", "hgpt", "sonic", "twist2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify full non-flipped train rollout coverage")
    parser.add_argument("train_json", type=Path)
    parser.add_argument("rollout_root", type=Path)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expected(train_json: Path) -> set[Path]:
    dataset_root = train_json.parent.resolve()
    rows = json.loads(train_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{train_json}: expected list")
    paths = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "category", "frames"}:
            raise ValueError(f"{train_json}:{row_index}: invalid row")
        relative_path = Path(str(row["path"]))
        if "flipped" in relative_path.name.lower():
            continue
        source_path = (dataset_root / relative_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path in paths:
            raise ValueError(f"duplicate train motion: {source_path}")
        paths.add(source_path)
    return paths


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    train_json = args.train_json.resolve()
    rollout_root = args.rollout_root.resolve()
    expected = load_expected(train_json)
    if len(expected) != 9858:
        raise ValueError(f"expected 9858 non-flipped motions, got {len(expected)}")

    meta_paths = sorted(rollout_root.glob("*/traj_csv/*.meta.json"))
    if len(meta_paths) != len(expected) * len(TRACKERS):
        raise ValueError(f"expected {len(expected) * len(TRACKERS)} metadata files, got {len(meta_paths)}")

    coverage: dict[str, set[Path]] = defaultdict(set)
    records = []
    for meta_path in meta_paths:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        tracker = str(record["tracker"])
        if tracker not in TRACKERS:
            raise ValueError(f"{meta_path}: invalid tracker {tracker}")
        source_path = Path(record["source_motion_path"]).resolve()
        if source_path not in expected:
            raise ValueError(f"{meta_path}: source is not non-flipped train motion")
        if source_path in coverage[tracker]:
            raise ValueError(f"duplicate rollout: {tracker}/{source_path}")
        rollout_path = Path(record["rollout_npz_path"]).resolve()
        if rollout_path != meta_path.with_suffix("").with_suffix(".npz").resolve():
            raise ValueError(f"{meta_path}: rollout path mismatch")
        if not rollout_path.is_file():
            raise FileNotFoundError(rollout_path)
        coverage[tracker].add(source_path)
        records.append(record)

    for tracker in TRACKERS:
        missing = expected - coverage[tracker]
        extra = coverage[tracker] - expected
        if missing or extra:
            raise ValueError(f"{tracker}: missing={len(missing)} extra={len(extra)}")

    manifest_path = rollout_root / "rollouts.jsonl"
    manifest = load_jsonl(manifest_path)
    manifest_keys = Counter((row["tracker"], str(Path(row["source_motion_path"]).resolve())) for row in manifest)
    record_keys = Counter((row["tracker"], str(Path(row["source_motion_path"]).resolve())) for row in records)
    if manifest_keys != record_keys:
        raise ValueError("rollouts.jsonl does not match sidecar metadata")

    summary = {
        "source_policy": "non_flipped_train",
        "train_json": str(train_json),
        "train_json_sha256": file_hash(train_json),
        "rollout_root": str(rollout_root),
        "tracker_motion_counts": {tracker: len(coverage[tracker]) for tracker in TRACKERS},
        "total_rollouts": len(records),
        "manifest": str(manifest_path),
    }
    summary_path = rollout_root / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
