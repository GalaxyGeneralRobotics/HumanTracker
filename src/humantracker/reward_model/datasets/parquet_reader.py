"""Strict reader for HumanTracker preference annotation shards."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_annotations(data_dir: Path) -> list[dict]:
    paths = sorted(data_dir.glob("hf_records_idx_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no annotation shards in {data_dir}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=[
                    "record_idx",
                    "record_id",
                    "pair_id",
                    "choice_type",
                    "invalid",
                    "annotation_json",
                ],
            )
            for path in paths
        ]
    )
    rows = table.to_pylist()
    if [row["record_idx"] for row in rows] != list(range(len(rows))):
        raise ValueError("record_idx must be contiguous and ordered")
    annotations = []
    for row in rows:
        annotation = json.loads(row["annotation_json"])
        choice = annotation["preference"]["choice_type"]
        if annotation["record_id"] != row["record_id"] or annotation["pair_id"] != row["pair_id"]:
            raise ValueError(f"{row['record_id']}: identifier mismatch")
        if choice != row["choice_type"] or bool(row["invalid"]) != (choice == "bad_traj"):
            raise ValueError(f"{row['record_id']}: label mismatch")
        annotations.append(annotation)
    record_ids = [item["record_id"] for item in annotations]
    pair_ids = [item["pair_id"] for item in annotations]
    if len(set(record_ids)) != len(record_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("record_id and pair_id must be unique")
    return annotations
