"""Strict reader for HumanTracker preference records and the rollouts they compare.

A preference label is meaningless on its own: it says that one of two tracker rollouts
looked better, so the two rollouts have to arrive with it. The published dataset carries
them inline -- one NPZ payload per candidate, in the same parquet row as the label -- and
this module is the only place that knows how to take them apart. Every consumer receives
the same thing regardless of layout: an annotation dict, and the two rollout payloads as
raw NPZ bytes.

Two layouts are read. The published one, `<split>/<split>-NNNNN-of-NNNNN.parquet` beside
`train.json` and `test.json`, is self-contained and is what the paper's numbers are
reproduced from. The internal one, `hf_records_idx_*.parquet`, stores absolute
`traj_path`s into a rollout export instead, and is what the annotation pipeline emits
before publication.

Nothing here is lenient. A row whose columns disagree with its own `annotation_json`, a
split whose parquet contents disagree with its manifest, or a missing rollout is an error,
because each of those failures would otherwise train a model on a quietly different
dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = ("train", "test")

# Columns that duplicate `annotation_json` and are cross-checked against it. The payload
# columns are read separately, only while streaming, because they dominate the row size.
METADATA_COLUMNS = (
    "record_idx",
    "record_id",
    "pair_id",
    "choice_type",
    "invalid",
    "candidate_0_tracker",
    "candidate_1_tracker",
    "annotation_json",
)

PAYLOAD_COLUMNS = ("record_id", "candidate_0_npz", "candidate_1_npz")

LEGACY_COLUMNS = (
    "record_idx",
    "record_id",
    "pair_id",
    "choice_type",
    "invalid",
    "annotation_json",
)


def published_shards(data_dir: Path, split: str) -> list[Path]:
    paths = sorted((data_dir / split).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {data_dir / split}")
    return paths


def legacy_shards(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("hf_records_idx_*.parquet"))


def is_published(data_dir: Path) -> bool:
    return all((data_dir / split).is_dir() for split in SPLITS)


def _check_row(row: dict, annotation: dict) -> None:
    record_id = row["record_id"]
    if annotation["record_id"] != record_id or annotation["pair_id"] != row["pair_id"]:
        raise ValueError(f"{record_id}: identifier mismatch")
    choice = annotation["preference"]["choice_type"]
    if choice != row["choice_type"] or bool(row["invalid"]) != (choice == "bad_traj"):
        raise ValueError(f"{record_id}: label mismatch")
    for index in (0, 1):
        column = row.get(f"candidate_{index}_tracker")
        if column is None:
            continue
        candidates = {int(item["candidate_idx"]): item for item in annotation["candidates"]}
        if str(candidates[index]["tracker"]) != column:
            raise ValueError(f"{record_id}: candidate {index} tracker mismatch")


def _read_annotations(paths: list[Path], columns: tuple[str, ...], label: str) -> list[dict]:
    rows = pa.concat_tables(
        [pq.read_table(path, columns=list(columns)) for path in paths]
    ).to_pylist()
    if [row["record_idx"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{label}: record_idx must be contiguous and ordered")
    annotations = []
    for row in rows:
        annotation = json.loads(row["annotation_json"])
        _check_row(row, annotation)
        annotations.append(annotation)
    return annotations


def _check_manifest(data_dir: Path, split: str, annotations: list[dict]) -> None:
    manifest = json.loads((data_dir / f"{split}.json").read_text())
    if manifest.get("split") != split:
        raise ValueError(f"{data_dir / f'{split}.json'}: expected split={split}")
    if manifest["record_ids"] != [item["record_id"] for item in annotations]:
        raise ValueError(f"{split}: parquet shards disagree with {split}.json")


def load_annotations(data_dir: Path) -> list[dict]:
    """Every annotation, in the order `iter_rollout_pairs` will stream them."""
    if is_published(data_dir):
        annotations = []
        for split in SPLITS:
            records = _read_annotations(
                published_shards(data_dir, split), METADATA_COLUMNS, f"{data_dir}/{split}"
            )
            _check_manifest(data_dir, split, records)
            annotations.extend(records)
    else:
        paths = legacy_shards(data_dir)
        if not paths:
            raise FileNotFoundError(f"no annotation shards in {data_dir}")
        annotations = _read_annotations(paths, LEGACY_COLUMNS, str(data_dir))
    record_ids = [item["record_id"] for item in annotations]
    pair_ids = [item["pair_id"] for item in annotations]
    if len(set(record_ids)) != len(record_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("record_id and pair_id must be unique")
    return annotations


def _legacy_payloads(annotation: dict) -> dict[int, bytes]:
    payloads = {}
    for candidate in annotation["candidates"]:
        path = Path(candidate["traj_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[int(candidate["candidate_idx"])] = path.read_bytes()
    return payloads


def iter_rollout_pairs(data_dir: Path, batch_rows: int = 8) -> Iterator[tuple[dict, dict[int, bytes]]]:
    """Annotations paired with the two rollout payloads they compare.

    Streamed rather than returned as a list: the payloads for the full dataset are several
    gigabytes, and every caller consumes them one pair at a time.
    """
    if not is_published(data_dir):
        for annotation in load_annotations(data_dir):
            yield annotation, _legacy_payloads(annotation)
        return
    for split in SPLITS:
        for path in published_shards(data_dir, split):
            metadata = pq.read_table(path, columns=list(METADATA_COLUMNS)).to_pylist()
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=batch_rows, columns=list(PAYLOAD_COLUMNS)
            ):
                for row in batch.to_pylist():
                    expected = metadata[index]
                    if row["record_id"] != expected["record_id"]:
                        raise ValueError(f"{path}: payload rows are out of order")
                    index += 1
                    annotation = json.loads(expected["annotation_json"])
                    _check_row(expected, annotation)
                    yield annotation, {0: row["candidate_0_npz"], 1: row["candidate_1_npz"]}
            if index != len(metadata):
                raise ValueError(f"{path}: streamed {index} of {len(metadata)} rows")
