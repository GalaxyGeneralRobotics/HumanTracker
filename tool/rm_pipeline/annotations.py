"""Strict annotation aggregation and reward-model export."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


VALID_CHOICES = {"preference", "similar", "bad_traj"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_pair_record(pair: Mapping[str, Any]) -> None:
    required = {
        "pair_idx",
        "pair_id",
        "motion_idx",
        "motion_id",
        "clip_idx",
        "clip_uid",
        "source_start_frame",
        "source_end_frame",
        "fps",
        "category",
        "tracker_pair_key",
        "candidates",
    }
    missing = sorted(required - pair.keys())
    if missing:
        raise KeyError(f"{pair.get('pair_id')!r}: missing fields {missing}")
    candidates = pair["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(f"{pair['pair_id']}: expected exactly two candidates")
    if [int(candidate["candidate_idx"]) for candidate in candidates] != [0, 1]:
        raise ValueError(f"{pair['pair_id']}: candidate_idx must be [0, 1]")
    if len({str(candidate["tracker"]) for candidate in candidates}) != 2:
        raise ValueError(f"{pair['pair_id']}: candidates must use distinct trackers")
    for candidate in candidates:
        if str(candidate["motion_id"]) != str(pair["motion_id"]):
            raise ValueError(f"{pair['pair_id']}: candidate motion mismatch")
        for key in ("traj_path", "source_rollout_path", "source_motion_path"):
            path = Path(str(candidate[key]))
            if not path.is_file():
                raise FileNotFoundError(path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["annotation_json"])
    return [json.loads(value) for value in table["annotation_json"].to_pylist()]


def _read_indexed_directory(path: Path) -> list[dict[str, Any]]:
    index_paths = sorted(path.glob("*.index.json"))
    if len(index_paths) != 1:
        raise ValueError(f"{path}: expected exactly one annotation index, got {len(index_paths)}")
    index = json.loads(index_paths[0].read_text(encoding="utf-8"))
    rows = []
    expected_start = 0
    indexed_names = []
    for item in index["files"]:
        start = int(item["start_idx"])
        end = int(item["end_idx"])
        count = int(item["count"])
        if start != expected_start or end - start + 1 != count:
            raise ValueError(f"{index_paths[0]}: non-contiguous shard {item}")
        shard = path / str(item["path"])
        indexed_names.append(shard.name)
        shard_rows = _read_parquet(shard)
        if len(shard_rows) != count:
            raise ValueError(f"{shard}: row count mismatch")
        rows.extend(shard_rows)
        expected_start = end + 1
    if expected_start != int(index["total_records"]):
        raise ValueError(f"{index_paths[0]}: total_records mismatch")
    disk_names = sorted(item.name for item in path.glob("*.parquet"))
    if indexed_names != disk_names:
        raise ValueError(f"{index_paths[0]}: indexed shard set does not match disk")
    return rows


def read_annotations(inputs: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_dir():
            records.extend(_read_indexed_directory(path))
        elif path.suffix == ".parquet" and path.is_file():
            records.extend(_read_parquet(path))
        elif path.suffix == ".jsonl" and path.is_file():
            records.extend(read_jsonl(path))
        else:
            raise FileNotFoundError(f"annotation input must be a directory, parquet, or jsonl: {path}")
    record_ids = []
    for record in records:
        validate_annotation(record)
        record_ids.append(str(record["record_id"]))
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate annotation record_id")
    return records


def validate_annotation(record: Mapping[str, Any]) -> None:
    required = {
        "record_id",
        "pair_idx",
        "pair_id",
        "motion_idx",
        "motion_id",
        "category",
        "tracker_pair_key",
        "meta",
        "candidates",
        "preference",
        "flags",
        "comparison",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise KeyError(f"{record.get('record_id')!r}: missing fields {missing}")
    candidates = record["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(f"{record['record_id']}: expected two candidates")
    if [int(candidate["candidate_idx"]) for candidate in candidates] != [0, 1]:
        raise ValueError(f"{record['record_id']}: candidate_idx must be [0, 1]")
    if len({str(candidate["tracker"]) for candidate in candidates}) != 2:
        raise ValueError(f"{record['record_id']}: candidates must use distinct trackers")
    for candidate in candidates:
        required_candidate = {
            "candidate_idx",
            "candidate_uid",
            "tracker",
            "traj_path",
            "extra",
        }
        missing_candidate = sorted(required_candidate - candidate.keys())
        if missing_candidate:
            raise KeyError(f"{record['record_id']}: candidate missing {missing_candidate}")
        extra = candidate["extra"]
        for key in ("clip_uid", "source_start_frame", "source_end_frame", "fps"):
            if key not in extra:
                raise KeyError(f"{record['record_id']}: candidate.extra.{key} is required")
        if not Path(str(candidate["traj_path"])).is_file():
            raise FileNotFoundError(candidate["traj_path"])
    preference = record["preference"]
    choice_type = preference["choice_type"]
    preferred = preference["preferred_candidate_idx"]
    if choice_type not in VALID_CHOICES:
        raise ValueError(f"{record['record_id']}: invalid choice_type {choice_type!r}")
    if choice_type == "preference" and int(preferred) not in (0, 1):
        raise ValueError(f"{record['record_id']}: invalid preferred_candidate_idx")
    if choice_type != "preference" and preferred is not None:
        raise ValueError(f"{record['record_id']}: non-preference label selected a candidate")
    if bool(record["flags"]["invalid"]) != (choice_type == "bad_traj"):
        raise ValueError(f"{record['record_id']}: flags.invalid does not match choice_type")


def _candidate_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(candidate["candidate_idx"]),
        str(candidate["candidate_uid"]),
        str(candidate["tracker"]),
        str(candidate["traj_path"]),
    )


def aggregate_annotations(
    inputs: Sequence[str],
    output: Path,
    min_annotations: int,
    min_agreement: int,
) -> list[dict[str, Any]]:
    if min_annotations < 1 or min_agreement < 1:
        raise ValueError("minimum annotation counts must be positive")
    records = read_annotations(inputs)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pair_id"])].append(record)

    aggregates = []
    for pair_id, annotations in sorted(grouped.items()):
        first = annotations[0]
        signatures = [_candidate_signature(candidate) for candidate in first["candidates"]]
        votes: Counter[str] = Counter()
        annotation_rows = []
        annotators = set()
        for record in annotations:
            if (
                int(record["pair_idx"]) != int(first["pair_idx"])
                or str(record["motion_id"]) != str(first["motion_id"])
                or str(record["category"]) != str(first["category"])
                or [_candidate_signature(candidate) for candidate in record["candidates"]]
                != signatures
            ):
                raise ValueError(f"{pair_id}: inconsistent annotation metadata")
            annotator_id = str(record["meta"]["annotator_id"])
            if annotator_id in annotators:
                raise ValueError(f"{pair_id}: duplicate annotator {annotator_id!r}")
            annotators.add(annotator_id)
            choice_type = str(record["preference"]["choice_type"])
            vote = (
                f"candidate_{int(record['preference']['preferred_candidate_idx'])}"
                if choice_type == "preference"
                else choice_type
            )
            votes[vote] += 1
            annotation_rows.append({
                "record_id": record["record_id"],
                "annotator_id": annotator_id,
                "choice_type": choice_type,
                "preferred_candidate_idx": record["preference"]["preferred_candidate_idx"],
                "vote": vote,
                "timestamp": record["meta"]["timestamp"],
            })

        top_count = max(votes.values())
        top_votes = sorted(key for key, count in votes.items() if count == top_count)
        top_vote = top_votes[0] if len(top_votes) == 1 else "tie"
        valid = (
            len(annotations) >= min_annotations
            and top_count >= min_agreement
            and top_vote in {"candidate_0", "candidate_1"}
        )
        chosen_idx = int(top_vote[-1]) if valid else None
        candidates = first["candidates"]
        candidate_extra = candidates[0]["extra"]
        aggregates.append({
            "pair_id": pair_id,
            "pair_idx": int(first["pair_idx"]),
            "motion_idx": int(first["motion_idx"]),
            "motion_id": str(first["motion_id"]),
            "category": str(first["category"]),
            "tracker_pair_key": str(first["tracker_pair_key"]),
            "clip_uid": str(candidate_extra["clip_uid"]),
            "source_start_frame": int(candidate_extra["source_start_frame"]),
            "source_end_frame": int(candidate_extra["source_end_frame"]),
            "fps": int(candidate_extra["fps"]),
            "num_annotations": len(annotations),
            "vote_counts": dict(votes),
            "top_vote": top_vote,
            "top_vote_count": top_count,
            "is_valid": valid,
            "is_ambiguous": not valid,
            "chosen": candidates[chosen_idx] if chosen_idx is not None else None,
            "rejected": candidates[1 - chosen_idx] if chosen_idx is not None else None,
            "annotations": annotation_rows,
        })
    write_jsonl(output, aggregates)
    return aggregates


def export_rm_parquet(aggregates_path: Path, output: Path) -> int:
    aggregates = read_jsonl(aggregates_path)
    rows = []
    for aggregate in aggregates:
        if not aggregate["is_valid"]:
            continue
        chosen = aggregate["chosen"]
        rejected = aggregate["rejected"]
        row = {
            "pair_id": str(aggregate["pair_id"]),
            "motion_id": str(aggregate["motion_id"]),
            "clip_uid": str(aggregate["clip_uid"]),
            "chosen_path": str(chosen["traj_path"]),
            "rejected_path": str(rejected["traj_path"]),
            "chosen_tracker": str(chosen["tracker"]),
            "rejected_tracker": str(rejected["tracker"]),
            "source_start_frame": int(aggregate["source_start_frame"]),
            "source_end_frame": int(aggregate["source_end_frame"]),
            "fps": int(aggregate["fps"]),
            "num_annotations": int(aggregate["num_annotations"]),
            "vote_counts_json": json.dumps(aggregate["vote_counts"], ensure_ascii=False),
            "annotations_json": json.dumps(aggregate["annotations"], ensure_ascii=False),
            "chosen_metadata_json": json.dumps(chosen["extra"], ensure_ascii=False),
            "rejected_metadata_json": json.dumps(rejected["extra"], ensure_ascii=False),
            "aggregate_json": json.dumps(aggregate, ensure_ascii=False),
        }
        row["duration_sec"] = (
            row["source_end_frame"] - row["source_start_frame"]
        ) / row["fps"]
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("pair_id", pa.string()),
        ("motion_id", pa.string()),
        ("clip_uid", pa.string()),
        ("chosen_path", pa.string()),
        ("rejected_path", pa.string()),
        ("chosen_tracker", pa.string()),
        ("rejected_tracker", pa.string()),
        ("source_start_frame", pa.int64()),
        ("source_end_frame", pa.int64()),
        ("fps", pa.int64()),
        ("duration_sec", pa.float64()),
        ("num_annotations", pa.int64()),
        ("vote_counts_json", pa.string()),
        ("annotations_json", pa.string()),
        ("chosen_metadata_json", pa.string()),
        ("rejected_metadata_json", pa.string()),
        ("aggregate_json", pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)
    return len(rows)
