"""Strict Parquet writer for canonical human-preference annotations."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.hf_proto import HFRecord


DEFAULT_SHARD_SIZE = 200
VALID_CHOICES = {"preference", "similar", "bad_traj"}


class HFRecorder:
    """Append annotations to immutable Parquet shards with one strict index."""

    def __init__(
        self,
        out_dir: str | os.PathLike[str],
        filename_stem: str = "hf_records",
        ensure_ascii: bool = False,
        shard_size: int = DEFAULT_SHARD_SIZE,
        undo_limit: int = 0,
    ) -> None:
        if not filename_stem or any(char in filename_stem for char in "/\\"):
            raise ValueError(f"invalid filename_stem: {filename_stem!r}")
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        if undo_limit < 0:
            raise ValueError("undo_limit must be non-negative")
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.filename_stem = filename_stem
        self.ensure_ascii = bool(ensure_ascii)
        self.shard_size = int(shard_size)
        self.undo_limit = int(undo_limit)
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._recover_commit()
        self._index = self._load_index()
        self._records = self._read_pending()

    @property
    def index_path(self) -> Path:
        return self.out_dir / f"{self.filename_stem}.index.json"

    @property
    def pending_path(self) -> Path:
        return self.out_dir / f"{self.filename_stem}.pending.jsonl"

    @property
    def commit_path(self) -> Path:
        return self.out_dir / f"{self.filename_stem}.commit.json"

    def _parquet_path(self, start_idx: int, end_idx: int) -> Path:
        return self.out_dir / f"{self.filename_stem}_idx_{start_idx:06d}-{end_idx:06d}.parquet"

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _empty_index() -> dict[str, Any]:
        return {
            "total_records": 0,
            "files": [],
            "data_source": None,
        }

    def _load_index(self) -> dict[str, Any]:
        parquet_files = sorted(self.out_dir.glob(f"{self.filename_stem}_idx_*.parquet"))
        if not self.index_path.exists():
            if parquet_files:
                raise FileNotFoundError(
                    f"index is missing for {len(parquet_files)} existing Parquet shards: {self.index_path}"
                )
            return self._empty_index()

        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        expected_start = 0
        indexed_names = []
        for item in index["files"]:
            start_idx = int(item["start_idx"])
            end_idx = int(item["end_idx"])
            count = int(item["count"])
            if start_idx != expected_start or end_idx - start_idx + 1 != count:
                raise ValueError(f"{self.index_path}: non-contiguous shard entry {item}")
            shard = self.out_dir / str(item["path"])
            if not shard.is_file():
                raise FileNotFoundError(shard)
            if pq.ParquetFile(shard).metadata.num_rows != count:
                raise ValueError(f"{shard}: row count does not match index")
            indexed_names.append(shard.name)
            expected_start = end_idx + 1

        if int(index["total_records"]) != expected_start:
            raise ValueError(f"{self.index_path}: total_records is inconsistent")
        if indexed_names != [path.name for path in parquet_files]:
            raise ValueError(f"{self.index_path}: indexed shard set does not match disk")
        return index

    def _write_index(self, index: dict[str, Any]) -> None:
        temp_path = self.index_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.index_path)

    def _read_pending(self) -> list[dict[str, Any]]:
        if not self.pending_path.exists():
            return []
        records = []
        with self.pending_path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"{self.pending_path}:{line_no}: blank line")
                annotation = json.loads(line)
                if not isinstance(annotation, dict):
                    raise TypeError(f"{self.pending_path}:{line_no}: expected an object")
                records.append(annotation)
        if records:
            self._table(records, 0)
        return records

    def _append_pending(self, annotation: dict[str, Any]) -> None:
        with self.pending_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(annotation, ensure_ascii=self.ensure_ascii) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_pending(self, records: list[dict[str, Any]]) -> None:
        if not records:
            self.pending_path.unlink(missing_ok=True)
            return
        temp_path = self.pending_path.with_suffix(".jsonl.tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            for annotation in records:
                stream.write(
                    json.dumps(annotation, ensure_ascii=self.ensure_ascii) + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(self.pending_path)

    def _write_commit(self, commit: dict[str, Any]) -> None:
        temp_path = self.commit_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(commit, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(self.commit_path)

    def _recover_commit(self) -> None:
        if not self.commit_path.exists():
            return
        commit = json.loads(self.commit_path.read_text(encoding="utf-8"))
        entry = dict(commit["index_entry"])
        record_ids = [str(value) for value in commit["record_ids"]]
        output_path = self.out_dir / str(entry["path"])
        temp_path = self.out_dir / str(commit["temp_path"])
        if output_path.exists():
            if temp_path.exists():
                raise FileExistsError(temp_path)
        elif temp_path.exists():
            temp_path.replace(output_path)
        else:
            raise FileNotFoundError(
                f"commit has neither output nor temporary Parquet: {output_path}"
            )

        table = pq.read_table(output_path, columns=["record_id"])
        if table["record_id"].to_pylist() != record_ids:
            raise ValueError(f"{output_path}: record ids do not match commit")
        if table.num_rows != int(entry["count"]):
            raise ValueError(f"{output_path}: row count does not match commit")

        index = (
            json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index_path.exists()
            else self._empty_index()
        )
        indexed = [item for item in index["files"] if item["path"] == entry["path"]]
        if indexed:
            if indexed != [entry]:
                raise ValueError(f"{self.index_path}: commit index entry differs")
        else:
            if int(index["total_records"]) != int(entry["start_idx"]):
                raise ValueError(f"{self.index_path}: commit start index differs")
            index["files"] = [*index["files"], entry]
            index["total_records"] = int(entry["end_idx"]) + 1
            index["updated_at"] = self.now_iso()
            self._write_index(index)

        pending = self._read_pending()
        pending_ids = [str(annotation["record_id"]) for annotation in pending]
        present = [record_id in pending_ids for record_id in record_ids]
        if all(present):
            selected = set(record_ids)
            self._write_pending([
                annotation
                for annotation in pending
                if str(annotation["record_id"]) not in selected
            ])
        elif any(present):
            raise ValueError(f"{self.pending_path}: commit records are partially present")
        self.commit_path.unlink()

    @staticmethod
    def _to_primitive(value: Any) -> Any:
        if is_dataclass(value):
            return HFRecorder._to_primitive(asdict(value))
        if isinstance(value, dict):
            return {str(key): HFRecorder._to_primitive(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [HFRecorder._to_primitive(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported annotation value: {type(value).__name__}")

    @staticmethod
    def _validate_extra(value: Any) -> Any:
        forbidden = {
            "left",
            "right",
            "video_left",
            "video_right",
            "candidate_id",
            "clip_npz_path",
            "canonical_side",
            "display_side",
        }
        if isinstance(value, dict):
            invalid = sorted(forbidden & value.keys())
            if invalid:
                raise ValueError(f"legacy display fields are not allowed: {invalid}")
            return {
                str(key): HFRecorder._validate_extra(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [HFRecorder._validate_extra(item) for item in value]
        return value

    @staticmethod
    def _canonical_annotation(record: HFRecord) -> dict[str, Any]:
        raw = HFRecorder._to_primitive(record)
        meta = dict(raw["meta"])
        comparison = dict(raw["comparison"])
        comparison_extra = dict(comparison["extra"])
        preference = dict(raw["preference"])
        flags = dict(raw["flags"])

        candidates = []
        displayed_candidate_indices = []
        for side_key in ("video_left", "video_right"):
            side = dict(raw[side_key])
            extra = dict(side["extra"])
            candidate_idx = int(extra.pop("candidate_idx"))
            tracker = str(extra.pop("tracker"))
            traj_path = str(extra.pop("traj_path"))
            candidate_uid = str(extra.pop("candidate_uid"))
            if candidate_idx not in (0, 1):
                raise ValueError(f"invalid candidate_idx: {candidate_idx}")
            if not tracker or not traj_path or not candidate_uid:
                raise ValueError(f"{side_key}: tracker, traj_path, and candidate_uid are required")
            displayed_candidate_indices.append(candidate_idx)
            candidates.append({
                "candidate_idx": candidate_idx,
                "candidate_uid": candidate_uid,
                "tracker": tracker,
                "traj_path": traj_path,
                "npz_name": str(side["npz_name"]),
                "start_frame": int(side["start_frame"]),
                "end_frame": int(side["end_frame"]),
                "extra": HFRecorder._validate_extra(extra),
            })
        if sorted(displayed_candidate_indices) != [0, 1]:
            raise ValueError(f"displayed candidate indices must be [0, 1]: {displayed_candidate_indices}")
        candidates.sort(key=lambda item: item["candidate_idx"])

        human_preference = dict(comparison_extra.pop("human_preference"))
        choice_type = str(human_preference["choice_type"])
        preferred_candidate_idx = human_preference["preferred_candidate_idx"]
        if choice_type not in VALID_CHOICES:
            raise ValueError(f"invalid choice_type: {choice_type}")
        if choice_type == "preference":
            preferred_candidate_idx = int(preferred_candidate_idx)
            if preferred_candidate_idx not in (0, 1):
                raise ValueError(f"invalid preferred_candidate_idx: {preferred_candidate_idx}")
        elif preferred_candidate_idx is not None:
            raise ValueError(f"{choice_type} must not select a candidate")

        display_order = [int(value) for value in comparison_extra.pop("display_order")]
        if display_order != displayed_candidate_indices:
            raise ValueError(
                f"display_order {display_order} does not match displayed candidates {displayed_candidate_indices}"
            )
        display_choice_idx = comparison_extra.pop("display_choice_idx")
        expected_display_choice = (
            display_order.index(preferred_candidate_idx) if preferred_candidate_idx is not None else None
        )
        if display_choice_idx != expected_display_choice:
            raise ValueError(
                f"display_choice_idx {display_choice_idx!r} != expected {expected_display_choice!r}"
            )

        pair_id = str(comparison_extra.pop("pair_id"))
        pair_idx = int(comparison_extra.pop("pair_idx"))
        motion_id = str(comparison_extra.pop("motion_id"))
        category = str(comparison_extra.pop("category"))
        tracker_pair_key = str(comparison_extra.pop("tracker_pair_key"))
        motion_idx = int(comparison_extra.pop("motion_idx"))
        if not all((pair_id, motion_id, category, tracker_pair_key)):
            raise ValueError("pair_id, motion_id, category, and tracker_pair_key are required")

        return {
            "record_id": str(meta["record_id"]),
            "pair_idx": pair_idx,
            "pair_id": pair_id,
            "motion_idx": motion_idx,
            "motion_id": motion_id,
            "category": category,
            "tracker_pair_key": tracker_pair_key,
            "meta": meta,
            "candidates": candidates,
            "preference": {
                "choice_type": choice_type,
                "preferred_candidate_idx": preferred_candidate_idx,
                "confidence": float(preference["confidence"]),
            },
            "flags": flags,
            "comparison": {
                "type": str(comparison["type"]),
                "play_mode": str(comparison["play_mode"]),
                "camera": str(comparison["camera"]),
                "length_frames": int(comparison["length_frames"]),
                "display_order": display_order,
                "display_choice_idx": display_choice_idx,
                "extra": HFRecorder._validate_extra(comparison_extra),
            },
            "notes": raw.get("notes"),
        }

    @staticmethod
    def _schema() -> pa.Schema:
        candidate_type = pa.struct([
            ("candidate_idx", pa.int8()),
            ("candidate_uid", pa.string()),
            ("tracker", pa.string()),
            ("npz_name", pa.string()),
            ("traj_path", pa.string()),
            ("start_frame", pa.int32()),
            ("end_frame", pa.int32()),
        ])
        return pa.schema([
            ("record_idx", pa.int64()),
            ("record_id", pa.string()),
            ("pair_idx", pa.int64()),
            ("pair_id", pa.string()),
            ("annotator_id", pa.string()),
            ("timestamp", pa.string()),
            ("choice_type", pa.string()),
            ("preferred_candidate_idx", pa.int8()),
            ("invalid", pa.bool_()),
            ("candidate_indices", pa.list_(pa.int8(), 2)),
            ("candidate_trackers", pa.list_(pa.string(), 2)),
            ("candidates", pa.list_(candidate_type, 2)),
            ("annotation_json", pa.string()),
        ]).with_metadata({
            b"candidate_identity": b"candidate_idx is stable; display order is transient",
        })

    def _table(self, records: list[dict[str, Any]], start_idx: int) -> pa.Table:
        rows = []
        for offset, annotation in enumerate(records):
            candidates = annotation["candidates"]
            rows.append({
                "record_idx": start_idx + offset,
                "record_id": annotation["record_id"],
                "pair_idx": annotation["pair_idx"],
                "pair_id": annotation["pair_id"],
                "annotator_id": annotation["meta"]["annotator_id"],
                "timestamp": annotation["meta"]["timestamp"],
                "choice_type": annotation["preference"]["choice_type"],
                "preferred_candidate_idx": annotation["preference"]["preferred_candidate_idx"],
                "invalid": bool(annotation["flags"]["invalid"]),
                "candidate_indices": [candidate["candidate_idx"] for candidate in candidates],
                "candidate_trackers": [candidate["tracker"] for candidate in candidates],
                "candidates": [{
                    "candidate_idx": candidate["candidate_idx"],
                    "candidate_uid": candidate["candidate_uid"],
                    "tracker": candidate["tracker"],
                    "npz_name": candidate["npz_name"],
                    "traj_path": candidate["traj_path"],
                    "start_frame": candidate["start_frame"],
                    "end_frame": candidate["end_frame"],
                } for candidate in candidates],
                "annotation_json": json.dumps(annotation, ensure_ascii=self.ensure_ascii),
            })
        return pa.Table.from_pylist(rows, schema=self._schema())

    def append(self, record: HFRecord) -> Path | None:
        if not isinstance(record, HFRecord):
            raise TypeError(f"expected HFRecord, got {type(record).__name__}")
        annotation = self._canonical_annotation(record)
        with self._lock:
            if self.commit_path.exists():
                raise RuntimeError(f"unfinished commit requires restart: {self.commit_path}")
            self._append_pending(annotation)
            self._records.append(annotation)
            return self._save_ready_locked()

    def _protected_record_ids(self) -> set[str]:
        if self.undo_limit == 0:
            return set()
        counts: dict[str, int] = {}
        protected = set()
        for annotation in reversed(self._records):
            annotator_id = str(annotation["meta"]["annotator_id"])
            count = counts.get(annotator_id, 0)
            if count < self.undo_limit:
                protected.add(str(annotation["record_id"]))
                counts[annotator_id] = count + 1
        return protected

    def _flushable_records(self) -> list[dict[str, Any]]:
        protected = self._protected_record_ids()
        return [
            annotation
            for annotation in self._records
            if str(annotation["record_id"]) not in protected
        ]

    def _save_batch_locked(self, records: list[dict[str, Any]]) -> Path:
        count = len(records)
        if count != self.shard_size:
            raise ValueError(f"Parquet shards must contain exactly {self.shard_size} records")
        selected_ids = [str(annotation["record_id"]) for annotation in records]
        if len(set(selected_ids)) != count:
            raise ValueError("selected batch has duplicate record IDs")
        pending_ids = [str(annotation["record_id"]) for annotation in self._records]
        if any(pending_ids.count(record_id) != 1 for record_id in selected_ids):
            raise ValueError("selected batch does not match pending records")
        start_idx = int(self._index["total_records"])
        end_idx = start_idx + count - 1
        output_path = self._parquet_path(start_idx, end_idx)
        if output_path.exists():
            raise FileExistsError(output_path)
        table = self._table(records, start_idx)
        temp_path = output_path.with_suffix(".parquet.tmp")
        temp_path.unlink(missing_ok=True)
        pq.write_table(table, temp_path, compression="zstd", use_dictionary=True)
        if pq.ParquetFile(temp_path).metadata.num_rows != count:
            raise ValueError(f"{temp_path}: written row count mismatch")

        entry = {
            "path": output_path.name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "count": count,
            "created_at": self.now_iso(),
        }
        self._write_commit({
            "index_entry": entry,
            "record_ids": [str(annotation["record_id"]) for annotation in records],
            "temp_path": temp_path.name,
        })
        temp_path.replace(output_path)

        next_index = dict(self._index)
        next_index["files"] = [*self._index["files"], entry]
        next_index["total_records"] = end_idx + 1
        next_index["updated_at"] = self.now_iso()
        self._write_index(next_index)
        selected = set(selected_ids)
        remaining = [
            annotation
            for annotation in self._records
            if str(annotation["record_id"]) not in selected
        ]
        self._write_pending(remaining)
        self.commit_path.unlink()
        self._index = next_index
        self._records = remaining
        return output_path

    def _save_ready_locked(self) -> Path | None:
        last_output = None
        flushable = self._flushable_records()
        while len(flushable) >= self.shard_size:
            last_output = self._save_batch_locked(flushable[: self.shard_size])
            flushable = self._flushable_records()
        return last_output

    def save(self) -> Path | None:
        last_output = None
        with self._lock:
            if self.commit_path.exists():
                raise RuntimeError(f"unfinished commit requires restart: {self.commit_path}")
            last_output = self._save_ready_locked()
        return last_output

    def close(self) -> Path | None:
        with self._lock:
            if self.commit_path.exists():
                raise RuntimeError(f"unfinished commit requires restart: {self.commit_path}")
            self.undo_limit = 0
            return self._save_ready_locked()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._records)

    def undo(self, record_id: str, annotator_id: str, pair_id: str) -> None:
        record_id = str(record_id).strip()
        annotator_id = str(annotator_id).strip()
        pair_id = str(pair_id).strip()
        if not record_id or not annotator_id or not pair_id:
            raise ValueError("record_id, annotator_id, and pair_id are required")
        with self._lock:
            if self.commit_path.exists():
                raise RuntimeError(f"unfinished commit requires restart: {self.commit_path}")
            matches = [
                annotation
                for annotation in self._records
                if annotation["record_id"] == record_id
            ]
            if len(matches) != 1:
                raise ValueError(f"record {record_id!r} is not pending")
            annotation = matches[0]
            if annotation["meta"]["annotator_id"] != annotator_id:
                raise ValueError("record belongs to another annotator")
            if annotation["pair_id"] != pair_id:
                raise ValueError("record belongs to another pair")
            if record_id not in self._protected_record_ids():
                raise ValueError(f"record {record_id!r} is outside the undo window")
            self._records.remove(annotation)
            self._write_pending(self._records)

    def load_all_records(self) -> list[dict[str, Any]]:
        records = []
        for item in self._index["files"]:
            path = self.out_dir / item["path"]
            table = pq.read_table(path, columns=["annotation_json"])
            records.extend(json.loads(text) for text in table["annotation_json"].to_pylist())
        records.extend(self._records)
        if len(records) != int(self._index["total_records"]) + len(self._records):
            raise ValueError("loaded record count does not match index")
        return records

    def get_data_source(self) -> dict[str, Any] | None:
        return self._index.get("data_source")

    def set_data_source(self, data_source: dict[str, Any]) -> None:
        if not isinstance(data_source, dict):
            raise TypeError("data_source must be a dict")
        next_index = dict(self._index)
        next_index["data_source"] = self._to_primitive(data_source)
        next_index["updated_at"] = self.now_iso()
        self._write_index(next_index)
        self._index = next_index

    def get_total_count(self) -> int:
        return int(self._index["total_records"]) + len(self._records)
