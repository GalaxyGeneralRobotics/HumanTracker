"""Strict loading, leasing, and progress tracking for pair annotations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_LABELS = {"left", "right", "similar", "bad_traj"}
UNDO_LIMIT = 5


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
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
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise TypeError(f"{path}: expected a list of objects")
        return rows
    raise ValueError(f"task file must be .jsonl or .json: {path}")


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(raw)
    required = {
        "candidate_idx",
        "candidate_uid",
        "tracker",
        "traj_path",
        "motion_id",
        "source_start_frame",
        "source_end_frame",
        "local_start_frame",
        "local_end_frame",
        "num_frames",
        "fps",
    }
    missing = sorted(required - candidate.keys())
    if missing:
        raise KeyError(f"candidate is missing fields: {missing}")
    candidate_idx = int(candidate["candidate_idx"])
    if candidate_idx not in (0, 1):
        raise ValueError(f"candidate_idx must be 0 or 1: {candidate_idx}")
    candidate["candidate_idx"] = candidate_idx
    for key in ("traj_path", "source_rollout_path", "source_motion_path"):
        if key in candidate:
            candidate[key] = str(_resolve_path(candidate[key]))
    traj_path = Path(candidate["traj_path"])
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    if int(candidate["local_end_frame"]) <= int(candidate["local_start_frame"]):
        raise ValueError(f"candidate has invalid local frame range: {candidate}")
    return candidate


def _normalize_pair(raw: dict[str, Any]) -> dict[str, Any]:
    pair = dict(raw)
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
        raise KeyError(f"pair is missing fields: {missing}")
    candidates = pair["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError(f"pair {pair['pair_id']}: expected exactly two candidates")
    pair["candidates"] = [_normalize_candidate(candidate) for candidate in candidates]
    if [candidate["candidate_idx"] for candidate in pair["candidates"]] != [0, 1]:
        raise ValueError(f"pair {pair['pair_id']}: candidates must be ordered [0, 1]")
    if len({candidate["tracker"] for candidate in pair["candidates"]}) != 2:
        raise ValueError(f"pair {pair['pair_id']}: candidates must use distinct trackers")
    local_ranges = {
        (
            int(candidate["local_start_frame"]),
            int(candidate["local_end_frame"]),
            int(candidate["num_frames"]),
        )
        for candidate in pair["candidates"]
    }
    if len(local_ranges) != 1:
        raise ValueError(f"pair {pair['pair_id']}: candidate local ranges differ")
    for candidate in pair["candidates"]:
        if candidate["motion_id"] != pair["motion_id"]:
            raise ValueError(f"pair {pair['pair_id']}: candidate motion mismatch")
        if int(candidate["source_start_frame"]) != int(pair["source_start_frame"]):
            raise ValueError(f"pair {pair['pair_id']}: candidate start frame mismatch")
        if int(candidate["source_end_frame"]) != int(pair["source_end_frame"]):
            raise ValueError(f"pair {pair['pair_id']}: candidate end frame mismatch")
        if int(candidate["fps"]) != int(pair["fps"]):
            raise ValueError(f"pair {pair['pair_id']}: candidate fps mismatch")
    return pair


def _validate_display_order(value: list[int]) -> list[int]:
    order = [int(item) for item in value]
    if order not in ([0, 1], [1, 0]):
        raise ValueError(f"display_order must be [0, 1] or [1, 0]: {value}")
    return order


def _require_identifier(value: str, name: str) -> str:
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{name} must not be empty")
    return identifier


def load_pairs(pairs_file: str | Path) -> list[dict[str, Any]]:
    path = _resolve_path(pairs_file)
    records = [_normalize_pair(row) for row in _read_rows(path)]
    if not records:
        raise ValueError(f"no pair records in {path}")
    for expected_idx, record in enumerate(records):
        if int(record["pair_idx"]) != expected_idx:
            raise ValueError(
                f"{path}: pair_idx {record['pair_idx']} at row {expected_idx}"
            )
    pair_ids = [str(record["pair_id"]) for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"duplicate pair_id in {path}")
    return records


def get_frame_range(record: dict[str, Any]) -> tuple[int, int]:
    candidate = record["candidates"][0]
    start = int(candidate["local_start_frame"])
    end = int(candidate["local_end_frame"])
    if end - start != int(candidate["num_frames"]):
        raise ValueError(f"pair {record['pair_id']}: inconsistent local frame metadata")
    return start, end


class DynamicTaskManager:
    """Lease each canonical pair until it reaches the annotation target."""

    def __init__(
        self,
        pairs_file: str | Path,
        state_file: str | Path,
        annotations_per_pair: int = 3,
        lease_seconds: int = 120,
    ) -> None:
        if annotations_per_pair < 1:
            raise ValueError("annotations_per_pair must be positive")
        if lease_seconds < 60:
            raise ValueError("lease_seconds must be at least 60")
        self.task_path = _resolve_path(pairs_file)
        self.state_path = _resolve_path(state_file)
        self.annotations_per_pair = int(annotations_per_pair)
        self.lease_seconds = int(lease_seconds)
        self.annotators: list[str] = []
        self._candidate_orders: dict[str, list[int]] = {}
        self._lock = threading.Lock()

        self.records = load_pairs(self.task_path)
        pair_ids = [str(record["pair_id"]) for record in self.records]
        self._pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
        self.state = self._load_state()

    def _empty_state(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "pairs_path": str(self.task_path),
            "annotations_per_pair": self.annotations_per_pair,
            "lease_seconds": self.lease_seconds,
            "created_at": now,
            "updated_at": now,
            "completed": {},
            "leases": {},
            "undo_history": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = self._empty_state()
            self._save_state(state)
            return state
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        expected = {
            "pairs_path": str(self.task_path),
            "annotations_per_pair": self.annotations_per_pair,
            "lease_seconds": self.lease_seconds,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"{self.state_path}: {key}={state.get(key)!r}, expected {value!r}"
                )
        if not isinstance(state.get("completed"), dict) or not isinstance(state.get("leases"), dict):
            raise TypeError(f"{self.state_path}: completed and leases must be objects")
        state.setdefault("undo_history", {})
        if not isinstance(state["undo_history"], dict):
            raise TypeError(f"{self.state_path}: undo_history must be an object")
        self._prune_expired(state, time.time())
        self._validate_state(state)
        self._save_state(state)
        return state

    def _validate_state(self, state: dict[str, Any]) -> None:
        known_pair_ids = set(self._pair_index)
        unknown_completed = sorted(set(state["completed"]) - known_pair_ids)
        if unknown_completed:
            raise ValueError(f"{self.state_path}: unknown completed pairs {unknown_completed}")
        unknown_lease_pairs = sorted({
            str(lease["pair_id"])
            for lease in state["leases"].values()
            if str(lease["pair_id"]) not in known_pair_ids
        })
        if unknown_lease_pairs:
            raise ValueError(f"{self.state_path}: unknown leased pairs {unknown_lease_pairs}")

        active_by_pair: dict[str, list[dict[str, Any]]] = {}
        for lease_id, lease in state["leases"].items():
            if lease.get("lease_id") != lease_id:
                raise ValueError(f"{self.state_path}: lease id mismatch for {lease_id}")
            pair_id = str(lease["pair_id"])
            pair_index = int(lease["pair_index"])
            if pair_index != self._pair_index[pair_id]:
                raise ValueError(f"{self.state_path}: pair index mismatch for {lease_id}")
            if not str(lease["annotator_id"]).strip():
                raise ValueError(f"{self.state_path}: empty annotator for {lease_id}")
            if not str(lease["client_id"]).strip():
                raise ValueError(f"{self.state_path}: empty client for {lease_id}")
            _validate_display_order(lease["display_order"])
            float(lease["expires_at"])
            active_by_pair.setdefault(pair_id, []).append(lease)

        for pair_id in known_pair_ids:
            completed = state["completed"].get(pair_id, [])
            active = active_by_pair.get(pair_id, [])
            if not isinstance(completed, list):
                raise TypeError(f"{self.state_path}: completed[{pair_id!r}] must be a list")
            completed_annotators = []
            for item in completed:
                annotator_id = str(item["annotator_id"])
                _require_identifier(str(item["client_id"]), "client_id")
                if "record_id" in item:
                    _require_identifier(str(item["record_id"]), "record_id")
                label = str(item["label"])
                display_order = _validate_display_order(item["display_order"])
                expected_preference = {
                    "choice_type": "preference" if label in ("left", "right") else label,
                    "preferred_candidate_idx": self.resolve_preferred_candidate_idx(
                        label, display_order
                    ),
                }
                if item["human_preference"] != expected_preference:
                    raise ValueError(
                        f"{self.state_path}: invalid human_preference for {pair_id}"
                    )
                completed_annotators.append(annotator_id)
            active_annotators = [str(lease["annotator_id"]) for lease in active]
            all_annotators = completed_annotators + active_annotators
            if len(all_annotators) != len(set(all_annotators)):
                raise ValueError(f"{self.state_path}: duplicate annotator for {pair_id}")
            if len(completed) + len(active) > self.annotations_per_pair:
                raise ValueError(f"{self.state_path}: annotation target exceeded for {pair_id}")

        active_clients = [str(lease["client_id"]) for lease in state["leases"].values()]
        if len(active_clients) != len(set(active_clients)):
            raise ValueError(f"{self.state_path}: a client owns multiple active leases")

        completed_by_record = {
            str(item["record_id"]): (pair_id, str(item["annotator_id"]))
            for pair_id, items in state["completed"].items()
            for item in items
            if "record_id" in item
        }
        if len(completed_by_record) != sum(
            "record_id" in item for items in state["completed"].values() for item in items
        ):
            raise ValueError(f"{self.state_path}: duplicate completed record_id")
        for annotator_id, history in state["undo_history"].items():
            _require_identifier(str(annotator_id), "annotator_id")
            if not isinstance(history, list) or len(history) > UNDO_LIMIT:
                raise ValueError(
                    f"{self.state_path}: undo history for {annotator_id!r} exceeds {UNDO_LIMIT}"
                )
            for item in history:
                record_id = _require_identifier(str(item["record_id"]), "record_id")
                pair_id = _require_identifier(str(item["pair_id"]), "pair_id")
                if completed_by_record.get(record_id) != (pair_id, annotator_id):
                    raise ValueError(
                        f"{self.state_path}: undo record {record_id!r} is not active"
                    )

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        target = self.state if state is None else state
        target["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(target, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.state_path)

    @staticmethod
    def _prune_expired(state: dict[str, Any], now: float) -> bool:
        expired = [
            lease_id
            for lease_id, lease in state["leases"].items()
            if float(lease["expires_at"]) <= now
        ]
        for lease_id in expired:
            del state["leases"][lease_id]
        return bool(expired)

    def _completed_for(self, pair_id: str) -> list[dict[str, Any]]:
        return self.state["completed"].get(pair_id, [])

    def _active_leases_for(self, pair_id: str) -> list[dict[str, Any]]:
        return [
            lease for lease in self.state["leases"].values() if lease["pair_id"] == pair_id
        ]

    def _candidate_indices(
        self,
        annotator_id: str,
        limit: int | None = None,
        state: dict[str, Any] | None = None,
    ) -> list[int]:
        target = self.state if state is None else state
        active_by_pair: dict[str, list[dict[str, Any]]] = {}
        for lease in target["leases"].values():
            active_by_pair.setdefault(str(lease["pair_id"]), []).append(lease)
        indices = []
        for index in self._candidate_order(annotator_id):
            record = self.records[index]
            pair_id = str(record["pair_id"])
            completed = target["completed"].get(pair_id, [])
            active = active_by_pair.get(pair_id, [])
            if any(item["annotator_id"] == annotator_id for item in completed):
                continue
            if any(item["annotator_id"] == annotator_id for item in active):
                continue
            if len(completed) + len(active) >= self.annotations_per_pair:
                continue
            indices.append(index)
            if limit is not None and len(indices) == limit:
                break
        return indices

    def _candidate_order(self, annotator_id: str) -> list[int]:
        order = self._candidate_orders.get(annotator_id)
        if order is None:
            order = sorted(
                range(self.total_pairs),
                key=lambda index: hashlib.blake2s(
                    f"task\0{annotator_id}\0{self.records[index]['pair_id']}".encode(),
                    digest_size=16,
                ).digest(),
            )
            self._candidate_orders[annotator_id] = order
        return order

    @property
    def total_pairs(self) -> int:
        return len(self.records)

    def get_annotators(self) -> list[str]:
        return list(self.annotators)

    def display_order_for(self, index: int, annotator_id: str) -> list[int]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        pair_id = str(self.records[index]["pair_id"])
        side = hashlib.blake2s(
            f"{annotator_id}\0{pair_id}".encode(), digest_size=1
        ).digest()[0] & 1
        return [side, 1 - side]

    def get_pair_for_display_order(
        self,
        index: int,
        display_order: list[int],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
        order = _validate_display_order(display_order)
        record = self.records[index]
        candidates = record["candidates"]
        return candidates[order[0]], candidates[order[1]], record, order

    def get_pair_for_lease(
        self,
        lease_id: str,
        annotator_id: str,
        client_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        with self._lock:
            self._prune_expired(self.state, time.time())
            lease = self.state["leases"].get(lease_id)
            if lease is None:
                raise ValueError("lease expired")
            if (
                lease["annotator_id"] != annotator_id
                or lease["client_id"] != client_id
            ):
                raise ValueError("lease belongs to another client")
            return self.get_pair_for_display_order(
                int(lease["pair_index"]),
                _validate_display_order(lease["display_order"]),
            )

    def get_pair(
        self,
        index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
        return self.get_pair_for_display_order(index, [0, 1])

    @staticmethod
    def get_npz_path(candidate: dict[str, Any]) -> Path:
        return Path(candidate["traj_path"])

    @staticmethod
    def get_frame_range(record: dict[str, Any]) -> tuple[int, int]:
        return get_frame_range(record)

    @staticmethod
    def resolve_preferred_candidate_idx(label: str, display_order: list[int]) -> int | None:
        if label not in VALID_LABELS:
            raise ValueError(f"invalid label: {label}")
        order = _validate_display_order(display_order)
        if label in ("similar", "bad_traj"):
            return None
        return order[0] if label == "left" else order[1]

    def _new_lease(
        self,
        index: int,
        annotator_id: str,
        client_id: str,
        now: float,
        display_order: list[int] | None = None,
        state: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        target = self.state if state is None else state
        lease_id = uuid.uuid4().hex
        if display_order is None:
            display_order = self.display_order_for(index, annotator_id)
        else:
            display_order = _validate_display_order(display_order)
        lease = {
            "lease_id": lease_id,
            "pair_id": str(self.records[index]["pair_id"]),
            "pair_index": index,
            "annotator_id": annotator_id,
            "client_id": client_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": now + self.lease_seconds,
            "display_order": display_order,
        }
        target["leases"][lease_id] = lease
        return lease_id, lease

    def lease_next(self, annotator_id: str, client_id: str) -> dict[str, Any] | None:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        now = time.time()
        with self._lock:
            pruned = self._prune_expired(self.state, now)
            for lease_id, lease in self.state["leases"].items():
                if lease["client_id"] != client_id:
                    continue
                if lease["annotator_id"] != annotator_id:
                    raise ValueError("client already has a lease for another annotator")
                lease["expires_at"] = now + self.lease_seconds
                return self._lease_payload(lease_id, lease)
            candidates = self._candidate_indices(annotator_id, limit=1)
            if not candidates:
                if pruned:
                    self._save_state()
                return None
            lease_id, lease = self._new_lease(
                candidates[0], annotator_id, client_id, now
            )
            self._save_state()
            return self._lease_payload(lease_id, lease)

    def prefetch_indices(self, annotator_id: str, limit: int = 2) -> list[int]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            self._prune_expired(self.state, time.time())
            return self._candidate_indices(annotator_id, limit=limit)

    def prefetch_snapshot(
        self,
        annotator_id: str,
        limit: int = 2,
    ) -> tuple[list[tuple[int, list[int]]], dict[str, int]]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            self._prune_expired(self.state, time.time())
            indices = self._candidate_indices(annotator_id, limit=limit)
            previews = [
                (index, self.display_order_for(index, annotator_id))
                for index in indices
            ]
            return previews, self._stats_unlocked()

    def renew_lease(
        self,
        lease_id: str,
        annotator_id: str,
        client_id: str,
    ) -> dict[str, Any]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        now = time.time()
        with self._lock:
            self._prune_expired(self.state, now)
            lease = self.state["leases"].get(lease_id)
            if lease is None:
                raise ValueError("lease expired")
            if (
                lease["annotator_id"] != annotator_id
                or lease["client_id"] != client_id
            ):
                raise ValueError("lease belongs to another client")
            lease["expires_at"] = now + self.lease_seconds
            return self._lease_payload(lease_id, lease)

    def release_lease(
        self,
        lease_id: str,
        annotator_id: str,
        client_id: str,
    ) -> bool:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        with self._lock:
            changed = self._prune_expired(self.state, time.time())
            lease = self.state["leases"].get(lease_id)
            if lease is None:
                if changed:
                    self._save_state()
                return False
            if (
                lease["annotator_id"] != annotator_id
                or lease["client_id"] != client_id
            ):
                raise ValueError("lease belongs to another client")
            del self.state["leases"][lease_id]
            self._save_state()
            return True

    def _complete_lease_unlocked(
        self,
        state: dict[str, Any],
        lease_id: str,
        annotator_id: str,
        client_id: str,
        label: str,
        record_id: str,
    ) -> None:
        lease = state["leases"].get(lease_id)
        if lease is None:
            raise ValueError("lease expired")
        if (
            lease["annotator_id"] != annotator_id
            or lease["client_id"] != client_id
        ):
            raise ValueError("lease belongs to another client")
        if any(
            item.get("record_id") == record_id
            for items in state["completed"].values()
            for item in items
        ):
            raise ValueError(f"duplicate completed record_id {record_id!r}")
        pair_id = str(lease["pair_id"])
        completed = list(state["completed"].get(pair_id, []))
        if any(item["annotator_id"] == annotator_id for item in completed):
            raise ValueError(f"annotator {annotator_id} already completed {pair_id}")
        if len(completed) >= self.annotations_per_pair:
            raise ValueError(f"annotation target exceeded for {pair_id}")
        display_order = _validate_display_order(lease["display_order"])
        preferred_candidate_idx = self.resolve_preferred_candidate_idx(label, display_order)
        completion = {
            "annotator_id": annotator_id,
            "client_id": client_id,
            "lease_id": lease_id,
            "record_id": record_id,
            "label": label,
            "display_order": display_order,
            "human_preference": {
                "choice_type": "preference" if label in ("left", "right") else label,
                "preferred_candidate_idx": preferred_candidate_idx,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        completed.append(completion)
        state["completed"][pair_id] = completed
        history = list(state["undo_history"].get(annotator_id, []))
        history.append({
            "record_id": record_id,
            "pair_id": pair_id,
            "client_id": client_id,
            "timestamp": completion["timestamp"],
        })
        del history[:-UNDO_LIMIT]
        state["undo_history"][annotator_id] = history
        del state["leases"][lease_id]

    def _copy_state(self) -> dict[str, Any]:
        return {
            **self.state,
            "completed": dict(self.state["completed"]),
            "leases": dict(self.state["leases"]),
            "undo_history": dict(self.state["undo_history"]),
        }

    def complete_lease(
        self,
        lease_id: str,
        annotator_id: str,
        client_id: str,
        label: str,
        record_id: str,
    ) -> None:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        record_id = _require_identifier(record_id, "record_id")
        with self._lock:
            next_state = self._copy_state()
            self._prune_expired(next_state, time.time())
            self._complete_lease_unlocked(
                next_state, lease_id, annotator_id, client_id, label, record_id
            )
            self._save_state(next_state)
            self.state = next_state

    def complete_and_lease_next(
        self,
        lease_id: str,
        annotator_id: str,
        client_id: str,
        label: str,
        record_id: str,
    ) -> dict[str, Any]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        record_id = _require_identifier(record_id, "record_id")
        with self._lock:
            now = time.time()
            next_state = self._copy_state()
            self._prune_expired(next_state, now)
            self._complete_lease_unlocked(
                next_state, lease_id, annotator_id, client_id, label, record_id
            )
            candidates = self._candidate_indices(
                annotator_id, limit=1, state=next_state
            )
            next_lease = None
            if candidates:
                next_lease_id, lease = self._new_lease(
                    candidates[0], annotator_id, client_id, now, state=next_state
                )
                next_lease = self._lease_payload(next_lease_id, lease)
            self._save_state(next_state)
            self.state = next_state
            return {
                "lease": next_lease,
                "stats": self._stats_unlocked(),
                "undo_depth": len(next_state["undo_history"].get(annotator_id, [])),
            }

    def undo_depth(self, annotator_id: str) -> int:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        with self._lock:
            return len(self.state["undo_history"].get(annotator_id, []))

    def peek_undo(self, annotator_id: str) -> dict[str, Any]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        with self._lock:
            history = self.state["undo_history"].get(annotator_id, [])
            if not history:
                raise ValueError("no annotation available to undo")
            return dict(history[-1])

    def undo_last(
        self,
        annotator_id: str,
        client_id: str,
        expected_record_id: str,
    ) -> dict[str, Any]:
        annotator_id = _require_identifier(annotator_id, "annotator_id")
        client_id = _require_identifier(client_id, "client_id")
        expected_record_id = _require_identifier(expected_record_id, "record_id")
        with self._lock:
            self._prune_expired(self.state, time.time())
            history = self.state["undo_history"].get(annotator_id, [])
            if not history or history[-1]["record_id"] != expected_record_id:
                raise ValueError("undo history changed")
            target = history[-1]
            pair_id = str(target["pair_id"])
            completed = self.state["completed"].get(pair_id, [])
            matches = [
                item
                for item in completed
                if item.get("record_id") == expected_record_id
                and item["annotator_id"] == annotator_id
            ]
            if len(matches) != 1:
                raise ValueError(f"completion {expected_record_id!r} is not active")
            completion = matches[0]

            owned_leases = [
                lease_id
                for lease_id, lease in self.state["leases"].items()
                if lease["client_id"] == client_id
            ]
            for lease_id in owned_leases:
                lease = self.state["leases"][lease_id]
                if lease["annotator_id"] != annotator_id:
                    raise ValueError("client already has a lease for another annotator")
                del self.state["leases"][lease_id]

            completed.remove(completion)
            if not completed:
                del self.state["completed"][pair_id]
            history.pop()
            if not history:
                del self.state["undo_history"][annotator_id]

            index = self._pair_index[pair_id]
            lease_id, lease = self._new_lease(
                index,
                annotator_id,
                client_id,
                time.time(),
                display_order=completion["display_order"],
            )
            self._validate_state(self.state)
            self._save_state()
            return self._lease_payload(lease_id, lease)

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._prune_expired(self.state, time.time())
            return self._stats_unlocked()

    def _stats_unlocked(self) -> dict[str, int]:
        done_pairs = sum(
            len(self.state["completed"].get(str(record["pair_id"]), []))
            >= self.annotations_per_pair
            for record in self.records
        )
        done_annotations = sum(len(items) for items in self.state["completed"].values())
        return {
            "pairs_done": done_pairs,
            "pairs_total": self.total_pairs,
            "annotations_done": done_annotations,
            "annotations_total": self.total_pairs * self.annotations_per_pair,
            "active_leases": len(self.state["leases"]),
        }

    def _lease_payload(self, lease_id: str, lease: dict[str, Any]) -> dict[str, Any]:
        record = self.records[int(lease["pair_index"])]
        return {
            "lease_id": lease_id,
            "idx": int(lease["pair_index"]),
            "pair_id": str(record["pair_id"]),
            "motion_id": str(record["motion_id"]),
            "clip_uid": str(record["clip_uid"]),
            "expires_at": float(lease["expires_at"]),
            "annotator_id": str(lease["annotator_id"]),
            "client_id": str(lease["client_id"]),
            "display_order": _validate_display_order(lease["display_order"]),
        }
