from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from utils.data_collector.hf_recorder import HFRecorder
from utils.hf_proto import (
    HFComparison,
    HFFlags,
    HFMeta,
    HFPreference,
    HFRecord,
    HFVideoSide,
)
from tool.rm_pipeline.annotations import aggregate_annotations, export_rm_parquet
from web_labeler.combined_video import CRF, FILTER_GRAPH, cache_filename
from web_labeler.task_manager import DynamicTaskManager


def _candidate(path: Path, candidate_idx: int, tracker: str) -> dict:
    return {
        "candidate_idx": candidate_idx,
        "candidate_uid": f"candidate_{candidate_idx}",
        "tracker": tracker,
        "traj_path": str(path),
        "motion_id": "motion_0",
        "source_start_frame": 0,
        "source_end_frame": 10,
        "local_start_frame": 0,
        "local_end_frame": 10,
        "num_frames": 10,
        "fps": 50,
    }


def _pair(left: Path, right: Path, pair_idx: int = 0) -> dict:
    return {
        "pair_idx": pair_idx,
        "pair_id": f"pair_{pair_idx}",
        "motion_idx": 0,
        "motion_id": "motion_0",
        "clip_idx": pair_idx,
        "clip_uid": f"clip_{pair_idx}",
        "source_start_frame": 0,
        "source_end_frame": 10,
        "fps": 50,
        "category": "Daily",
        "tracker_pair_key": "hgpt|sonic",
        "candidates": [_candidate(left, 0, "hgpt"), _candidate(right, 1, "sonic")],
    }


def _record(left: Path, right: Path) -> HFRecord:
    def side(path: Path, candidate_idx: int, tracker: str) -> HFVideoSide:
        return HFVideoSide(
            npz_name=path.name,
            start_frame=0,
            end_frame=10,
            policy=tracker,
            extra={
                "candidate_idx": candidate_idx,
                "candidate_uid": f"candidate_{candidate_idx}",
                "tracker": tracker,
                "traj_path": str(path),
                "clip_uid": "clip_0",
                "source_start_frame": 0,
                "source_end_frame": 10,
                "fps": 50,
            },
        )

    return HFRecord(
        meta=HFMeta(
            record_id="record_0",
            annotator_id="alice",
            timestamp="2026-07-20T00:00:00+00:00",
            tool="web_labeler",
            task="G1-TrackMJ",
            scene="flat_ground",
            fps=50,
        ),
        video_left=side(left, 1, "sonic"),
        video_right=side(right, 0, "hgpt"),
        preference=HFPreference(winner="left"),
        flags=HFFlags(),
        comparison=HFComparison(
            length_frames=10,
            extra={
                "pair_idx": 0,
                "pair_id": "pair_0",
                "motion_idx": 0,
                "motion_id": "motion_0",
                "category": "Daily",
                "tracker_pair_key": "hgpt|sonic",
                "human_preference": {
                    "choice_type": "preference",
                    "preferred_candidate_idx": 1,
                },
                "display_order": [1, 0],
                "display_choice_idx": 0,
            },
        ),
    )


@contextmanager
def expect_error(error_type: type[Exception], pattern: str):
    try:
        yield
    except error_type as error:
        if re.search(pattern, str(error)) is None:
            raise AssertionError(f"{error!r} does not match {pattern!r}") from error
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def check_dynamic_task_manager(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, joint_pos=np.zeros((10, 2)))
    np.savez(right, joint_pos=np.zeros((10, 2)))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(json.dumps(_pair(left, right, pair_idx)) + "\n" for pair_idx in range(2)),
        encoding="utf-8",
    )

    manager = DynamicTaskManager(pairs, tmp_path / "state.json", annotations_per_pair=1)
    assert len(manager.prefetch_indices("alice", limit=2)) == 2
    assert manager.stats()["active_leases"] == 0

    alice = manager.lease_next("alice", "client_a")
    assert alice is not None
    same_alice = manager.lease_next("alice", "client_a")
    assert same_alice is not None
    assert same_alice["lease_id"] == alice["lease_id"]
    with expect_error(ValueError, "another annotator"):
        manager.lease_next("other_name", "client_a")

    bob = manager.lease_next("bob", "client_b")
    assert bob is not None
    assert bob["pair_id"] != alice["pair_id"]
    with expect_error(ValueError, "another client"):
        manager.get_pair_for_lease(alice["lease_id"], "alice", "client_b")

    assert manager.release_lease(alice["lease_id"], "alice", "client_a")
    assert not manager.release_lease(alice["lease_id"], "alice", "client_a")
    charlie = manager.lease_next("charlie", "client_c")
    assert charlie is not None
    assert charlie["pair_id"] == alice["pair_id"]
    previous_expiry = charlie["expires_at"]
    renewed = manager.renew_lease(charlie["lease_id"], "charlie", "client_c")
    assert renewed["expires_at"] >= previous_expiry
    manager.complete_lease(
        charlie["lease_id"], "charlie", "client_c", "left", "record_charlie"
    )

    manager.state["leases"][bob["lease_id"]]["expires_at"] = 0
    dana = manager.lease_next("dana", "client_d")
    assert dana is not None
    assert dana["pair_id"] == bob["pair_id"]
    manager.complete_lease(
        dana["lease_id"], "dana", "client_d", "right", "record_dana"
    )
    assert manager.stats()["annotations_done"] == 2
    assert manager.stats()["active_leases"] == 0
    target = manager.peek_undo("charlie")
    assert target == {
        "record_id": "record_charlie",
        "pair_id": charlie["pair_id"],
        "client_id": "client_c",
        "timestamp": target["timestamp"],
    }
    correction = manager.undo_last(
        "charlie", "client_c", target["record_id"]
    )
    assert correction["pair_id"] == charlie["pair_id"]
    assert manager.undo_depth("charlie") == 0
    assert manager.state["completed"][dana["pair_id"]][0]["record_id"] == "record_dana"
    manager.complete_lease(
        correction["lease_id"],
        "charlie",
        "client_c",
        "right",
        "record_charlie_corrected",
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["annotations_per_pair"] == 1


def check_hf_recorder(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, joint_pos=np.zeros((10, 2)))
    np.savez(right, joint_pos=np.zeros((10, 2)))

    recorder = HFRecorder(tmp_path / "annotations", shard_size=1)
    shard = recorder.append(_record(left, right))
    assert shard is not None
    index = json.loads(recorder.index_path.read_text(encoding="utf-8"))
    assert index["total_records"] == 1
    annotation = json.loads(pq.read_table(shard)["annotation_json"][0].as_py())
    assert [item["candidate_idx"] for item in annotation["candidates"]] == [0, 1]
    assert annotation["preference"]["preferred_candidate_idx"] == 1

    aggregates_path = tmp_path / "aggregates.jsonl"
    aggregates = aggregate_annotations(
        [str(recorder.out_dir)],
        aggregates_path,
        min_annotations=1,
        min_agreement=1,
    )
    assert aggregates[0]["chosen"]["candidate_idx"] == 1
    output = tmp_path / "rm_pairs.parquet"
    assert export_rm_parquet(aggregates_path, output) == 1
    assert pq.ParquetFile(output).metadata.num_rows == 1

    broken = _record(left, right)
    del broken.video_left.extra["candidate_idx"]
    with expect_error(KeyError, "candidate_idx"):
        recorder.append(broken)


def check_hf_recorder_undo(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, joint_pos=np.zeros((10, 2)))
    np.savez(right, joint_pos=np.zeros((10, 2)))

    def record(record_id: str, annotator: str, pair_id: str) -> HFRecord:
        value = _record(left, right)
        value.meta.record_id = record_id
        value.meta.annotator_id = annotator
        value.comparison.extra["pair_id"] = pair_id
        return value

    output = tmp_path / "undo_shards"
    recorder = HFRecorder(output, shard_size=3, undo_limit=2)
    for index in range(5):
        recorder.append(record(f"alice_{index}", "alice", f"alice_pair_{index}"))
    assert recorder.pending_count == 2
    for index in range(5):
        recorder.append(record(f"bob_{index}", "bob", f"bob_pair_{index}"))
    assert [item["count"] for item in json.loads(recorder.index_path.read_text())["files"]] == [3, 3]
    assert recorder.pending_count == 4

    recorder.undo("bob_4", "bob", "bob_pair_4")
    recorder.undo("alice_4", "alice", "alice_pair_4")
    assert recorder.get_total_count() == 8
    active_ids = {item["record_id"] for item in recorder.load_all_records()}
    assert "bob_4" not in active_ids
    assert "alice_4" not in active_ids
    assert "alice_3" in active_ids
    with expect_error(ValueError, "not pending"):
        recorder.undo("alice_2", "alice", "alice_pair_2")

    restarted = HFRecorder(output, shard_size=3, undo_limit=2)
    restarted.undo("alice_3", "alice", "alice_pair_3")
    assert restarted.get_total_count() == 7


def check_hf_recorder_shards(tmp_path: Path) -> None:
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left, joint_pos=np.zeros((10, 2)))
    np.savez(right, joint_pos=np.zeros((10, 2)))

    output = tmp_path / "sharded"
    recorder = HFRecorder(output)
    for record_idx in range(199):
        record = _record(left, right)
        record.meta.record_id = f"record_{record_idx}"
        assert recorder.append(record) is None
    assert recorder.pending_count == 199
    assert recorder.pending_path.is_file()
    assert not list(output.glob("*.parquet"))

    restarted = HFRecorder(output)
    assert restarted.pending_count == 199
    record = _record(left, right)
    record.meta.record_id = "record_199"
    full_shard = restarted.append(record)
    assert full_shard is not None
    assert pq.ParquetFile(full_shard).metadata.num_rows == 200
    assert restarted.pending_count == 0
    assert not restarted.pending_path.exists()

    tail = _record(left, right)
    tail.meta.record_id = "record_200"
    assert restarted.append(tail) is None
    assert restarted.close() is None
    resumed = HFRecorder(output)
    assert resumed.pending_count == 1
    assert resumed.get_total_count() == 201
    index = json.loads(restarted.index_path.read_text(encoding="utf-8"))
    assert index["total_records"] == 200
    assert [item["count"] for item in index["files"]] == [200]

    recovery_output = tmp_path / "recovery"
    interrupted = HFRecorder(recovery_output, shard_size=3)
    for record_idx in range(2):
        record = _record(left, right)
        record.meta.record_id = f"recovery_{record_idx}"
        assert interrupted.append(record) is None

    def fail_index_write(index: dict) -> None:
        raise RuntimeError("simulated index interruption")

    interrupted._write_index = fail_index_write
    third = _record(left, right)
    third.meta.record_id = "recovery_2"
    with expect_error(RuntimeError, "simulated index interruption"):
        interrupted.append(third)
    assert interrupted.commit_path.is_file()

    recovered = HFRecorder(recovery_output, shard_size=3)
    assert recovered.get_total_count() == 3
    assert recovered.pending_count == 0
    assert not recovered.commit_path.exists()
    recovery_index = json.loads(recovered.index_path.read_text(encoding="utf-8"))
    assert [item["count"] for item in recovery_index["files"]] == [3]

    protected_output = tmp_path / "protected_close"
    protected = HFRecorder(protected_output, shard_size=3, undo_limit=2)
    for record_idx in range(3):
        record = _record(left, right)
        record.meta.record_id = f"protected_{record_idx}"
        assert protected.append(record) is None
    final_shard = protected.close()
    assert final_shard is not None
    assert pq.ParquetFile(final_shard).metadata.num_rows == 3
    assert protected.pending_count == 0


class StrictAnnotationTest(unittest.TestCase):
    def test_combined_video_contract(self) -> None:
        filename = cache_filename("scene", "left.mp4", "right.mp4")
        self.assertRegex(filename, r"^combined_[0-9a-f]{12}\.mp4$")
        self.assertEqual(CRF, 26)
        self.assertIn("scale=480:360", FILTER_GRAPH)
        self.assertEqual(filename, cache_filename("scene", "left.mp4", "right.mp4"))
        self.assertNotEqual(filename, cache_filename("scene", "right.mp4", "left.mp4"))

    def test_dynamic_task_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            check_dynamic_task_manager(Path(directory))

    def test_hf_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            check_hf_recorder(Path(directory))

    def test_hf_recorder_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            check_hf_recorder_shards(Path(directory))

    def test_hf_recorder_undo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            check_hf_recorder_undo(Path(directory))

    def test_five_step_undo_is_per_annotator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, joint_pos=np.zeros((10, 2)))
            np.savez(right, joint_pos=np.zeros((10, 2)))
            pairs = root / "pairs.jsonl"
            pairs.write_text(
                "".join(json.dumps(_pair(left, right, index)) + "\n" for index in range(8)),
                encoding="utf-8",
            )
            manager = DynamicTaskManager(pairs, root / "state.json", annotations_per_pair=1)
            for index in range(6):
                lease = manager.lease_next("alice", "alice_client")
                assert lease is not None
                manager.complete_lease(
                    lease["lease_id"],
                    "alice",
                    "alice_client",
                    "left",
                    f"alice_record_{index}",
                )
            bob = manager.lease_next("bob", "bob_client")
            assert bob is not None
            manager.complete_lease(
                bob["lease_id"], "bob", "bob_client", "right", "bob_record"
            )

            self.assertEqual(manager.undo_depth("alice"), 5)
            for expected in reversed(range(1, 6)):
                target = manager.peek_undo("alice")
                self.assertEqual(target["record_id"], f"alice_record_{expected}")
                manager.undo_last("alice", "alice_client", target["record_id"])
            self.assertEqual(manager.undo_depth("alice"), 0)
            self.assertEqual(manager.undo_depth("bob"), 1)
            self.assertEqual(
                manager.state["completed"][bob["pair_id"]][0]["record_id"],
                "bob_record",
            )
            with expect_error(ValueError, "no annotation"):
                manager.peek_undo("alice")

    def test_concurrent_clients_reclaim_released_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, joint_pos=np.zeros((10, 2)))
            np.savez(right, joint_pos=np.zeros((10, 2)))
            pairs = root / "pairs.jsonl"
            pairs.write_text(
                "".join(
                    json.dumps(_pair(left, right, pair_idx)) + "\n"
                    for pair_idx in range(24)
                ),
                encoding="utf-8",
            )
            manager = DynamicTaskManager(
                pairs, root / "state.json", annotations_per_pair=1
            )

            def lease(client_number: int) -> dict:
                result = manager.lease_next(
                    f"annotator_{client_number}", f"client_{client_number}"
                )
                assert result is not None
                return result

            with ThreadPoolExecutor(max_workers=12) as executor:
                leases = list(executor.map(lease, range(24)))
            self.assertEqual(len({item["lease_id"] for item in leases}), 24)
            self.assertEqual(len({item["pair_id"] for item in leases}), 24)

            def release(client_number: int) -> bool:
                item = leases[client_number]
                return manager.release_lease(
                    item["lease_id"],
                    f"annotator_{client_number}",
                    f"client_{client_number}",
                )

            with ThreadPoolExecutor(max_workers=12) as executor:
                self.assertTrue(all(executor.map(release, range(12))))
                reclaimed = list(executor.map(lease, range(24, 36)))
            self.assertEqual(
                {item["pair_id"] for item in reclaimed},
                {item["pair_id"] for item in leases[:12]},
            )
            self.assertEqual(manager.stats()["active_leases"], 24)

    def test_prefetch_order_matches_atomic_next_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, joint_pos=np.zeros((10, 2)))
            np.savez(right, joint_pos=np.zeros((10, 2)))
            pairs = root / "pairs.jsonl"
            pairs.write_text(
                "".join(json.dumps(_pair(left, right, index)) + "\n" for index in range(2000)),
                encoding="utf-8",
            )
            manager = DynamicTaskManager(
                pairs, root / "state.json", annotations_per_pair=1
            )
            previews, _ = manager.prefetch_snapshot("alice", limit=8)
            lease = manager.lease_next("alice", "alice_client")
            assert lease is not None
            self.assertEqual(dict(previews)[lease["idx"]], lease["display_order"])

            orders = [manager.display_order_for(index, "alice") for index in range(2000)]
            forward = sum(order == [0, 1] for order in orders)
            self.assertLess(abs(forward / len(orders) - 0.5), 0.03)

            writes = 0
            save_state = manager._save_state

            def counted_save(state=None):
                nonlocal writes
                writes += 1
                save_state(state)

            manager._save_state = counted_save
            transition = manager.complete_and_lease_next(
                lease["lease_id"],
                "alice",
                "alice_client",
                "left",
                "record_atomic",
            )
            self.assertEqual(writes, 1)
            self.assertIsNotNone(transition["lease"])
            self.assertEqual(transition["stats"]["annotations_done"], 1)
            self.assertEqual(transition["stats"]["active_leases"], 1)

            separate = DynamicTaskManager(
                pairs, root / "separate_state.json", annotations_per_pair=1
            )
            expected = {}
            for index in range(6):
                annotator = f"annotator_{index}"
                snapshot, _ = separate.prefetch_snapshot(annotator, limit=8)
                expected[annotator] = [pair_index for pair_index, _ in snapshot]
            self.assertEqual(
                len(set().union(*(set(indices) for indices in expected.values()))),
                48,
            )
            for index, (annotator, indices) in enumerate(expected.items()):
                lease = separate.lease_next(annotator, f"client_{index}")
                assert lease is not None
                self.assertEqual(lease["idx"], indices[0])

    def test_six_annotators_submit_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, joint_pos=np.zeros((10, 2)))
            np.savez(right, joint_pos=np.zeros((10, 2)))
            pairs = root / "pairs.jsonl"
            pairs.write_text(
                "".join(json.dumps(_pair(left, right, index)) + "\n" for index in range(24)),
                encoding="utf-8",
            )
            manager = DynamicTaskManager(
                pairs, root / "state.json", annotations_per_pair=1
            )
            leases = [
                manager.lease_next(f"annotator_{index}", f"client_{index}")
                for index in range(6)
            ]
            self.assertTrue(all(lease is not None for lease in leases))

            writes = 0
            writes_lock = threading.Lock()
            save_state = manager._save_state

            def counted_save(state=None):
                nonlocal writes
                with writes_lock:
                    writes += 1
                save_state(state)

            manager._save_state = counted_save

            def submit(index: int) -> dict:
                lease = leases[index]
                assert lease is not None
                return manager.complete_and_lease_next(
                    lease["lease_id"],
                    f"annotator_{index}",
                    f"client_{index}",
                    "left",
                    f"record_{index}",
                )

            with ThreadPoolExecutor(max_workers=6) as executor:
                transitions = list(executor.map(submit, range(6)))

            next_leases = [transition["lease"] for transition in transitions]
            self.assertTrue(all(lease is not None for lease in next_leases))
            self.assertEqual(writes, 6)
            self.assertEqual(len({lease["pair_id"] for lease in next_leases}), 6)
            self.assertEqual(manager.stats()["annotations_done"], 6)
            self.assertEqual(manager.stats()["active_leases"], 6)

    def test_atomic_state_failure_does_not_mutate_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.npz"
            right = root / "right.npz"
            np.savez(left, joint_pos=np.zeros((10, 2)))
            np.savez(right, joint_pos=np.zeros((10, 2)))
            pairs = root / "pairs.jsonl"
            pairs.write_text(json.dumps(_pair(left, right)) + "\n", encoding="utf-8")
            manager = DynamicTaskManager(
                pairs, root / "state.json", annotations_per_pair=1
            )
            lease = manager.lease_next("alice", "alice_client")
            assert lease is not None
            before = json.dumps(manager.state, sort_keys=True)

            def fail_save(state=None):
                raise RuntimeError("simulated state write failure")

            manager._save_state = fail_save
            with expect_error(RuntimeError, "simulated state write failure"):
                manager.complete_and_lease_next(
                    lease["lease_id"],
                    "alice",
                    "alice_client",
                    "left",
                    "record_failure",
                )
            self.assertEqual(json.dumps(manager.state, sort_keys=True), before)

    def test_ui_uses_one_lease_per_page(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        html = (package_root / "web_labeler" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        app_source = (package_root / "web_labeler" / "app.py").read_text(
            encoding="utf-8"
        )
        for removed in (
            "LEFT · RIGHT",
            'class="video-label"',
            "/api/lease_batch",
            "TASK_QUEUE_TARGET",
            "taskQueue",
        ):
            self.assertNotIn(removed, html)
        self.assertNotIn("/api/lease_batch", app_source)
        self.assertIn("const clientId = crypto.randomUUID();", html)
        self.assertIn("/api/lease/heartbeat", html)
        self.assertIn("/api/lease/release", html)
        self.assertIn('data-testid="undo"', html)
        self.assertIn("/api/undo", html)
        self.assertIn('@app.post("/api/undo")', app_source)
        self.assertIn("response.next_task", html)
        self.assertIn("complete_and_lease_next", app_source)
        self.assertIn("asyncio.to_thread(\n            _label_response", app_source)
        self.assertIn("pageParams.get('prefetch') || 4", html)
        self.assertIn("limit: int = 4", app_source)
        self.assertIn(">Cannot compare <", html)
        self.assertNotIn(">Bad Traj <", html)


if __name__ == "__main__":
    unittest.main()
