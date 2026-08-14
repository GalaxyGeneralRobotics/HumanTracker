from __future__ import annotations

import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from tool.rm_pipeline.build_motion_pairs import (
    TRACKERS,
    load_train_index,
    tracker_pair_plan,
    uniform_indices,
)


class UniformSamplingTest(unittest.TestCase):
    def test_midpoint_indices_are_unique_and_uniform(self) -> None:
        indices = uniform_indices(42190, 6000)
        gaps = [right - left for left, right in zip(indices, indices[1:])]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual((min(gaps), max(gaps)), (7, 8))

    def test_pair_plan_is_exactly_balanced(self) -> None:
        plan = tracker_pair_plan(6000)
        combinations = Counter(pair for pair, _ in plan)
        trackers = Counter(tracker for pair, _ in plan for tracker in pair)
        orientations: dict[tuple[str, str], Counter[bool]] = defaultdict(Counter)
        for pair, reverse in plan:
            orientations[pair][reverse] += 1
        self.assertEqual(set(combinations.values()), {1000})
        self.assertEqual(trackers, Counter({tracker: 3000 for tracker in TRACKERS}))
        self.assertTrue(all(counts == Counter({False: 500, True: 500}) for counts in orientations.values()))

    def test_train_index_excludes_flipped_rows(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            category = root / "Daily"
            category.mkdir()
            (category / "walk.npz").touch()
            (category / "walk_flipped.npz").touch()
            train_json = root / "train.json"
            train_json.write_text(json.dumps([
                {"path": "Daily/walk.npz", "category": "Daily", "frames": 500},
                {"path": "Daily/walk_flipped.npz", "category": "Daily", "frames": 500},
            ]))

            indexed = load_train_index(train_json, root)

            self.assertEqual(len(indexed), 1)
            self.assertEqual({row["motion_idx"] for row in indexed.values()}, {0})


if __name__ == "__main__":
    unittest.main()
