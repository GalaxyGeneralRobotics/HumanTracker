"""Combine the per-shard results of one ``--shard_count`` run into one results file.

Every shard must come from the same command: the merger rejects shards whose
termination rule or reference-noise record differ, duplicate or out-of-range
trajectory ids, rows that do not match the test manifest entry of their id, and any
trajectory left uncovered. Summaries are then recomputed from all rows by the
tracker's own backend, exactly as an unsharded run would write them.

Shard ids are indices into the manifest, so shard runs must not filter it
(``--categories``, ``--skip_flipped``); ``--max_trajs N`` runs merge with
``--expected_count N``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from humantracker.eval.backends import BACKEND_NAMES, load_backend
from humantracker.eval.runner import print_summaries, save_json

HEADER_KEYS = ("termination_metric", "ref_noise")


def merge_shards(manifest: List[Dict], shards: Dict[Path, Dict]) -> tuple[Dict, List[Dict]]:
    """Return the shards' common header and their rows in manifest order."""
    header = None
    by_id: Dict[int, Dict] = {}
    for path, payload in shards.items():
        shard_header = {key: payload[key] for key in HEADER_KEYS if key in payload}
        if header is None:
            header = shard_header
        elif shard_header != header:
            raise ValueError(f"{path} was produced with different run settings: {shard_header}")
        for metric in payload["per_trajectory"]:
            index = metric["traj_id"]
            if index in by_id:
                raise ValueError(f"Duplicate trajectory {index} in {path}")
            if not 0 <= index < len(manifest):
                raise ValueError(f"Trajectory {index} in {path} is outside the manifest")
            entry = manifest[index]
            if metric["file_name"] != Path(entry["path"]).name or metric["category"] != entry["category"]:
                raise ValueError(f"Trajectory {index} in {path} does not match the manifest")
            by_id[index] = metric

    missing = sorted(set(range(len(manifest))) - by_id.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} trajectories; first ids: {missing[:20]}")
    return header, [by_id[index] for index in range(len(manifest))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=BACKEND_NAMES, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_count", type=int, default=0,
                        help="number of leading manifest entries the run covered "
                             "(its --max_trajs); defaults to the whole manifest")
    args = parser.parse_args()

    manifest = json.loads(args.test_json.read_text())
    if args.expected_count:
        if not 0 < args.expected_count <= len(manifest):
            raise ValueError("--expected_count is outside the test manifest")
        manifest = manifest[:args.expected_count]
    shards = {path: json.loads(path.read_text()) for path in args.inputs}
    header, metrics = merge_shards(manifest, shards)

    backend = load_backend(args.tracker)
    print_summaries(backend, metrics)
    path = save_json(args.output_json, header, backend, metrics)
    errors = sum(bool(metric.get("error")) for metric in metrics)
    print(f"Merged {len(metrics)} trajectories into {path}; {errors} evaluation exceptions")


if __name__ == "__main__":
    main()
