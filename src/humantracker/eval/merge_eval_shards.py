"""Validate and combine trajectory shards using the normal tracker summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humantracker.eval.backends import BACKEND_NAMES, load_backend
from humantracker.eval.runner import print_summaries, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=BACKEND_NAMES, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--expected_count", type=int, default=0,
                        help="number of leading test entries; defaults to the full manifest")
    args = parser.parse_args()

    manifest = json.loads(args.test_json.read_text())
    if args.expected_count:
        if not 0 < args.expected_count <= len(manifest):
            raise ValueError("--expected_count is outside the test manifest")
        manifest = manifest[:args.expected_count]
    by_id = {}
    termination = None
    for path in args.inputs:
        payload = json.loads(path.read_text())
        rule = payload["termination_metric"]
        if termination is not None and rule != termination:
            raise ValueError(f"Termination mismatch in {path}: {rule} != {termination}")
        termination = rule
        for metric in payload["per_trajectory"]:
            index = metric["traj_id"]
            if index in by_id:
                raise ValueError(f"Duplicate trajectory {index} in {path}")
            if not 0 <= index < len(manifest):
                raise ValueError(f"Trajectory {index} outside the expected manifest")
            entry = manifest[index]
            if metric["file_name"] != Path(entry["path"]).name or metric["category"] != entry["category"]:
                raise ValueError(f"Trajectory {index} does not match the test manifest")
            by_id[index] = metric

    missing = sorted(set(range(len(manifest))) - by_id.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} trajectories; first IDs: {missing[:20]}")
    metrics = [by_id[index] for index in range(len(manifest))]
    backend = load_backend(args.tracker)
    args.termination_metric = termination
    args.timestamp_output = False
    print_summaries(backend, metrics)
    path = save_json(args, backend, metrics)
    errors = sum(bool(metric.get("error")) for metric in metrics)
    print(f"Merged {len(metrics)} trajectories into {path}; {errors} evaluation exceptions")


if __name__ == "__main__":
    main()
