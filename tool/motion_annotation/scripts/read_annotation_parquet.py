#!/usr/bin/env python3
"""Validate and inspect annotation inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from tool.rm_pipeline.annotations import read_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--head", type=int, default=3)
    args = parser.parse_args()
    if args.head < 0:
        raise ValueError("--head must be non-negative")

    records = read_annotations(args.inputs)
    summary = {
        "records": len(records),
        "pairs": len({record["pair_id"] for record in records}),
        "annotators": len({record["meta"]["annotator_id"] for record in records}),
        "choices": dict(Counter(record["preference"]["choice_type"] for record in records)),
        "categories": dict(Counter(record["category"] for record in records)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for record in records[: args.head]:
        print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
