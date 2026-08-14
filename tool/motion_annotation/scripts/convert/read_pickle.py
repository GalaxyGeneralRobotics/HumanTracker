#!/usr/bin/env python3
"""Inspect one pickle motion or every pickle motion in a directory."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_pickle(path: str | Path) -> Any:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        return pickle.load(stream)


def summarize_value(name: str, value: Any) -> str:
    if not isinstance(value, np.ndarray):
        return f"{name}: {type(value).__name__}"
    summary = f"{name}: array shape={value.shape} dtype={value.dtype}"
    if value.size and np.issubdtype(value.dtype, np.number):
        summary += (
            f" min={float(value.min()):.6g}"
            f" max={float(value.max()):.6g}"
            f" mean={float(value.mean()):.6g}"
        )
    return summary


def summarize_motion(path: str | Path) -> list[str]:
    source = Path(path).resolve()
    data = load_pickle(source)
    lines = [f"file={source}", f"type={type(data).__name__}"]
    if isinstance(data, dict):
        lines.append(f"keys={list(data)}")
        lines.extend(summarize_value(str(key), value) for key, value in data.items())
    elif isinstance(data, np.ndarray):
        lines.append(summarize_value("value", data))
    else:
        lines.append(repr(data))
    return lines


def resolve_inputs(path: str | Path) -> list[Path]:
    source = Path(path).resolve()
    if source.is_file():
        if source.suffix.lower() != ".pkl":
            raise ValueError(f"expected .pkl file, got {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = sorted(source.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no .pkl files in {source}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="pickle file or directory")
    args = parser.parse_args()
    paths = resolve_inputs(args.path)
    for index, path in enumerate(paths):
        if index:
            print()
        print("\n".join(summarize_motion(path)))


if __name__ == "__main__":
    main()
