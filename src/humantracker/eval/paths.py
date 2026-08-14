"""Path resolution shared by the evaluation runner and its backends.

Every path on the evaluation command line may be given relative to the repository root,
so a command copied out of the README works from any working directory. Nothing here
falls back to a default location: a path that does not exist raises.
"""

from __future__ import annotations

from pathlib import Path

# <repo>/src/humantracker/eval/paths.py
ROOT = Path(__file__).resolve().parents[3]


def repo_path(value: str) -> Path:
    """Resolve ``value`` against the repository root unless it is already absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def required_file(value: str) -> str:
    """Resolve ``value`` and require it to be a file.

    Returns ``str`` because the consumers are third-party loaders (ONNX, MuJoCo, Torch)
    that take path strings.
    """
    path = repo_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return str(path)


def required_dir(value: str) -> Path:
    """Resolve ``value`` and require it to be a directory."""
    path = repo_path(value)
    if not path.is_dir():
        raise NotADirectoryError(f"Required directory not found: {path}")
    return path
