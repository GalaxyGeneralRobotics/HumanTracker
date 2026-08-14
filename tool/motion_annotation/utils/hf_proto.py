"""Typed in-memory records used by the annotation UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UiLabel = Literal["left", "right", "similar", "bad_traj"]


@dataclass
class HFPreference:
    winner: UiLabel
    confidence: float = 1.0


@dataclass
class HFMeta:
    record_id: str
    annotator_id: str
    timestamp: str
    tool: str
    task: str
    scene: str
    fps: int


@dataclass
class HFVideoSide:
    npz_name: str
    start_frame: int
    end_frame: int
    policy: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HFComparison:
    type: Literal["pairwise"] = "pairwise"
    play_mode: Literal["synchronized", "free"] = "synchronized"
    camera: str = "fixed_front"
    length_frames: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HFFlags:
    invalid: bool = False


@dataclass
class HFNotes:
    short_reason: str = ""
    detailed: str = ""


@dataclass
class HFRecord:
    meta: HFMeta
    video_left: HFVideoSide
    video_right: HFVideoSide
    preference: HFPreference
    flags: HFFlags
    comparison: HFComparison
    notes: HFNotes | None = None
