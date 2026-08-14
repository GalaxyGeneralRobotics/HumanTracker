#!/usr/bin/env python3
"""Evaluate a reward-model checkpoint on the independent held-out cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PosixPath

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from humantracker.reward_model.features import SEQ_LEN
from humantracker.reward_model.models.reward_model import RewardModel
from humantracker.reward_model.train.trainer import PreferenceDataset, cache_paths


MIN_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedTestDataset(Dataset):
    def __init__(self, dataset: PreferenceDataset):
        self.dataset = dataset
        self.indices = np.flatnonzero(
            (dataset.splits == 2)
            & ~dataset.is_augmented
            & (dataset.valid_frames >= MIN_FRAMES)
        ).tolist()
        if not self.indices:
            raise ValueError("test split is empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        trajectory_a, trajectory_b, is_preference, valid_mask = self.dataset[source_index]
        return trajectory_a, trajectory_b, is_preference, valid_mask, source_index


def summarize(rows: list[dict]) -> dict:
    preferences = [row for row in rows if row["is_preference"]]
    ties = [row for row in rows if not row["is_preference"]]
    if not preferences:
        raise ValueError("group has no strict preferences")
    return {
        "pairs": len(rows),
        "strict_preferences": len(preferences),
        "correct": sum(row["correct"] for row in preferences),
        "preference_accuracy": sum(row["correct"] for row in preferences) / len(preferences),
        "mean_preference_score_gap": float(np.mean([row["score_gap"] for row in preferences])),
        "similar": len(ties),
        "mean_similar_abs_score_gap": (
            float(np.mean([abs(row["score_gap"]) for row in ties])) if ties else math.nan
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    cache_dir = args.cache_dir.resolve()
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["args"]
    if config.get("padding") != "right_zero" or config.get("use_padding_mask") is not True:
        raise ValueError("checkpoint must use right-zero padding with a padding mask")
    if config.get("augmentation") != "original_and_bilateral_flip":
        raise ValueError("checkpoint must use original-plus-bilateral-flip augmentation")
    downsample_rate = int(config["downsample_rate"])
    dataset = PreferenceDataset(cache_dir, downsample_rate)
    test_dataset = IndexedTestDataset(dataset)
    metadata = json.loads(cache_paths(cache_dir)["metadata"].read_text())
    if metadata.get("padding") != "right_zero" or metadata.get("use_padding_mask") is not True:
        raise ValueError("cache must use right-zero padding with a padding mask")
    if metadata.get("augmentation") != "original_and_bilateral_flip":
        raise ValueError("cache must use original-plus-bilateral-flip augmentation")
    sample_meta = metadata["sample_meta"]
    if len(sample_meta) != len(dataset):
        raise ValueError("sample metadata length mismatch")

    model = RewardModel(
        input_dim=int(metadata["feature_dim"]),
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        num_layers=int(config["num_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_seq_len=math.ceil(SEQ_LEN / downsample_rate) + 1,
        pooling=str(config["pooling"]),
    ).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    rows = []
    with torch.no_grad():
        for trajectory_a, trajectory_b, is_preference, valid_mask, indices in loader:
            trajectory_a = trajectory_a.cuda(non_blocking=True)
            trajectory_b = trajectory_b.cuda(non_blocking=True)
            valid_mask = valid_mask.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(config["bf16"])):
                score_a, score_b = model(trajectory_a, trajectory_b, valid_mask)
            gaps = (score_a - score_b).flatten().float().cpu().numpy()
            for source_index, preference, gap in zip(indices.tolist(), is_preference.tolist(), gaps.tolist()):
                meta = sample_meta[source_index]
                rows.append(
                    {
                        "record_id": meta["record_id"],
                        "category": meta["category"],
                        "tracker_pair": meta["tracker_pair"],
                        "is_preference": bool(preference),
                        "score_gap": float(gap),
                        "correct": bool(gap > 0),
                    }
                )

    by_category: dict[str, list[dict]] = defaultdict(list)
    by_tracker_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_tracker_pair[row["tracker_pair"]].append(row)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selection_split": "test",
        "test_used_for_hyperparameter_selection": False,
        "padding": "right_zero",
        "use_padding_mask": True,
        "training_augmentation": "original_and_bilateral_flip",
        "test_augmentation": "none",
        "minimum_frames": MIN_FRAMES,
        "fixed_input_frames": SEQ_LEN,
        "downsample_rate": downsample_rate,
        "overall": summarize(rows),
        "by_category": {
            key: summarize(value) for key, value in sorted(by_category.items())
        },
        "by_tracker_pair": {
            key: summarize(value) for key, value in sorted(by_tracker_pair.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
