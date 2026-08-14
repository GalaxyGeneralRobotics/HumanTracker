#!/usr/bin/env python3
"""Train a fixed-length trajectory reward model from preference annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from numpy.lib.format import open_memmap
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_cosine_schedule_with_warmup


from humantracker.reward_model.datasets.parquet_reader import load_annotations
from humantracker.reward_model.features import (
    FEATURE_HEADS,
    FEATURE_KEYS,
    SEQ_LEN,
    flip_features,
)
from humantracker.reward_model.datasets.split_annotations import select_test_groups
from humantracker.reward_model.models.loss import SoftTargetBradleyTerryLoss
from humantracker.reward_model.models.reward_model import RewardModel


# The reported checkpoint does not use future reference residuals (Sec. 3.3 /
# appendix Table 6): drop the three next-frame reference blocks from the full
# feature schema and train on the rest (539 of the 850 dimensions).
FUTURE_REFERENCE_KEYS = frozenset(
    {"next_ref2acu_gv_vel", "next_ref2acu_kpt_pose", "next_ref2acu_kpt_cvel"}
)
TRAIN_FEATURE_KEYS = tuple(key for key in FEATURE_KEYS if key not in FUTURE_REFERENCE_KEYS)
TRAIN_FEATURE_DIM = sum(FEATURE_HEADS[key] for key in TRAIN_FEATURE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name")
    parser.add_argument("--wandb_project", default="humantracker-reward-model")
    parser.add_argument("--wandb_group")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=("mean", "max", "cls"), default="mean")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--schedule_epochs", type=int)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tie_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--downsample_rate", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--final_fit", action="store_true")
    return parser.parse_args()


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "trajectory_a": cache_dir / "trajectory_a.npy",
        "trajectory_b": cache_dir / "trajectory_b.npy",
        "valid_frames": cache_dir / "valid_frames.npy",
        "labels": cache_dir / "is_preference.npy",
        "splits": cache_dir / "split.npy",
        "is_augmented": cache_dir / "is_augmented.npy",
        "metadata": cache_dir / "metadata.json",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clip_length(annotation: dict) -> int:
    lengths = {
        int(candidate["end_frame"]) - int(candidate["start_frame"])
        for candidate in annotation["candidates"]
    }
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError(f"{annotation['record_id']}: invalid candidate lengths {lengths}")
    return next(iter(lengths))


def trajectory_features(candidate: dict) -> np.ndarray:
    path = Path(candidate["traj_path"])
    start = int(candidate["start_frame"])
    end = int(candidate["end_frame"])
    valid_frames = end - start
    if not 1 <= valid_frames <= SEQ_LEN:
        raise ValueError(f"{path}: expected 1..{SEQ_LEN} frames, got {valid_frames}")
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as trajectory:
        missing = set(FEATURE_KEYS) - set(trajectory.files)
        if missing:
            raise KeyError(f"{path}: missing features {sorted(missing)}")
        arrays = [np.asarray(trajectory[key][start:end], dtype=np.float32) for key in TRAIN_FEATURE_KEYS]
    if any(array.ndim != 2 or array.shape[0] != valid_frames for array in arrays):
        shapes = {key: value.shape for key, value in zip(TRAIN_FEATURE_KEYS, arrays)}
        raise ValueError(f"{path}: expected {valid_frames} frames, got {shapes}")
    features = np.concatenate(arrays, axis=-1)
    if features.shape != (valid_frames, TRAIN_FEATURE_DIM):
        raise ValueError(f"{path}: expected {(valid_frames, TRAIN_FEATURE_DIM)}, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite features")
    padded = np.zeros((SEQ_LEN, TRAIN_FEATURE_DIM), dtype=np.float32)
    padded[:valid_frames] = features
    return padded


def annotation_pair(
    annotation: dict,
) -> tuple[np.ndarray, np.ndarray, np.uint8, np.uint16]:
    candidates = {int(item["candidate_idx"]): item for item in annotation["candidates"]}
    if set(candidates) != {0, 1}:
        raise ValueError(f"{annotation['record_id']}: candidate_idx must be 0 and 1")
    choice = annotation["preference"]["choice_type"]
    valid_frames = np.uint16(clip_length(annotation))
    if choice == "preference":
        preferred = int(annotation["preference"]["preferred_candidate_idx"])
        if preferred not in candidates:
            raise ValueError(f"{annotation['record_id']}: invalid preferred candidate")
        other = 1 - preferred
        return (
            trajectory_features(candidates[preferred]),
            trajectory_features(candidates[other]),
            np.uint8(1),
            valid_frames,
        )
    if choice == "similar":
        return (
            trajectory_features(candidates[0]),
            trajectory_features(candidates[1]),
            np.uint8(0),
            valid_frames,
        )
    raise ValueError(f"{annotation['record_id']}: unsupported choice {choice}")


def bilateral_flip(features: np.ndarray) -> np.ndarray:
    heads = {key: FEATURE_HEADS[key] for key in TRAIN_FEATURE_KEYS}
    return flip_features(features, TRAIN_FEATURE_KEYS, heads)


def load_manifest(path: Path, split: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("split") != split:
        raise ValueError(f"{path}: expected split={split}")
    return manifest


def validate_existing_cache(paths: dict[str, Path], expected: dict) -> dict:
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return {}
    if len(existing) != len(paths):
        raise RuntimeError(f"incomplete cache: {[str(path) for path in existing]}")
    metadata = json.loads(paths["metadata"].read_text())
    for key, value in expected.items():
        if key != "feature_keys" and metadata.get(key) != value:
            raise ValueError(f"stale cache metadata {key}: {metadata.get(key)} != {value}")
        if key == "feature_keys" and metadata.get(key) != value:
            raise ValueError(f"stale cache feature_keys: {metadata.get(key)} != {value}")
    feature_dim = metadata.get("feature_dim", TRAIN_FEATURE_DIM)
    shape = (metadata["num_samples"], SEQ_LEN, feature_dim)
    if np.load(paths["trajectory_a"], mmap_mode="r").shape != shape:
        raise ValueError("trajectory_a cache shape mismatch")
    if np.load(paths["trajectory_b"], mmap_mode="r").shape != shape:
        raise ValueError("trajectory_b cache shape mismatch")
    valid_frames = np.load(paths["valid_frames"], mmap_mode="r")
    if valid_frames.shape != (metadata["num_samples"],):
        raise ValueError("valid_frames cache shape mismatch")
    if valid_frames.dtype != np.uint16:
        raise ValueError("valid_frames cache dtype mismatch")
    if valid_frames.min() < 1 or valid_frames.max() > SEQ_LEN:
        raise ValueError("valid_frames cache values are outside 1..250")
    if np.load(paths["labels"], mmap_mode="r").shape != (metadata["num_samples"],):
        raise ValueError("label cache shape mismatch")
    if np.load(paths["splits"], mmap_mode="r").shape != (metadata["num_samples"],):
        raise ValueError("split cache shape mismatch")
    is_augmented = np.load(paths["is_augmented"], mmap_mode="r")
    if is_augmented.shape != (metadata["num_samples"],):
        raise ValueError("augmentation cache shape mismatch")
    if is_augmented.dtype != np.bool_:
        raise ValueError("augmentation cache dtype mismatch")
    return metadata


def prepare_cache(data_dir: Path, cache_dir: Path, eval_ratio: float, seed: int) -> dict:
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"
    train_manifest = load_manifest(train_path, "train")
    test_manifest = load_manifest(test_path, "test")
    expected = {
        "source_dir": str(data_dir.resolve()),
        "train_manifest_sha256": file_sha256(train_path),
        "test_manifest_sha256": file_sha256(test_path),
        "eval_ratio": eval_ratio,
        "seed": seed,
        "seq_len": SEQ_LEN,
        "feature_dim": TRAIN_FEATURE_DIM,
        "feature_keys": TRAIN_FEATURE_KEYS,
        "padding": "right_zero",
        "use_padding_mask": True,
        "augmentation": "original_and_bilateral_flip",
        "augmentation_multiplier": 2,
        "augmented_splits": ["train", "eval", "test"],
    }
    paths = cache_paths(cache_dir)
    existing = validate_existing_cache(paths, expected)
    if existing:
        print(f"validated dataset cache: {cache_dir}", flush=True)
        return existing

    annotations = load_annotations(data_dir)
    record_ids = {item["record_id"] for item in annotations}
    train_ids = set(train_manifest["record_ids"])
    test_ids = set(test_manifest["record_ids"])
    if train_ids & test_ids or train_ids | test_ids != record_ids:
        raise ValueError("train/test manifests do not form an exact disjoint partition")
    eligible = [
        item
        for item in annotations
        if item["preference"]["choice_type"] in {"preference", "similar"}
    ]
    lengths = [clip_length(item) for item in eligible]
    if any(length > SEQ_LEN for length in lengths):
        raise ValueError(f"clip length exceeds {SEQ_LEN}")
    eligible_train = [item for item in eligible if item["record_id"] in train_ids]
    eval_motions = select_test_groups(eligible_train, eval_ratio, seed)
    base_split_counts = Counter()
    cached_split_counts = Counter()
    label_counts = Counter()
    assignments = []
    for annotation in eligible:
        if annotation["record_id"] in test_ids:
            split = 2
        elif annotation["motion_id"] in eval_motions:
            split = 1
        else:
            split = 0
        assignments.append(split)
        base_split_counts[split] += 1
        cached_split_counts[split] += 2
        label_counts[annotation["preference"]["choice_type"]] += 1

    sample_count = 2 * len(eligible)

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = {name: path.with_name(path.name + ".tmp") for name, path in paths.items()}
    for path in temporary.values():
        if path.exists():
            raise RuntimeError(f"unfinished cache file exists: {path}")
    shape = (sample_count, SEQ_LEN, TRAIN_FEATURE_DIM)
    trajectory_a = open_memmap(
        temporary["trajectory_a"], mode="w+", dtype=np.float32, shape=shape
    )
    trajectory_b = open_memmap(
        temporary["trajectory_b"], mode="w+", dtype=np.float32, shape=shape
    )
    valid_frames = open_memmap(
        temporary["valid_frames"],
        mode="w+",
        dtype=np.uint16,
        shape=(sample_count,),
    )
    labels = open_memmap(temporary["labels"], mode="w+", dtype=np.uint8, shape=(sample_count,))
    splits = open_memmap(temporary["splits"], mode="w+", dtype=np.uint8, shape=(sample_count,))
    is_augmented = open_memmap(
        temporary["is_augmented"], mode="w+", dtype=np.bool_, shape=(sample_count,)
    )
    sample_meta = []
    sample_index = 0
    for base_index, (annotation, split) in enumerate(zip(eligible, assignments)):
        original_a, original_b, label, length = annotation_pair(annotation)
        augmented_pairs = [
            (original_a, original_b, False),
            (bilateral_flip(original_a), bilateral_flip(original_b), True),
        ]
        for sample_a, sample_b, augmented in augmented_pairs:
            trajectory_a[sample_index] = sample_a
            trajectory_b[sample_index] = sample_b
            labels[sample_index] = label
            valid_frames[sample_index] = length
            splits[sample_index] = split
            is_augmented[sample_index] = augmented
            sample_meta.append(
                {
                    "record_id": annotation["record_id"],
                    "motion_id": annotation["motion_id"],
                    "category": annotation["category"],
                    "tracker_pair": annotation["tracker_pair_key"],
                    "valid_frames": int(length),
                    "augmentation": "bilateral_flip" if augmented else "none",
                }
            )
            sample_index += 1
        if (base_index + 1) % 100 == 0 or base_index + 1 == len(eligible):
            print(f"cached {base_index + 1}/{len(eligible)} base pairs", flush=True)
    if sample_index != sample_count:
        raise RuntimeError(f"cached {sample_index} samples, expected {sample_count}")
    for array in (trajectory_a, trajectory_b, valid_frames, labels, splits, is_augmented):
        array.flush()
    del trajectory_a, trajectory_b, valid_frames, labels, splits, is_augmented
    for name in (
        "trajectory_a",
        "trajectory_b",
        "valid_frames",
        "labels",
        "splits",
        "is_augmented",
    ):
        temporary[name].replace(paths[name])
    metadata = expected | {
        "num_base_pairs": len(eligible),
        "num_samples": sample_count,
        "num_augmented_samples": sample_count - len(eligible),
        "num_base_preferences": label_counts["preference"],
        "num_base_similar": label_counts["similar"],
        "num_base_full_length": sum(length == SEQ_LEN for length in lengths),
        "num_base_padded": sum(length < SEQ_LEN for length in lengths),
        "base_split_counts": {
            "train": base_split_counts[0],
            "eval": base_split_counts[1],
            "test": base_split_counts[2],
        },
        "cached_split_counts": {
            "train": cached_split_counts[0],
            "eval": cached_split_counts[1],
            "test": cached_split_counts[2],
        },
        "sample_meta": sample_meta,
    }
    temporary["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary["metadata"].replace(paths["metadata"])
    print(json.dumps(metadata["cached_split_counts"], indent=2), flush=True)
    return metadata


class PreferenceDataset(Dataset):
    def __init__(self, cache_dir: Path, downsample_rate: int):
        if downsample_rate < 1:
            raise ValueError("downsample_rate must be positive")
        paths = cache_paths(cache_dir)
        self.trajectory_a = np.load(paths["trajectory_a"], mmap_mode="r")
        self.trajectory_b = np.load(paths["trajectory_b"], mmap_mode="r")
        self.valid_frames = np.load(paths["valid_frames"], mmap_mode="r")
        self.labels = np.load(paths["labels"], mmap_mode="r")
        self.splits = np.load(paths["splits"], mmap_mode="r")
        self.is_augmented = np.load(paths["is_augmented"], mmap_mode="r")
        self.downsample_rate = downsample_rate

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_slice = slice(None, None, self.downsample_rate)
        trajectory_a = np.array(self.trajectory_a[index, frame_slice], copy=True)
        trajectory_b = np.array(self.trajectory_b[index, frame_slice], copy=True)
        frame_indices = np.arange(0, SEQ_LEN, self.downsample_rate)
        valid_mask = frame_indices < int(self.valid_frames[index])
        return (
            torch.from_numpy(trajectory_a),
            torch.from_numpy(trajectory_b),
            torch.tensor(self.labels[index], dtype=torch.bool),
            torch.from_numpy(valid_mask),
        )


def data_indices(dataset: PreferenceDataset, final_fit: bool) -> dict[str, list[int]]:
    indices = {
        "train": np.flatnonzero(dataset.splits == 0).tolist(),
        "eval": np.flatnonzero(dataset.splits == 1).tolist(),
        "test": np.flatnonzero(dataset.splits == 2).tolist(),
    }
    indices["fit"] = (
        np.flatnonzero(dataset.splits < 2).tolist() if final_fit else indices["train"]
    )
    return indices



def batch_statistics(loss: torch.Tensor, score_a: torch.Tensor, score_b: torch.Tensor, is_preference: torch.Tensor) -> list[float]:
    gap = (score_a.detach() - score_b.detach()).flatten()
    probability = gap.sigmoid()
    preference = is_preference.bool()
    tie = ~preference
    targets = torch.where(preference, torch.ones_like(gap), torch.full_like(gap, 0.5))
    nll = F.binary_cross_entropy_with_logits(gap, targets, reduction="none")
    return [
        loss.item() * len(gap),
        (gap[preference] > 0).sum().item(),
        gap[preference].sum().item(),
        probability[preference].sum().item(),
        nll[preference].sum().item(),
        preference.sum().item(),
        gap[tie].abs().sum().item(),
        (probability[tie] - 0.5).abs().sum().item(),
        nll[tie].sum().item(),
        tie.sum().item(),
        ((probability - targets) ** 2).sum().item(),
        len(gap),
    ]


def summarize(values: list[float]) -> dict[str, float]:
    loss, correct, gap, probability, pref_nll, pref_count, tie_gap, tie_error, tie_nll, tie_count, brier, count = values
    metrics = {"loss": loss / count, "brier_score": brier / count}
    if pref_count:
        metrics |= {
            "preference_accuracy": correct / pref_count,
            "preference_score_gap": gap / pref_count,
            "preference_probability": probability / pref_count,
            "preference_nll": pref_nll / pref_count,
        }
    if tie_count:
        metrics |= {
            "tie_score_gap": tie_gap / tie_count,
            "tie_probability_error": tie_error / tie_count,
            "tie_nll": tie_nll / tie_count,
        }
    return metrics


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device, bf16: bool) -> dict[str, float]:
    model.eval()
    totals = [0.0] * 12
    selection_loss_sum = 0.0
    with torch.no_grad():
        for trajectory_a, trajectory_b, is_preference, valid_mask in loader:
            trajectory_a = trajectory_a.to(device, non_blocking=True)
            trajectory_b = trajectory_b.to(device, non_blocking=True)
            is_preference = is_preference.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                score_a, score_b = model(trajectory_a, trajectory_b, valid_mask)
                loss = criterion(score_a, score_b, is_preference)
                gap = (score_a - score_b).flatten()
                targets = torch.where(is_preference, torch.ones_like(gap), torch.full_like(gap, 0.5))
                selection = F.binary_cross_entropy_with_logits(gap, targets)
            values = batch_statistics(loss, score_a, score_b, is_preference)
            totals = [total + value for total, value in zip(totals, values)]
            selection_loss_sum += selection.item() * values[-1]
    metrics = summarize(totals)
    metrics["selection_loss"] = selection_loss_sum / totals[-1]
    return metrics


def checkpoint_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, args: argparse.Namespace, metadata: dict, epoch: int, step: int, metrics: dict) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args) | {
            "padding": "right_zero",
            "use_padding_mask": True,
            "augmentation": "original_and_bilateral_flip",
            "augmentation_multiplier": 2,
        },
        "dataset": {key: value for key, value in metadata.items() if key != "sample_meta"},
        "epoch": epoch,
        "global_step": step,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    metadata = prepare_cache(args.data_dir.resolve(), args.cache_dir.resolve(), args.eval_ratio, args.seed)
    if args.prepare_data:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", 0)
    dataset = PreferenceDataset(args.cache_dir, args.downsample_rate)
    split_indices = data_indices(dataset, args.final_fit)
    if any(not indices for indices in split_indices.values()):
        raise ValueError(f"empty data split: { {key: len(value) for key, value in split_indices.items()} }")
    train_dataset = Subset(dataset, split_indices["fit"])
    eval_dataset = Subset(dataset, split_indices["eval"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = None if args.final_fit else DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = RewardModel(
        input_dim=TRAIN_FEATURE_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=math.ceil(SEQ_LEN / args.downsample_rate) + 1,
        pooling=args.pooling,
    ).to(device)
    criterion = SoftTargetBradleyTerryLoss(args.tie_weight, args.temperature)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule_epochs = args.schedule_epochs or args.max_epochs
    if schedule_epochs < args.max_epochs:
        raise ValueError("schedule_epochs must not be shorter than max_epochs")
    total_steps = schedule_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    run_name = args.run_name or datetime.now().strftime("rm_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "train_samples": len(train_dataset),
        "eval_samples": 0 if args.final_fit else len(eval_dataset),
        "held_out_test_samples": len(split_indices["test"]),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "padding": "right_zero",
        "use_padding_mask": True,
        "augmentation": "original_and_bilateral_flip",
        "augmentation_multiplier": 2,
        "validation_augmentation": "original_and_bilateral_flip",
        "test_augmentation": "original_and_bilateral_flip",
    }
    wandb.init(project=args.wandb_project, name=run_name, group=args.wandb_group, config=config)
    print(json.dumps(config, indent=2, default=str), flush=True)

    best_key = (-math.inf, -math.inf)
    global_step = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        interval = [0.0] * 12
        interval_start = time.perf_counter()
        for trajectory_a, trajectory_b, is_preference, valid_mask in train_loader:
            trajectory_a = trajectory_a.to(device, non_blocking=True)
            trajectory_b = trajectory_b.to(device, non_blocking=True)
            is_preference = is_preference.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                score_a, score_b = model(trajectory_a, trajectory_b, valid_mask)
                loss = criterion(score_a, score_b, is_preference)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            values = batch_statistics(loss, score_a, score_b, is_preference)
            interval = [total + value for total, value in zip(interval, values)]
            global_step += 1
            if global_step % args.logging_steps == 0:
                metrics = summarize(interval)
                elapsed = time.perf_counter() - interval_start
                log = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm),
                    "train/samples_per_second": interval[-1] / elapsed,
                } | {f"train/{key}": value for key, value in metrics.items()}
                wandb.log(log)
                print(json.dumps(log), flush=True)
                interval = [0.0] * 12
                interval_start = time.perf_counter()

        if args.final_fit:
            state = checkpoint_state(model, optimizer, scheduler, args, metadata, epoch, global_step, {})
            torch.save(state, run_dir / "last.pt")
            torch.save(state, run_dir / "best.pt")
            wandb.log({"global_step": global_step, "epoch": epoch})
            continue
        eval_metrics = evaluate(model, eval_loader, criterion, device, args.bf16)
        key = (eval_metrics["preference_accuracy"], -eval_metrics["selection_loss"])
        is_best = key > best_key
        if is_best:
            best_key = key
        log = {
            "global_step": global_step,
            "epoch": epoch,
            "eval/best_preference_accuracy": best_key[0],
        } | {f"eval/{name}": value for name, value in eval_metrics.items()}
        wandb.log(log)
        print(json.dumps(log), flush=True)
        state = checkpoint_state(model, optimizer, scheduler, args, metadata, epoch, global_step, eval_metrics)
        torch.save(state, run_dir / "last.pt")
        if is_best:
            torch.save(state, run_dir / "best.pt")
            wandb.run.summary["best_preference_accuracy"] = best_key[0]
            wandb.run.summary["best_selection_loss"] = eval_metrics["selection_loss"]
            wandb.run.summary["best_epoch"] = epoch
    wandb.finish()


if __name__ == "__main__":
    main()
