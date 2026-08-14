"""Strict reward-model loading and trajectory scoring for tracker evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import torch

from humantracker.reward_model.features import (
    FEATURE_DIM,
    FEATURE_HEADS,
    FEATURE_KEYS,
    SEQ_LEN,
)
from humantracker.reward_model.models.reward_model import RewardModel


def load_reward_model(ckpt_path: str, device: str) -> RewardModel:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reward-model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_sections = {"model", "dataset", "args"}
    missing_sections = required_sections - checkpoint.keys()
    if missing_sections:
        raise KeyError(f"RM checkpoint missing sections: {sorted(missing_sections)}")

    dataset = checkpoint["dataset"]
    model_args = checkpoint["args"]
    required_dataset = {"feature_dim", "feature_keys", "seq_len"}
    required_args = {
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "pooling",
        "downsample_rate",
        "use_padding_mask",
    }
    if required_dataset - dataset.keys():
        raise KeyError(f"RM dataset metadata missing: {sorted(required_dataset - dataset.keys())}")
    if required_args - model_args.keys():
        raise KeyError(f"RM model args missing: {sorted(required_args - model_args.keys())}")
    feature_keys = tuple(dataset["feature_keys"])
    unknown_keys = set(feature_keys) - FEATURE_HEADS.keys()
    if unknown_keys:
        raise ValueError(f"RM checkpoint contains unknown features: {sorted(unknown_keys)}")
    expected_keys = tuple(key for key in FEATURE_KEYS if key in feature_keys)
    if feature_keys != expected_keys:
        raise ValueError("RM feature order does not match the evaluator feature schema")
    expected_dim = sum(FEATURE_HEADS[key] for key in feature_keys)
    if int(dataset["feature_dim"]) != expected_dim:
        raise ValueError(
            f"Expected RM feature_dim={expected_dim}, got {dataset['feature_dim']}"
        )
    if int(dataset["seq_len"]) != SEQ_LEN:
        raise ValueError(f"Expected RM seq_len={SEQ_LEN}, got {dataset['seq_len']}")
    seq_len = int(dataset["seq_len"])
    downsample_rate = int(model_args["downsample_rate"])
    if downsample_rate < 1:
        raise ValueError(f"RM downsample_rate must be positive, got {downsample_rate}")
    # Required rather than defaulted: a checkpoint without the flag was written by code
    # that predates it, and guessing "unmasked" would score it with the wrong forward pass.
    use_padding_mask = bool(model_args["use_padding_mask"])
    if use_padding_mask:
        if model_args.get("padding") != "right_zero":
            raise ValueError("Masked RM checkpoint must use right-zero padding")
        if dataset.get("padding") != "right_zero" or dataset.get("use_padding_mask") is not True:
            raise ValueError("Masked RM dataset metadata is inconsistent")

    model = RewardModel(
        input_dim=int(dataset["feature_dim"]),
        d_model=int(model_args["d_model"]),
        nhead=int(model_args["nhead"]),
        num_layers=int(model_args["num_layers"]),
        dim_feedforward=int(model_args["dim_feedforward"]),
        dropout=float(model_args["dropout"]),
        max_seq_len=math.ceil(seq_len / downsample_rate) + 1,
        pooling=str(model_args["pooling"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device(device))
    model.eval()
    model.eval_seq_len = seq_len
    model.eval_downsample_rate = downsample_rate
    model.eval_use_padding_mask = use_padding_mask
    offsets = np.cumsum((0, *FEATURE_HEADS.values()))
    model.eval_feature_indices = np.concatenate(
        [
            np.arange(offsets[index], offsets[index + 1])
            for index, key in enumerate(FEATURE_KEYS)
            if key in feature_keys
        ]
    )
    print(
        f"[RM] Loaded {path} on {device} "
        f"(feature_dim={model.input_dim}, seq_len={seq_len}, "
        f"downsample_rate={downsample_rate}, use_padding_mask={use_padding_mask})",
        flush=True,
    )
    return model


def score_reward_model(
    reward_model: RewardModel,
    prompt_features: List[np.ndarray],
    trajectory_features: List[np.ndarray],
) -> dict[str, float | int | None]:
    """Score finite frames in chunks and weight every score by its represented frame count.

    A diverged rollout (MuJoCo ``Nan, Inf or huge value in CTRL``) can leave the feature
    history non-finite for part or all of the sequence. Whether that trajectory's tracking
    rule judged it "terminated" earlier is irrelevant here -- a frame is scored if and only if
    it, and everything before it in the same chunk, is finite. Chunks entirely before the first
    non-finite frame score normally; a chunk straddling it is truncated to the finite prefix and
    scored through the existing padding mask. The first non-finite frame is returned so summary
    code can exclude that trajectory rather than average fabricated state.
    """
    if not prompt_features or not trajectory_features:
        raise ValueError("RM scoring requires non-empty prompt and trajectory features")
    if len(prompt_features) != len(trajectory_features):
        raise ValueError(
            "RM prompt/trajectory length mismatch: "
            f"{len(prompt_features)} != {len(trajectory_features)}"
        )

    prompt = np.stack(prompt_features, axis=0)
    trajectory = np.stack(trajectory_features, axis=0)
    combined = np.concatenate((prompt, trajectory), axis=-1)
    if combined.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Evaluator feature dimension mismatch: {combined.shape[1]} != {FEATURE_DIM}"
        )
    combined = combined[:, reward_model.eval_feature_indices]
    if combined.shape[1] != reward_model.input_dim:
        raise ValueError(
            f"RM feature dimension mismatch: {combined.shape[1]} != {reward_model.input_dim}"
        )

    # A frame is valid iff it and every frame before it in the sequence is finite: once one
    # frame goes non-finite, everything the network would compute from it onward is fabricated.
    finite_mask = np.isfinite(combined).all(axis=-1)
    valid_len = int(np.argmax(~finite_mask)) if not finite_mask.all() else len(combined)
    if valid_len == 0:
        raise ValueError("RM features are non-finite from the first frame")

    device = next(reward_model.parameters()).device
    seq_len = reward_model.eval_seq_len
    downsample_rate = reward_model.eval_downsample_rate
    scores = []
    score_frames = []
    for start in range(0, valid_len, seq_len):
        chunk_end = min(start + seq_len, valid_len)
        chunk = combined[start:chunk_end:downsample_rate]
        if reward_model.eval_use_padding_mask:
            token_count = math.ceil(seq_len / downsample_rate)
            valid_tokens = len(chunk)
            padded = np.zeros((token_count, reward_model.input_dim), dtype=np.float32)
            padded[:valid_tokens] = chunk
            chunk = padded
            valid_mask = torch.arange(token_count, device=device).unsqueeze(0) < valid_tokens
        else:
            valid_mask = None
        chunk_t = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
        with torch.no_grad():
            score = float(reward_model.score(chunk_t, valid_mask).item())
        if not np.isfinite(score):
            raise ValueError("Reward model returned a non-finite score on finite input")
        scores.append(score)
        score_frames.append(chunk_end - start)
    return {
        "score": float(np.average(scores, weights=score_frames)),
        "score_frames": sum(score_frames),
        "nonfinite_frame": valid_len if valid_len < len(combined) else None,
    }
