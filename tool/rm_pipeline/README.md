# HumanTracker preference pipeline

This package provides one strict rollout-to-annotation-to-reward-model path.
Malformed rows, missing paths, and misaligned tracker clips are errors.

## Build annotation pairs

The canonical full train rollouts are stored at:

```text
storage/dataset/tracker_rollouts/humantracker_train_4track_20260720_114404
```

Split a rollout dataset into aligned five-second clips:

```bash
python -m tool.rm_pipeline clip-rollouts \
  --rollout-root storage/dataset/tracker_rollouts/<run> \
  --clips-root storage/preference_pair/preference_pipeline/<run>_clips \
  --run-id <YYYYMMDD_HHMMSS> \
  --clip-seconds 5
```

Build the fixed 6,000-pair annotation pool with two arguments:

```bash
python -m tool.rm_pipeline.build_motion_pairs \
  storage/preference_pair/preference_pipeline/<run>_clips/clips.jsonl \
  storage/preference_pair/preference_pipeline/<run>_pairs
```

`build_motion_pairs` concatenates every aligned clip in category/train order,
assigns a global `clip_idx`, and samples 6,000 midpoint-uniform indices. Each
selected clip produces exactly one tracker comparison. The six unordered
tracker combinations contain 1,000 pairs each; every tracker appears 3,000
times and occupies each candidate index 1,500 times. The final annotation order
is deterministically shuffled.

Outputs are `clip_catalog.jsonl`, `selected_clips.jsonl`, `pairs.jsonl`,
`pairs.parquet`, and `summary.json`.

## Validate and label

```bash
python -m tool.rm_pipeline validate-pairs --pairs <pairs.jsonl>
bash tool/motion_annotation/run_prerender_all.sh
```

The labeler dynamically leases tasks; no assignment-file stage exists.

## Aggregate and export

```bash
python -m tool.rm_pipeline aggregate \
  --inputs <annotation_directory> \
  --output <aggregates.jsonl> \
  --min-annotations 3 \
  --min-agreement 2

python -m tool.rm_pipeline export-rm-parquet \
  --aggregates <aggregates.jsonl> \
  --output <reward_model_pairs.parquet>
```

Schema definitions are in `data_formats.md`.
