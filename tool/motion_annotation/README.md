# Motion annotation

FastAPI-based pairwise motion annotation for canonical HumanTracker preference
pairs. The server renders both candidates, randomizes only display order,
leases tasks, and writes annotations in exact 200-row Parquet shards.

## One-click run

From the HumanTracker repository root:

```bash
bash tool/motion_annotation/run_prerender_all.sh
```

The fixed launcher starts the current 6,000-pair pool, full pre-rendering on
GPUs 0--7, and the Gradio public tunnel as one process group. Stopping the
launcher stops all three services.

## Custom server

```bash
bash tool/motion_annotation/run.sh \
  --task-file storage/preference_pair/preference_pipeline/<run>/pairs.jsonl \
  --hf-logs-dir storage/preference_pair/hf_preference_<run> \
  --annotations-per-pair 3 \
  --annotators annotator_a annotator_b annotator_c \
  --port 7860
```

Explicit paths must exist. Invalid task records stop startup.

## Required data

- Pair input: the `pairs.jsonl` written by `rm_pipeline`, each row carrying two
  candidates indexed `0` and `1`.
- Annotation output: one record per completed pair, with the choice stored
  against `candidate_idx` rather than screen position.
- Every Parquet shard contains exactly 200 annotations.
- Each annotator's latest five records remain durably journaled in
  `hf_records.pending.jsonl` so they can be undone. Older records are written
  only in exact 200-row Parquet shards; partial shards are never created.
- Dynamic lease state: `dynamic_state.json`, holding the pairs path, target
  count, lease duration, completions and active leases.

## Undo

`Undo` (keyboard shortcut `U`) removes exactly one most-recent submission for
the current annotator and leases that same pair back for correction. Each
annotator has an independent five-record undo stack. An undo never removes a
different annotator's completion, lease, or persisted record.

The recorder rejects `left`/`right`, `candidate_id` and `swapped` fields: a
choice is always stored against `candidate_idx`, never screen position.

## Source layout

```text
motion_annotation/
├── web_labeler/
├── utils/
├── scripts/
├── tests/
├── run.sh
└── run_prerender_all.sh
```

Rendering failures remain visible in `/api/progress`; malformed inputs are not
skipped.
