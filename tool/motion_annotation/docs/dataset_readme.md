# Annotation dataset

The web labeler writes annotations through
`utils.data_collector.hf_recorder.HFRecorder`. Application code should submit
labels through `/api/label`; it should not construct alternate row layouts.

Use the strict pipeline commands to consume the output:

```bash
python -m tool.rm_pipeline aggregate \
  --inputs storage/preference_pair/<annotation_run> \
  --output outputs/<run>/aggregates.jsonl

python -m tool.rm_pipeline export-rm-parquet \
  --aggregates outputs/<run>/aggregates.jsonl \
  --output outputs/<run>/reward_model_pairs.parquet
```

Field definitions are documented in `tool/rm_pipeline/data_formats.md`. Rows
that do not match are rejected, never coerced.
