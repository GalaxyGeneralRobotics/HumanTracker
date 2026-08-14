# HumanTracker tools

`tool/` holds the two utilities the released pipeline depends on. Source
directories use lowercase `snake_case`; generated data belongs under `storage/`
or `outputs/` and is never tracked.

| Directory | Responsibility | Entry point |
| --- | --- | --- |
| `rm_pipeline/` | Rollout clipping, preference-pair construction, validation, aggregation, and reward-model export | `python -m tool.rm_pipeline --help` |
| `motion_annotation/` | Pairwise motion rendering and human annotation | `bash tool/motion_annotation/run_prerender_all.sh` |

Canonical tracker rollouts live under `storage/dataset/tracker_rollouts/`;
preference artifacts live under `storage/preference_pair/`.

Runtime readers accept only the canonical schema. Records that do not match
fail validation; nothing is silently coerced or padded.
