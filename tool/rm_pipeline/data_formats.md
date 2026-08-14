# Preference pipeline data formats

All paths are persisted explicitly. Readers do not infer tracker names,
candidate identities, frame ranges, or alternate column names.

Readers validate structure — required fields, declared row counts and frame
alignment — and reject anything that does not match.

## Full rollout

Each rollout NPZ represents one tracker on one complete source motion. Required
frame-aligned arrays are:

| Field | Shape | Meaning |
| --- | --- | --- |
| `ref_pose` | `(T, 7)` | Reference root xyz and wxyz quaternion |
| `ref_joint_pos` | `(T, 29)` | Reference G1 joints |
| `imu_pose` | `(T, 7)` | Rollout root pose |
| `joint_pos` | `(T, 29)` | Rollout joints |

`<rollout>.meta.json` must contain:

- `motion_id`, `tracker`, `category`
- `source_motion_path`, `rollout_npz_path`, `log_folder`
- `fps`, `num_frames`, `source_start_frame`, `source_end_frame`

The four required arrays must have exactly `num_frames` rows. Both stored paths
must exist. Missing metadata is an error.

## Clip manifest

`clips.jsonl` holds one row per clip, recording `clip_id`, `motion_id`,
`tracker`, `category`, `traj_path`, source paths, global and local frame
ranges, `fps`, `num_frames`, and `duration_sec`.

By default, `clip-rollouts` writes full fixed-duration clips and drops the tail.
`--keep-tail` explicitly retains a shorter final clip.

## Pair manifest

`pairs.jsonl` and `pairs.parquet` hold the same records in two encodings.

| Field | Meaning |
| --- | --- |
| `pair_idx`, `pair_id` | Sequential index and stable pair identity |
| `motion_idx`, `motion_id` | Selected motion identity |
| `clip_idx`, `clip_uid`, `source_clip_idx` | Aligned clip identities |
| `category`, `split` | Motion taxonomy and source split |
| `tracker_pair_key`, `tracker_pair` | Canonically ordered tracker pair |
| `source_start_frame`, `source_end_frame`, `fps` | Shared source window |
| `candidates` | Exactly two canonical candidates |

Candidates are stored in `candidate_idx` order `[0, 1]`. Each candidate carries
`candidate_uid`, `tracker`, `traj_path`, provenance paths, motion identity, and
complete local/source frame metadata. Browser pane position is never candidate
identity.

## Dynamic annotation state

`dynamic_state.json` records the exact `pairs_path`, target annotation count,
lease duration, completed annotations, and active leases. Every lease stores `display_order` as `[0, 1]`
or `[1, 0]`; no `swapped` alias is accepted.

## Annotation shards

Annotations are written as JSON, then as Parquet shards with a companion
index. Each record contains canonical candidates sorted by `candidate_idx`, a
`choice_type` (`preference`, `similar`, or `bad_traj`), and nullable
`preferred_candidate_idx`. `comparison.display_order` and
`comparison.display_choice_idx` are audit fields only.

Shard ranges are contiguous and immutable. The index row counts, listed shard
set, and files on disk must agree exactly.

## Aggregates and RM export

`aggregated.jsonl` holds one majority-vote result per pair. Votes are counted
against stable `candidate_idx`; tied majorities, `similar`, and `bad_traj` remain
invalid aggregates.

The RM export parquet contains:

- `pair_id`, `motion_id`, `clip_uid`
- `chosen_path`, `rejected_path`
- `chosen_tracker`, `rejected_tracker`
- source frame range, `fps`, `duration_sec`
- annotation count, vote JSON, per-annotation JSON, candidate metadata JSON,
  and full aggregate JSON

Nothing is coerced or defaulted on read: a record missing any field above
fails validation.
