# Tracker evaluation

One entry point drives every supported tracker:

```bash
python -m humantracker.eval.eval_parallel_tracker --tracker <tracker> --rm_checkpoint <path>
```

Supported trackers are `sonic`, `twist2`, `gmt`, `hgpt` and `hgpt_sparse` — one
module each under `backends/`, registered in `backends/__init__.py`. Each backend declares the flags only it needs, so `--help`
shows the selected tracker's options and no others. HoloMotion is pinned to v1.4.1
and validates the exact official `model_16200.onnx` SHA-256 before workers start.

`--termination_metric whole_body` applies SONIC's official evaluation
termination terms uniformly to every tracker.

- `backends/` holds the policy-specific simulation loops. Each module implements the
  backend protocol documented in `backends/__init__.py` and exposes no command-line
  entry point of its own.
- `runner.py` holds the dataset validation, worker setup and result output that are
  the same for every tracker; `paths.py` resolves paths against the repository root.
- `core/` holds the shared smoothness metrics, termination metrics, HumanScore
  feature extraction and rollout export.

All backends use the same HumanTracker environment. `native/` contains the
required HGPT runtime components; no external source
checkout is imported. See `native/SOURCES.md` for provenance and license status.
Policies live under `storage/checkpoints/trackers/`; obtain public weights with
`python -m humantracker.download_weights --tracker all`. Datasets and the
HumanScore checkpoint are still separate downloads described in the main README.

`hgpt_sparse` shares HGPT's simulator and full-body metrics but replaces its
policy observation with the native 182-D five-point layout. It consumes pelvis,
left/right ankle and left/right wrist positions, rotations and spatial velocities,
plus proprioception and root commands. Supply its ONNX checkpoint separately;
it is not part of the public weight downloader. See the main README for the
verified artifact, input contract and an evaluation command. `--privileged` is
rejected; dense HGPT remains available as `--tracker hgpt`.
