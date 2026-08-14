# Tracker evaluation

One entry point drives every supported tracker:

```bash
python -m humantracker.eval.eval_parallel_tracker --tracker <tracker> --rm_checkpoint <path>
```

Supported trackers are `sonic`, `twist2`, `gmt` and `hgpt` — one module each under
`backends/`, registered in `backends/__init__.py`. Each backend declares the flags
only it needs, so `--help` shows the selected tracker's options and no others.

`--termination_metric whole_body` applies SONIC's official evaluation
termination terms uniformly to every tracker.

- `backends/` holds the policy-specific simulation loops. Each module implements the
  backend protocol documented in `backends/__init__.py` and exposes no command-line
  entry point of its own.
- `runner.py` holds the dataset validation, worker setup and result output that are
  the same for every tracker; `paths.py` resolves paths against the repository root.
- `core/` holds the shared smoothness metrics, termination metrics, HumanScore
  feature extraction and rollout export.

`hgpt` imports from `thirdparty/Humanoid-GPT`, which pins `numpy`/`jax`/`mujoco`
versions incompatible with the rest of this repository, so it needs its own
environment; see the repository README.
