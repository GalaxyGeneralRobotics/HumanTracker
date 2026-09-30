# Tracker evaluation

One entry point drives every supported tracker:

```bash
python -m humantracker.eval.eval_parallel_tracker --tracker <tracker> --termination_metric whole_body ...
```

`--termination_metric whole_body` applies SONIC's official evaluation termination
terms uniformly to every tracker. Each backend declares the flags only it needs, so
`--help` shows the selected tracker's options and no others.

| `--tracker` | Backend | Runs from |
| --- | --- | --- |
| `sonic`, `twist2`, `gmt`, `hgpt`, `hgpt_sparse` | one native module each under `backends/` | the HumanTracker environment |
| `heft`, `holomotion`, `mimiclite-ppo`, `mimiclite-roa`, `mimiclite-v1.1`, `scalebfm-m`, `scalebfm-xl`, `grit-v0.0.1`, `teleopit` | `backends/sim2real.py`, shared | sim2real's Python 3.10 environment |

## Layout

- `backends/` holds the policy-specific simulation loops. Each module implements the
  backend protocol documented in `backends/__init__.py` and exposes no command-line
  entry point of its own. `SIM2REAL_POLICIES` there is the one list of sim2real
  trackers.
- `runner.py` holds the dataset validation, worker setup, sharding and result output
  that are the same for every tracker; `paths.py` resolves paths against the
  repository root.
- `core/` holds the shared metrics, reference-noise model, HumanScore feature
  extraction and rollout export.
- `native/` holds the bundled Humanoid-GPT runtime; see `native/SOURCES.md` for
  provenance and license status.
- `sim2real/` holds the sim2real backend's setup, launcher, motion-view builder and
  asset checksums.
- `eval.sh` and `sim2real/eval.sh` launch full test-split runs, one tracker per GPU.
- `merge_eval_shards.py` combines a sharded run.

## Native backends

All native backends share the HumanTracker environment and import no external source
checkout. Their policies live under `storage/checkpoints/trackers/`; obtain the
public weights with `python -m humantracker.download_weights --tracker all`.
Datasets and the HumanScore checkpoint are separate downloads described in the main
README.

`hgpt_sparse` shares HGPT's simulator and full-body metrics but replaces its policy
observation with the native 182-D five-point layout. It consumes pelvis, left/right
ankle and left/right wrist positions, rotations and spatial velocities, plus
proprioception and root commands. Supply its ONNX checkpoint separately; it is not
part of the public weight downloader. See the main README for the verified artifact,
input contract and an evaluation command. `--privileged` is rejected; dense HGPT
remains available as `--tracker hgpt`.

`--ref_noise` corrupts the policy reference of `sonic`, `hgpt` and `hgpt_sparse`
only; metrics, HumanScore and the ghost stay on the clean reference.

## Released policies through sim2real

Nine released G1 trackers run on [sim2real](https://github.com/EGalahad/sim2real)'s
own policy runtime, each from its deploy `policy.yaml` + ONNX pair:

| `--tracker` | Policy directory under `checkpoints/` | Upstream |
| --- | --- | --- |
| `heft` | `heft/pmg` | [HEFT](https://heft.axell.top/) |
| `holomotion` | `holomotion/v1_4_0` | [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) |
| `mimiclite-ppo` | `mimic-lite/ppo` | [MimicLite](https://github.com/Roboparty/MimicLite) |
| `mimiclite-roa` | `mimic-lite/roa` | [MimicLite](https://github.com/Roboparty/MimicLite) |
| `mimiclite-v1.1` | `mimic-lite/v1_1` | [Mimic Lite v1.1](https://github.com/EGalahad/sim2real#mimic-lite-v11) |
| `scalebfm-m` | `scalebfm/humanoid_transformer_m` | [ScaleBFM](https://github.com/zengweishuai/ScaleBFM) |
| `scalebfm-xl` | `scalebfm/humanoid_transformer_xl` | [ScaleBFM](https://github.com/zengweishuai/ScaleBFM) |
| `grit-v0.0.1` | `grit/v0_0_1` | [GRIT](https://github.com/mrzuang/GRIT_teleop_deploy) |
| `teleopit` | `teleopit` | [TeleopIT](https://github.com/BotRunner64/Teleopit) |

The YAML supplies joint order, future motion frames, action scale, `kp` and `kd`;
the adapter replaces none of them. It only provides the HumanTracker G1 scene, 50 Hz
control with a 0.005 s physics step, sim2real's G1 torque limits, the reference
motion, the HumanTracker termination rule and HumanScore. Like upstream's loop it
advances the runtime's inference counter every step, which ScaleBFM's buffered
observations need to refresh. A control period that leaves a non-finite state, or in
which MuJoCo auto-resets a diverged simulation, fails the trajectory rather than
letting it be scored from the reset pose. Besides the common metrics each trajectory
records `joint_jerk_until_termination`, the joint jerk over its first
`jerk_prefix_frames` (executed) frames. It does not render videos or inject
reference noise.

### Setup

```bash
./setup_thirdparty.sh                        # clones and patches thirdparty/sim2real
bash src/humantracker/eval/sim2real/setup.sh # environment, extra assets, checksums
```

`setup.sh` builds sim2real's environment with `uv sync --frozen --extra
inference-cpu`, then needs the released policies in `thirdparty/sim2real/checkpoints/`
at the paths above:

- **Google Drive** — the [sim2real artifacts folder](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
  holds every `policy.yaml` and all ONNX files except HoloMotion's. Download its
  `checkpoints/` by hand; `setup.sh` lists anything still missing.
- **Hugging Face** — `setup.sh` fetches the HoloMotion v1.4.0 ONNX
  ([`model_14000.onnx`](https://huggingface.co/HorizonRobotics/HoloMotion_models/blob/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx),
  1.64 GB) and TeleopIT's robot model (below) itself. Set
  `HF_ENDPOINT=https://hf-mirror.com` where Hugging Face is unreachable.

It then checks every file against [`sim2real/assets.sha256`](sim2real/assets.sha256),
the exact artifacts the reported results used. Weights remain outside Git.

TeleopIT's YAML names `legacy/data/robots/g1/g1-mjlab.xml`, which neither the
sim2real checkout nor the checkpoint bundle contains. `setup.sh` links that path to
`unitree_g1/g1_29dof.xml` from the official Teleopit model archive
([`12e21/Teleopit-models`](https://huggingface.co/12e21/Teleopit-models) at revision
`94cf996444fea6894b87c28e86606cd4c2f1408f`). The YAML and controller settings are
unchanged, but equivalence to the missing legacy model cannot be verified: qualify
TeleopIT's scores accordingly.

### Running

```bash
export HUMANTRACKER_DATASET=/path/to/HumanTracker
bash src/humantracker/eval/sim2real/eval.sh              # all nine, one per GPU
bash src/humantracker/eval/sim2real/eval.sh heft teleopit
```

The launcher first builds the motion view, then runs each tracker from
`thirdparty/sim2real/.venv/bin/python` with `PYTHONPATH=src`. sim2real reads
reference motions as an any4hdmi dataset, which the test motions are not;
[`prepare_motion_view`](sim2real/prepare_motion_view.py) makes one by hard-linking
the manifest's motions beside an any4hdmi manifest, so the view must be on the
dataset's filesystem. It also caches sim2real's G1 model — the policies' FK model,
which has the toe links some observations read — so the runs can set
`HF_HUB_OFFLINE=1`. Each tracker gets its own `ANY4HDMI_QPOS_CACHE_ROOT`: the
upstream FK cache is shared writable state, and concurrent policies race on it.

To run one tracker by hand, reproduce the launcher's command:

```bash
thirdparty/sim2real/.venv/bin/python -m humantracker.eval.sim2real.prepare_motion_view \
    --mocap_path "$HUMANTRACKER_DATASET" --test_json "$HUMANTRACKER_DATASET/test.json" \
    --output "$HUMANTRACKER_DATASET/.humantracker_eval_view"

PYTHONPATH=src HF_HUB_OFFLINE=1 ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 \
ANY4HDMI_QPOS_CACHE_ROOT=thirdparty/sim2real/.cache/humantracker/heft \
thirdparty/sim2real/.venv/bin/python -m humantracker.eval.eval_parallel_tracker \
    --tracker heft --sim2real_root thirdparty/sim2real \
    --motion_view "$HUMANTRACKER_DATASET/.humantracker_eval_view" \
    --mocap_path "$HUMANTRACKER_DATASET" --test_json "$HUMANTRACKER_DATASET/test.json" \
    --rm_device cuda:0 --termination_metric whole_body --workers 8 --seed 0 \
    --output_json outputs/eval_runs/heft.json
```

The ONNX policy runs on CPU (`--inference_backend onnx-cpu`, the default); HumanScore
uses `--rm_device`. For GPU HumanScore, sim2real's environment needs a PyTorch build
matching the host driver.

## Sharding a run

Any tracker can split one run across processes or machines: start the same command
with `--shard_count N` and each `--shard_index` in `0 … N-1`, one output file per
shard, then merge:

```bash
python -m humantracker.eval.merge_eval_shards \
    --tracker heft --test_json "$HUMANTRACKER_DATASET/test.json" \
    --inputs outputs/eval_runs/heft.shard*.json --output_json outputs/eval_runs/heft.json
```

The merger rejects shards produced with different settings, duplicate or missing
trajectories and rows that do not match the manifest, then recomputes the category
and overall summaries from all rows. Shard ids index the manifest, so shard runs must
not use `--categories` or `--skip_flipped`.

## Released-policy results

The nine sim2real trackers were evaluated on 2026-09-28 on all 2,500 test motions
(Daily 974, Highly Dynamic 268, Interaction 1,094, Ground 164); per-family numbers
are in the main README. HumanScore is on the README's 0–100 scale (result JSON stores
0–1), frame-weighted as for every other tracker; it is not the public sim2real
leaderboard metric.

| Tracker | HumanScore ↑ | Success ↑ | Mean length | MPJPE ↓ (rad) | Root position ↓ (mm) | Exceptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ScaleBFM-M | 47.49 | 86.52% | 0.9126 | 0.0933 | 134.47 | 0 |
| HEFT | 47.25 | 88.16% | 0.9273 | 0.0621 | 140.34 | 0 |
| ScaleBFM-XL | 46.14 | 86.52% | 0.9144 | 0.0909 | 131.87 | 0 |
| GRIT v0.0.1 | 42.68 | 85.16% | 0.9074 | 0.1129 | 191.50 | 0 |
| TeleopIT | 38.61 | 81.56% | 0.8853 | 0.1044 | 130.97 | 0 |
| MimicLite-PPO | 31.24 | 80.08% | 0.8806 | 0.0893 | 132.24 | 0 |
| MimicLite-ROA | 28.59 | 80.52% | 0.8837 | 0.0931 | 181.37 | 0 |
| HoloMotion | 28.28 | 82.24% | 0.8943 | 0.0922 | 235.05 | 0 |
| Mimic Lite v1.1 | 28.12 | 78.52% | 0.8714 | 0.0899 | 187.05 | 0 |

Settings: the adapter of commit `45d5e75`, sim2real `0962762`, the artifacts in
`sim2real/assets.sha256`, `whole_body` termination, observation seed 0, ONNX CPU
inference, CUDA HumanScore, eight workers per process and a private FK cache per
tracker. Test manifest SHA-256
`52140372d0943cb749874fe8ce672679ec8d850740095c17feb410dacdeaf406`; HumanScore
checkpoint `storage/checkpoints/reward_model/best.pt`, SHA-256
`f1bec3c1a6f8596bda0b5aa2891a8c8e3b90d7f24c1f3eb7068ac760393adca0`. The seven
non-ScaleBFM trackers ran before the inference-counter correction, which only
ScaleBFM's observations read; both ScaleBFM models were rerun with it. All nine ran
before the divergence check and the pre-termination jerk were added.
