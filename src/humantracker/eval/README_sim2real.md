# Released sim2real policies on the HumanTracker test set

Nine additional G1 trackers share one backend, `backends/sim2real.py`. It loads
each released `policy.yaml` through sim2real's own observation and action runtime.
The YAML supplies joint order, future motion frames, action scale, `kp` and `kd`;
the adapter does not replace those values. It also advances the upstream
inference counter each step so ScaleBFM buffered observations refresh. All
trackers use the HumanTracker G1
scene, 50 Hz control, 0.005 s physics step, HumanTracker termination and
HumanScore. The motor torque caps come from sim2real's G1 config. Only the motion
path and motion file backend are set at runtime. See the
[2,500-motion results](RESULTS_sim2real.md).

| `--tracker` | Released policy directory |
| --- | --- |
| `heft` | `heft/pmg` |
| `holomotion` | `holomotion/v1_4_0` |
| `mimiclite-ppo` | `mimic-lite/ppo` |
| `mimiclite-roa` | `mimic-lite/roa` |
| `mimiclite-v1.1` | `mimic-lite/v1_1` |
| `scalebfm-m` | `scalebfm/humanoid_transformer_m` |
| `scalebfm-xl` | `scalebfm/humanoid_transformer_xl` |
| `grit-v0.0.1` | `grit/v0_0_1` |
| `teleopit` | `teleopit` |

The policy names and files follow the [sim2real external-policy guide](https://egalahad.github.io/sim2real/tutorials/run-external-policies/).
The [released ONNX and YAML files](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
go under `thirdparty/sim2real/checkpoints/` in the paths shown above. Weights
remain outside Git. `setup_thirdparty.sh` pins sim2real at
`0962762449a902eae0646d108cd57b355d79e093` and applies two small patches: lazy loading of the unrelated H2 config and
compatibility with the released GRIT ONNX output name.
The [asset manifest](sim2real_assets.json) records SHA-256 and byte sizes for all
nine YAML/ONNX pairs used in this run.

sim2real uses Python 3.10. Create its environment in `thirdparty/sim2real` with
`uv sync --extra inference-cpu --frozen`; run HumanTracker's source through
`PYTHONPATH` from that interpreter. For GPU HumanScore, install a PyTorch wheel
compatible with the host NVIDIA driver and verify `torch.cuda.is_available()`.
The ONNX policy can remain on CPU via `--inference_backend onnx-cpu`.

The source `test.json` motions lack an any4hdmi manifest. Make a view using hard
links, so the test motions are neither changed nor copied. The output must be on
the same filesystem as the motions:

```bash
export HT_ROOT=/path/to/HumanTracker
export SIM2REAL_ROOT="$HT_ROOT/thirdparty/sim2real"
export DATA_ROOT=/path/to/HumanTracker_convert_low
export MOTION_VIEW="$DATA_ROOT/.humantracker_eval_view"
export PYTHONPATH="$HT_ROOT/src:$SIM2REAL_ROOT"
export G1_MJCF=/path/to/g1-mode_13_15.xml
export RESULT_DIR=/path/to/evaluation-results
mkdir -p "$RESULT_DIR"

"$SIM2REAL_ROOT/.venv/bin/python" -m humantracker.eval.prepare_sim2real_motion_view \
    --mocap_path "$DATA_ROOT" --test_json "$DATA_ROOT/test.json" \
    --robot_xml "$G1_MJCF" \
    --output "$MOTION_VIEW"
```

`G1_MJCF` is sim2real original `g1-mode_13_15.xml` asset, with its meshes available.
This model is used only for policy reference-motion FK. The actual rollout and
HumanTracker metrics still use the common HumanTracker scene. Its G1 model omits
toe-link bodies required by some released observations, so it cannot be used for
the motion view. The manifest stores the original `hf://` reference; keep
the model in your Hugging Face cache, and set `HF_HOME` accordingly. Once cached,
`HF_HUB_OFFLINE=1` avoids network requests in worker processes.

For simultaneous policy runs, use a distinct `ANY4HDMI_QPOS_CACHE_ROOT` per
tracker. The upstream FK cache shares writable files and concurrent policies can
race on a network filesystem. Set `ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0` for these
single-motion loads to avoid spawning nested loader processes.

Then evaluate one policy. Select an idle GPU with `CUDA_VISIBLE_DEVICES`;
`--rm_device cuda:0` refers to that selected GPU. Use a distinct output file per
policy. The evaluator checks all 2,500 manifest entries before starting workers.

```bash
ANY4HDMI_QPOS_CACHE_ROOT="$SIM2REAL_ROOT/.cache/humantracker/heft" \
ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 CUDA_VISIBLE_DEVICES=0 \
"$SIM2REAL_ROOT/.venv/bin/python" \
    -m humantracker.eval.eval_parallel_tracker \
    --tracker heft --sim2real_root "$SIM2REAL_ROOT" --motion_view "$MOTION_VIEW" \
    --mocap_path "$DATA_ROOT" --test_json "$DATA_ROOT/test.json" \
    --rm_checkpoint "$HT_ROOT/storage/checkpoints/reward_model/best.pt" \
    --rm_device cuda:0 --termination_metric whole_body --workers 2 --seed 0 \
    --output_json "$RESULT_DIR/heft.json"
```

For a large test run, start multiple copies with the same `--shard_count` and
unique `--shard_index` values from zero through count minus one. Keep one output
JSON per shard and the same tracker cache root; shard IDs are global test indices.
After every shard finishes, merge and verify coverage before reporting scores:

```bash
"$SIM2REAL_ROOT/.venv/bin/python" -m humantracker.eval.merge_eval_shards \
    --tracker heft --test_json "$DATA_ROOT/test.json" \
    --inputs "$RESULT_DIR"/heft_shard_*.json \
    --output_json "$RESULT_DIR/heft.json"
```

The merger rejects duplicates, missing trajectories, and manifest mismatches,
then recomputes the usual category and overall summaries from all 2,500 rows.

HoloMotion's YAML references its original G1 MJCF on Hugging Face for motion FK.
Cache that asset before an offline run; a proxy or `HF_ENDPOINT` can be used when
the host cannot reach Hugging Face directly. A result file with failed
trajectories includes each exception in `per_trajectory[].error` for diagnosis.

TeleopIT release YAML references `legacy/data/robots/g1/g1-mjlab.xml`, which is
absent from the released sim2real checkout and checkpoint bundle. For this run,
that path is a symlink to `unitree_g1/g1_29dof.xml` extracted from the official
Teleopit model archive (`12e21/Teleopit-models`, revision
`94cf996444fea6894b87c28e86606cd4c2f1408f`). The YAML and controller
settings are unchanged. The substitute XML hash is
`512ccfe8b811e2bfaec0c2fc57960941371e189b365db5c1ff874474166800a3`.
The missing legacy model prevents verification of exact kinematic equivalence;
label TeleopIT scores with this qualification.
