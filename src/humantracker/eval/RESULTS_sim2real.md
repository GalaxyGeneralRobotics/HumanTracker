# Released sim2real trackers on the HumanTracker test set

Evaluated on 2026-09-28 with all 2,500 HumanTracker test motions. The
previously reported trackers were not rerun. HumanScore is shown on the same
0–100 scale as the public README; raw result JSON uses a 0–1 score.

| Tracker | HumanScore ↑ | Success ↑ | Mean length | MPJPE ↓ (rad) | Root position ↓ (mm) | Exceptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ScaleBFM-M | 47.49 | 86.52% | 0.9126 | 0.0933 | 134.47 | 0 |
| HEFT | 47.25 | 88.16% | 0.9273 | 0.0621 | 140.34 | 0 |
| ScaleBFM-XL | 46.14 | 86.52% | 0.9144 | 0.0909 | 131.87 | 0 |
| GRIT v0.0.1 | 42.68 | 85.16% | 0.9074 | 0.1129 | 191.50 | 0 |
| TeleopIT* | 38.61 | 81.56% | 0.8853 | 0.1044 | 130.97 | 0 |
| MimicLite-PPO | 31.24 | 80.08% | 0.8806 | 0.0893 | 132.24 | 0 |
| MimicLite-ROA | 28.59 | 80.52% | 0.8837 | 0.0931 | 181.37 | 0 |
| HoloMotion | 28.28 | 82.24% | 0.8943 | 0.0922 | 235.05 | 0 |
| Mimic Lite v1.1 | 28.12 | 78.52% | 0.8714 | 0.0899 | 187.05 | 0 |

Each result covers Daily 974, Ground 164, Highly Dynamic 268, and Interaction
1,094 motions. Every run was checked against the test manifest and had zero
evaluation exceptions. These are HumanTracker frame-weighted HumanScores, not
the public sim2real leaderboard metric.

## Per-family results

Each cell shows HumanScore (0–100) / success rate.

| Tracker | Daily (974) | Highly Dynamic (268) | Ground (164) | Interaction (1,094) |
| --- | ---: | ---: | ---: | ---: |
| ScaleBFM-M | 44.5 / 90.8% | 39.3 / 72.4% | 43.4 / 17.7% | 54.0 / 96.5% |
| HEFT | 46.1 / 93.2% | 40.5 / 75.7% | 31.7 / 15.9% | 51.6 / 97.5% |
| ScaleBFM-XL | 44.2 / 90.1% | 37.5 / 74.3% | 40.8 / 17.1% | 51.2 / 96.7% |
| GRIT v0.0.1 | 40.2 / 90.2% | 41.4 / 65.7% | 24.9 / 11.0% | 48.5 / 96.5% |
| TeleopIT* | 34.0 / 84.0% | 35.8 / 54.9% | 47.1 / 13.4% | 45.7 / 96.2% |
| MimicLite-PPO | 28.0 / 81.1% | 28.1 / 58.6% | 20.1 / 7.9% | 37.9 / 95.2% |
| MimicLite-ROA | 24.8 / 83.1% | 26.9 / 59.0% | 26.6 / 2.4% | 35.1 / 95.2% |
| HoloMotion | 20.4 / 84.0% | 27.2 / 63.4% | 35.1 / 18.3% | 40.4 / 94.9% |
| Mimic Lite v1.1 | 25.0 / 80.1% | 22.8 / 51.5% | 23.1 / 7.3% | 34.3 / 94.4% |

## Reproduction settings

- HumanTracker evaluation adapter: commit `fdea849` and later documentation
  updates. The seven non-ScaleBFM trackers ran before the inference-counter
  correction; only ScaleBFM observations read that counter. Both ScaleBFM
  models were rerun from scratch with the corrected adapter.
- Test manifest: HumanTracker test split (`test.json`),
  SHA-256 `52140372d0943cb749874fe8ce672679ec8d850740095c17feb410dacdeaf406`.
- HumanScore checkpoint: `storage/checkpoints/reward_model/best.pt`,
  SHA-256 `f1bec3c1a6f8596bda0b5aa2891a8c8e3b90d7f24c1f3eb7068ac760393adca0`.
- Upstream policy runtime: [sim2real](https://github.com/EGalahad/sim2real)
  commit `0962762449a902eae0646d108cd57b355d79e093`; released ONNX/YAML
  file hashes are in [sim2real_assets.json](sim2real_assets.json).
- Common HumanTracker G1 scene, 50 Hz control, 0.005 s physics step, native
  G1 torque limits, `whole_body` termination, observation seed 0, ONNX CPU
  inference and CUDA HumanScore. Each policy used its unchanged YAML joint
  order, default poses, action scales, `kp`, `kd`, and motion settings.
- Concurrent runs used eight workers per shard and an isolated any4hdmi FK
  cache per tracker. The [evaluation guide](README_sim2real.md) documents
  the motion view, cache settings, and shard merger.

*TeleopIT release YAML references a legacy robot XML absent from the released
checkout. The run linked that path to the official Teleopit `g1_29dof.xml`
model (SHA-256 `512ccfe8b811e2bfaec0c2fc57960941371e189b365db5c1ff874474166800a3`).
Exact equivalence to the missing legacy XML cannot be verified; interpret
its score with that limitation.
