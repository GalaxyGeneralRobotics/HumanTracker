# Bundled evaluation runtimes

These files eliminate runtime imports from external project checkouts. They do
not include upstream training, robot deployment, or retargeting applications.
Do not replace policy observation layouts, joint order, PD gains or velocity
conventions without parity tests. Policy weights retain their separate licenses.

## Humanoid-GPT

- Source project: https://github.com/GalaxyGeneralRobotics/Humanoid-GPT
- Source snapshot: the clean local checkout at
  `de8498f4aaa6f1a0ce05ca69531c344e9f4b0c40`, used by this machine before
  native-runtime migration. This is NOT a claim of source parity with the
  earlier official setup pin `9f9e7b7`.
- Extracted from `tracking/constants.py`, `tracking/infer_utils.py`,
  `tracking/convert_qpos2kpt.py`, `utils/transforms_np.py`, `utils/sim_mj.py`.
- Apache-2.0 license is included in `hgpt/LICENSE`.
- Changes: only MuJoCo/ONNX evaluation, qpos resampling/velocity/FK conversion,
  observation construction and action scaling remain. No JAX, MJX, Flax,
  training config, logger, hand-mounting, real-robot or converter video/CLI
  dependencies. The benchmark uses its existing 29-DoF G1 scene and shared PD
  simulator. Configuration is a SimpleNamespace; reference dictionaries are
  sliced directly, not through JAX tree utilities.
- The native conversion, observations and motor targets were checked against
  the pre-migration runtime on six deterministic frames (RNG seed 42, the
  existing benchmark robot XML): conversion and observations agree within
  2e-6, action scaling exactly.
- Default policy remains official `pns_wo_priv264.onnx`, downloaded directly
  from public revision `9f9e7b74ecadb532abbb34b6a779d87191a9bbb6`; its SHA-256
  matches the previously used local artifact.
- Five-point sparse support additionally extracts the five-point configuration
  and observation builder from `projects/sparse_tracking/constants.py` and
  `projects/sparse_tracking/infer_utils.py` at the same source snapshot.
  `hgpt/sparse.py` inherits the bundled runtime; no `projects` or `tracking`
  import is required. Only `5point` is included (182-D input, 29-D output),
  with non-privileged observations and zero local keypoint offsets. Reference
  timing follows `projects/sparse_tracking/eval_parallel.py`, without the
  separate interactive inference command's optional EMA processing.
  A local 100-frame CPU comparison against that original inference class and
  the stage-2 checkpoint gave exactly equal observations and motor targets.
  Weights are separate artifacts; their identity is documented in the README.

## Existing backends and assets

SONIC, TWIST2 and GMT already implement their observation/inference/simulation
paths inside `backends/`. They use standalone policy files under
`storage/checkpoints/trackers/`, as does the extracted Humanoid-GPT runtime; none of
them requires `thirdparty/`, Git, a symlink to another repository, or an upstream
Python package. G1 assets remain under `storage/assets/unitree_g1_5010/` with their
existing license.

The released policies behind the `sim2real` backend are the exception by design:
they run on the pinned, patched sim2real checkout and its own environment rather
than on bundled code. See `thirdparty/patches/README.md`.
