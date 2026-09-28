# Patches against the pinned upstream checkouts

`thirdparty/` holds the evaluated trackers as plain clones, checked out at the
commits the reported numbers were produced with. Some of them need small edits to run
inside this repository. Those edits are kept here as patch files rather than committed
into the clones, so the pins stay honest: what you get is upstream's code plus exactly
the diffs listed below.

[`../../setup_thirdparty.sh`](../../setup_thirdparty.sh) clones each tracker and applies
every patch in this directory. Each subdirectory is named after the checkout it
patches; patches inside it apply in filename order.

## Humanoid-GPT

**`0002-standalone-script-imports.patch`** — `tracking/convert_qpos2kpt.py` uses
package-relative imports, so importing it from outside the Humanoid-GPT tree fails.
The patch prepends the checkout root to `sys.path` when the module is loaded outside a
package. Required by [`hgpt.py`](../../src/humantracker/eval/backends/hgpt.py), which
imports `qpos2kpt` directly.

## sim2real

**`0001-lazy-h2-config.patch`** — loading a G1 policy imported the unrelated H2
config and attempted to download its MJCF. Load H2 only when it is selected, so
G1 evaluation works with an offline checkpoint cache. Policy YAML and weights are
unchanged.

**`0002-grit-output-name.patch`** — the released GRIT ONNX calls its action output
`actions`; the shared policy runtime expected `action`. Accept both output names
without changing the model or action values.

## Not patched

`TWIST2`, `GR00T-WholeBodyControl` and `humanoid-general-motion-tracking` run
unmodified at their pinned commits.

Two local changes to Humanoid-GPT are deliberately *not* carried here:
`tracking/convert_parallel.py` (recursive conversion preserving the source directory
layout) and the README hunk documenting it. No released code path reaches
`convert_parallel.py`, so a patch for it would be maintenance with no effect on
anything in this repository.
