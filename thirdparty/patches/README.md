# Patches against the pinned upstream checkouts

`thirdparty/` holds the four evaluated trackers as plain clones, checked out at the
commits the reported numbers were produced with. Some of them need small edits to run
inside this repository. Those edits are kept here as patch files rather than committed
into the clones, so the pins stay honest: what you get is upstream's code plus exactly
the diffs listed below.

[`../../setup_thirdparty.sh`](../../setup_thirdparty.sh) clones each tracker and applies
every patch in this directory. Each subdirectory is named after the checkout it
patches; patches inside it apply in filename order.

## Humanoid-GPT

**`0001-headless-video-export.patch`** — `scripts/vis.py` unconditionally opened a GLFW
`launch_passive` viewer, which cannot be created on a headless machine. Video export
now renders offscreen through EGL, and the interactive viewer is built only when no
video path is given. Required by
[`visualize_jerk_p95.py`](../../src/humantracker/data/visualize_jerk_p95.py), which
drives `vis.py` on a GPU server to render the jerk figures.

**`0002-standalone-script-imports.patch`** — `tracking/convert_qpos2kpt.py` uses
package-relative imports, so importing it from outside the Humanoid-GPT tree fails.
The patch prepends the checkout root to `sys.path` when the module is loaded outside a
package. Required by [`hgpt.py`](../../src/humantracker/eval/backends/hgpt.py), which
imports `qpos2kpt` directly.

## Not patched

`TWIST2`, `GR00T-WholeBodyControl` and `humanoid-general-motion-tracking` run
unmodified at their pinned commits.

Two local changes to Humanoid-GPT are deliberately *not* carried here:
`tracking/convert_parallel.py` (recursive conversion preserving the source directory
layout) and the README hunk documenting it. No released code path reaches
`convert_parallel.py`, so a patch for it would be maintenance with no effect on
anything in this repository.
