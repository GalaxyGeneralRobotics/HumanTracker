# Patches against the pinned sim2real checkout

The native tracker backends are part of this repository (see
[`native/SOURCES.md`](../../src/humantracker/eval/native/SOURCES.md)). The nine
released policies evaluated through the sim2real backend run on
[sim2real](https://github.com/EGalahad/sim2real) itself, cloned at
`0962762449a902eae0646d108cd57b355d79e093`. Two small edits are needed to run it
inside this evaluator; they are kept here as patches rather than committed into the
clone, so the pin stays honest: upstream's code plus exactly the diffs below.

[`../../setup_thirdparty.sh`](../../setup_thirdparty.sh) clones sim2real and applies
the patches in `sim2real/` in filename order.

**`0001-lazy-h2-config.patch`** — loading a G1 policy imported the unrelated H2
config and attempted to download its MJCF. Load H2 only when it is selected, so
G1 evaluation works with an offline checkpoint cache. Policy YAML and weights are
unchanged.

**`0002-grit-output-name.patch`** — the released GRIT ONNX calls its action output
`actions`; the shared policy runtime expected `action`. Accept both output names
without changing the model or action values.
