#!/usr/bin/env bash
# Fetch what the tracker backends need from outside this repository.
#
#   1. Native backends: policy weights only, SHA-256 checked by
#      humantracker.download_weights. Arguments are forwarded to it, e.g.
#      `./setup_thirdparty.sh --tracker hgpt`.
#   2. sim2real backend: the sim2real checkout at its pinned commit, with the patches
#      in thirdparty/patches/sim2real applied. See thirdparty/patches/README.md.
#
# Re-running is safe: verified weights, a checkout already at its pin and patches
# already applied are detected and skipped.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"

echo "==> native tracker weights"
"${PYTHON:-python}" -m humantracker.download_weights "$@"

sim2real_url=https://github.com/EGalahad/sim2real.git
sim2real_commit=0962762449a902eae0646d108cd57b355d79e093
sim2real_dir=thirdparty/sim2real

echo "==> sim2real at $sim2real_commit"
if [ ! -e "$sim2real_dir/.git" ]; then
    git clone "$sim2real_url" "$sim2real_dir"
fi
if [ "$(git -C "$sim2real_dir" rev-parse HEAD)" != "$sim2real_commit" ]; then
    # Not a shallow fetch by sha: most git servers only serve branch/tag tips that way.
    git -C "$sim2real_dir" fetch origin
    git -C "$sim2real_dir" checkout "$sim2real_commit"
fi

for patch in thirdparty/patches/sim2real/*.patch; do
    if git -C "$sim2real_dir" apply --reverse --check "$repo/$patch" 2>/dev/null; then
        echo "    already applied  ${patch#thirdparty/patches/}"
    else
        git -C "$sim2real_dir" apply "$repo/$patch"
        echo "    applied          ${patch#thirdparty/patches/}"
    fi
done
