#!/usr/bin/env bash
# Clone the pinned upstream trackers and apply this repository's patches to them.
#
# Why patches instead of forks: the pinned commits stay exactly upstream's, and every
# local edit is reviewable in one place. See thirdparty/patches/README.md for the
# rationale behind each one. Re-running this script is safe -- a checkout already at
# its pinned commit, or a patch already applied, is detected and skipped.

set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is required: GR00T-WholeBodyControl distributes assets through it." >&2
    exit 1
fi

# upstream url, pinned commit -- one pair per thirdparty/<dir>, in the same order as
# the table in README.md.
declare -A UPSTREAM_URL=(
    [GR00T-WholeBodyControl]=https://github.com/NVlabs/GR00T-WholeBodyControl.git
    [Humanoid-GPT]=https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.git
    [TWIST2]=https://github.com/amazon-far/TWIST2.git
    [humanoid-general-motion-tracking]=https://github.com/zixuan417/humanoid-general-motion-tracking.git
)
declare -A UPSTREAM_COMMIT=(
    [GR00T-WholeBodyControl]=c3562ef0c303d888cdf26eef50ff9683447207fe
    [Humanoid-GPT]=457a040d2cc426e66a7ebf9fa84c05b5ed534b47
    [TWIST2]=d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a
    [humanoid-general-motion-tracking]=2a590de25a1eb08e47491977a738549c22f16e1f
)

echo "==> cloning upstream trackers at their pinned commits"
for name in "${!UPSTREAM_URL[@]}"; do
    dir="thirdparty/$name"
    if [ ! -e "$dir/.git" ]; then
        git clone "${UPSTREAM_URL[$name]}" "$dir"
    fi
    current="$(git -C "$dir" rev-parse HEAD)"
    if [ "$current" != "${UPSTREAM_COMMIT[$name]}" ]; then
        # Not a shallow fetch by sha: most git servers do not allow arbitrary
        # commits to be fetched that way, only branch/tag tips.
        git -C "$dir" fetch origin
        git -C "$dir" checkout "${UPSTREAM_COMMIT[$name]}"
    fi
done

echo "==> applying patches"
for dir in thirdparty/patches/*/; do
    name="$(basename "$dir")"
    repo="$root/thirdparty/$name"
    if [ ! -e "$repo/.git" ]; then
        echo "no checkout at thirdparty/$name -- run the clone step above first" >&2
        exit 1
    fi
    for patch in "$dir"*.patch; do
        label="${patch#thirdparty/patches/}"
        if git -C "$repo" apply --reverse --check "$root/$patch" 2>/dev/null; then
            echo "    already applied  $label"
        else
            git -C "$repo" apply "$root/$patch"
            echo "    applied          $label"
        fi
    done
done

echo "==> checkout state"
for name in "${!UPSTREAM_URL[@]}"; do
    printf '  %-40s %s\n' "$name" "$(git -C "thirdparty/$name" rev-parse HEAD)"
done
