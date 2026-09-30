#!/usr/bin/env bash
# Prepare the sim2real backend, after ./setup_thirdparty.sh has cloned and patched
# the sim2real checkout.
#
#   1. sim2real's Python 3.10 environment, built by uv from sim2real's own lock file.
#   2. The released policies. The Google Drive folder cannot be fetched by script, so
#      a missing file stops here with the link; the HoloMotion ONNX, which that folder
#      does not carry, is downloaded from Hugging Face.
#   3. The G1 model TeleopIT's YAML names (legacy/data/robots/g1/g1-mjlab.xml) but
#      neither sim2real nor the checkpoint bundle ships: the official Teleopit model
#      archive's unitree_g1/g1_29dof.xml, linked at that path.
#   4. SHA-256 of every file the reported results were produced with (assets.sha256).
#
# Usage: bash src/humantracker/eval/sim2real/setup.sh [sim2real_root]
# Hugging Face downloads honor HF_ENDPOINT (e.g. https://hf-mirror.com) and HF_HOME.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
sim2real="$(cd "${1:-$repo/thirdparty/sim2real}" && pwd)"
python="$sim2real/.venv/bin/python"
drive=https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e

test -e "$sim2real/.git" || { echo "no sim2real checkout at $sim2real; run ./setup_thirdparty.sh" >&2; exit 1; }
cd "$sim2real"

echo "==> environment"
if [ ! -x "$python" ]; then
    uv sync --frozen --extra inference-cpu
fi
"$python" -c "import sim2real, huggingface_hub"

# hf_fetch <repo_id> <filename> <revision> -> prints the cached file's path
hf_fetch() {
    "$python" -c 'import sys; from huggingface_hub import hf_hub_download as get; print(get(sys.argv[1], sys.argv[2], revision=sys.argv[3]))' "$@"
}

echo "==> HoloMotion v1.4.0 ONNX"
holomotion=checkpoints/holomotion/v1_4_0/policy.onnx
if [ ! -f "$holomotion" ]; then
    mkdir -p "$(dirname "$holomotion")"
    cp "$(hf_fetch HorizonRobotics/HoloMotion_models \
        HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx main)" "$holomotion"
fi

echo "==> TeleopIT robot model"
teleopit_xml=.assets/teleopit/unitree_g1/g1_29dof.xml
if [ ! -f "$teleopit_xml" ]; then
    mkdir -p .assets/teleopit
    tar -xzf "$(hf_fetch 12e21/Teleopit-models archives/robot_assets.tar.gz \
        94cf996444fea6894b87c28e86606cd4c2f1408f)" -C .assets/teleopit unitree_g1
fi
mkdir -p legacy/data/robots/g1
ln -sfn "../../../../$teleopit_xml" legacy/data/robots/g1/g1-mjlab.xml

echo "==> released checkpoints"
missing=0
while read -r _ file; do
    if [ ! -f "$file" ]; then
        echo "    missing $file" >&2
        missing=1
    fi
done < "$here/assets.sha256"
if [ "$missing" = 1 ]; then
    echo "Download the checkpoints/ folder from $drive into $sim2real, then re-run." >&2
    exit 1
fi

echo "==> checksums"
sha256sum --check --quiet "$here/assets.sha256"
echo "sim2real backend ready: $sim2real"
