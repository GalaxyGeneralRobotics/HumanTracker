#!/usr/bin/env bash
# Launch the web-based motion labeling server.
# Usage: ./run.sh --task-file /path/to/pairs.jsonl [--port 7860]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate h-gpt

# motion_annotation runs on headless GPU servers.  MuJoCo selects its OpenGL
# backend when it is first imported, so this must be exported before Python
# starts; setting it later inside video_cache.py is too late.
export MUJOCO_GL=egl

exec python -m web_labeler.app "$@"
