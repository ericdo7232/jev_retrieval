#!/usr/bin/env bash
# Start Kev-27B's /v1/systemone server on this node (needs one GPU with >= 80 GB).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
cd "$WORK_DIR/kev"
exec .venv/bin/python -m kev.serve --run jaredpalmer/kev-27b --host 127.0.0.1 --port "$KEV_PORT"
