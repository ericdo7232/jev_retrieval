#!/usr/bin/env bash
# Install Kev-27B's server and download the weights (~51 GB). Run once, on a login node or CPU job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Do not `pip install kev`: that name is a different package on PyPI.
[ -d kev ] || git clone https://github.com/jaredpalmer/kev.git
cd kev
echo "kev commit: $(git rev-parse HEAD)"
uv sync --extra serve --python 3.12

# Fused CUDA kernels. The version Kev expects is pinned in kev/fused_qwen35.py.
FLA_VERSION="$(.venv/bin/python - <<'EOF'
import re
try:
    m = re.search(r"FLA_VERSION\s*=\s*['\"]([^'\"]+)", open("kev/fused_qwen35.py").read())
    print(m.group(1) if m else "")
except OSError:
    print("")
EOF
)"
if [ -n "$FLA_VERSION" ]; then
  uv pip install --python .venv/bin/python "flash-linear-attention==$FLA_VERSION"
else
  echo "WARNING: could not find FLA_VERSION in kev/fused_qwen35.py; skipping flash-linear-attention (server will be slower)."
fi

.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('jaredpalmer/kev-27b'))"
echo "Kev-27B setup done."
