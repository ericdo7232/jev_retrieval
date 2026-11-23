#!/usr/bin/env bash
# Install Kev-27B's server and download the weights (~51 GB). Run once, on a login node or CPU job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Do not `pip install kev`: that name is a different package on PyPI.
# Pinned to the commit split-gpus.patch was written against (2026-11-20).
KEV_COMMIT=65b1e763943ace53fd3a4aae00d8d08dbd849772
[ -d kev ] || git clone https://github.com/jaredpalmer/kev.git
cd kev
git fetch -q origin || true
git checkout -q "$KEV_COMMIT"
echo "kev commit: $(git rev-parse HEAD)"

# Our patch: KEV_SPLIT_GPUS=N splits the model across N GPUs (Kev-27B needs 51 GB; an A6000 has 48 GB).
PATCH="$HERE/split-gpus.patch"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "split-gpus.patch already applied"
else
  git apply "$PATCH" && echo "applied split-gpus.patch"
fi
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
