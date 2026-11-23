#!/usr/bin/env bash
# Install pplx-decider-v1.2's environment and download the checkpoint (~52 GB) into $WORK_DIR/pplx.
# Run as a CPU job: sbatch slurm/setup.sub
#
# The checkpoint's requirements-fast.txt pins torch 2.14.0, transformers 5.17.0 and flash-linear-attention 0.5.2.
# torch from PyPI is built for a recent CUDA. If the smoke test reports that CUDA is unavailable (driver too old),
# reinstall with a wheel for an older CUDA, e.g.:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash pplx/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/"; exit 1; }
D="$WORK_DIR/pplx"
mkdir -p "$D"
cd "$D"
[ -d .venv ] || uv venv --python 3.12 .venv
PY="$D/.venv/bin/python"

uv pip install --python "$PY" "huggingface_hub[cli]==1.31.0"
"$D/.venv/bin/hf" download perplexity-ai/pplx-decider-v1.2 --local-dir "$PPLX_CKPT"

EXTRA=()
[ -n "${TORCH_INDEX:-}" ] && EXTRA=(--extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match)
uv pip install --python "$PY" ${EXTRA[@]+"${EXTRA[@]}"} -r "$PPLX_CKPT/requirements-fast.txt"

echo "Checking the installation ..."
"$PY" - "$PPLX_CKPT" <<'EOF'
import sys
sys.path.insert(0, f"{sys.argv[1]}/source/src")
import torch, transformers, fla, pplx_decider
from torch.nn.attention.varlen import varlen_attn  # noqa: F401
try:
    from torch.nn.attention.varlen import AuxRequest  # noqa: F401
    lse = "yes"
except ImportError:
    lse = "no (forked scoring falls back to slower PyTorch attention over the shared state)"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, "
      f"flash-linear-attention {getattr(fla, '__version__', '?')}, pplx_decider {pplx_decider.__version__}")
print(f"varlen attention returns log-sum-exp: {lse}")
EOF
echo "Done. Checkpoint: $PPLX_CKPT"
