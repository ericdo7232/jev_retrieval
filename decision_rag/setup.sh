#!/usr/bin/env bash
# Python environment for decision_rag in $DRAG_DIR/.venv. Run inside a job (slurm/prepare.sub runs it).
#
# Default: vLLM (for Qwen3-8B descriptions) plus the data and clustering packages; vLLM brings its own torch.
# If the GPU node's driver is too old for that torch, skip vLLM and pick a torch build for an older CUDA:
#   DRAG_VLLM=0 TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash setup.sh
# The description step then uses transformers (slower, same greedy outputs up to numerics).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$DRAG_DIR"
[ -d "$DRAG_DIR/.venv" ] || uv venv --python 3.12 "$DRAG_DIR/.venv"
PY="$DRAG_DIR/.venv/bin/python"

COMMON=(faiss-cpu pysbd pyarrow pandas pyyaml huggingface_hub numpy)
if [ "${DRAG_VLLM:-1}" = 1 ]; then
  uv pip install --python "$PY" vllm "${COMMON[@]}"
else
  EXTRA=()
  [ -n "${TORCH_INDEX:-}" ] && EXTRA=(--extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match)
  uv pip install --python "$PY" ${EXTRA[@]+"${EXTRA[@]}"} torch transformers accelerate "${COMMON[@]}"
fi
"$PY" -c "import torch, transformers, faiss, pysbd; print('torch', torch.__version__, 'CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| faiss', faiss.__version__)"
"$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm not installed: descriptions will use transformers"
