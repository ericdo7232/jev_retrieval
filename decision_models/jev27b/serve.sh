#!/usr/bin/env bash
# Start JEV-27B's /v1/decide server (vLLM) on this node (needs >= 80 GB of GPU memory in total).
#   PREFIX_CACHING=1 (default)  reuse a shared prompt prefix across requests; README's flags for many questions per state
#   PREFIX_CACHING=0            disable it, as the baseline for the prefix-caching test
#   TRUST_REQUEST_CHAT_TEMPLATE=1  add --trust-request-chat-template (serve_decide.py's docstring asks for it;
#                                  the README command leaves it out)
# Startup takes 3-8 minutes (CUDA-graph capture with LoRA).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
D="$WORK_DIR/jev27b/JEV-27B"

if [ "${PREFIX_CACHING:-1}" = 1 ]; then
  PC=(--enable-prefix-caching --mamba-cache-mode align)
else
  PC=(--no-enable-prefix-caching)
fi
EXTRA=()
[ "${TRUST_REQUEST_CHAT_TEMPLATE:-0}" = 1 ] && EXTRA+=(--trust-request-chat-template)

# Tensor parallelism across all GPUs Slurm gave this job (2 on two A6000s). Override with JEV_TP.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  N_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c GPU || echo 1)
fi
TP="${JEV_TP:-$N_GPUS}"
echo "JEV-27B: tensor parallel size $TP, prefix caching ${PREFIX_CACHING:-1}"

exec "$WORK_DIR/jev27b/.venv/bin/python" "$D/serve_decide.py" \
  --model "$D" --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 --lora-modules "jev-decision=$D/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len "$JEV_MAX_LEN" \
  --tensor-parallel-size "$TP" \
  --host 127.0.0.1 --port "$JEV_PORT" \
  "${PC[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
