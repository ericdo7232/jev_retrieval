#!/usr/bin/env bash
# Start JEV-27B's /v1/decide server (vLLM) on this node (needs one GPU with >= 80 GB).
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

exec "$WORK_DIR/jev27b/.venv/bin/python" "$D/serve_decide.py" \
  --model "$D" --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 --lora-modules "jev-decision=$D/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len "$JEV_MAX_LEN" \
  --host 127.0.0.1 --port "$JEV_PORT" \
  "${PC[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
