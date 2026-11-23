#!/usr/bin/env bash
# Start Kev-27B's /v1/systemone server on this node.
# One GPU with >= 80 GB: Kev's normal path (CUDA graphs and fused kernels on).
# Several smaller GPUs (e.g. two 48 GB A6000s): our split-gpus.patch spreads the layers across them; CUDA graphs and
# fused kernels are turned off because both assume one GPU, so it is slower than Kev's published numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  N_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c GPU || echo 1)
fi
if [ "$N_GPUS" -gt 1 ]; then
  export KEV_SPLIT_GPUS="$N_GPUS" KEV_CUDA_GRAPHS=0 KEV_FUSED=0
  export KEV_SPLIT_MAX_MEMORY="${KEV_SPLIT_MAX_MEMORY:-36GiB}"   # per GPU, for weights; the rest is left for activations and caches
  echo "Kev-27B: splitting across $N_GPUS GPUs (CUDA graphs and fused kernels off)"
fi

cd "$WORK_DIR/kev"
exec .venv/bin/python -m kev.serve --run jaredpalmer/kev-27b --host 127.0.0.1 --port "$KEV_PORT"
