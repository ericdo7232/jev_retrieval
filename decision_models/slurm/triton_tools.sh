# Source from a job script after setting PY to the environment's python:   source slurm/triton_tools.sh
#
# Triton (used by torch, flash-linear-attention and vLLM) runs its bundled ptxas/cuobjdump/nvdisasm from the
# environment. On Unicorn those files can't be executed from ~/scratch (Permission denied: missing execute bit or a
# noexec mount). If so, copy them to node-local temporary storage and point Triton at the copies.
TRITON_BIN="$("$PY" -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin"))' 2>/dev/null || true)"
if [ -n "$TRITON_BIN" ] && [ -d "$TRITON_BIN" ]; then
  if "$TRITON_BIN/ptxas" --version >/dev/null 2>&1; then
    echo "Triton tools: $TRITON_BIN (executable in place)"
  else
    echo "Triton tools: $TRITON_BIN not executable here ($(stat -c '%A' "$TRITON_BIN/ptxas" 2>/dev/null), mount: $(findmnt -no OPTIONS -T "$TRITON_BIN" 2>/dev/null | tr ',' ' ' | grep -o 'noexec' || echo 'exec'))"
    LOCAL_BIN="${TMPDIR:-/tmp}/triton-bin-${SLURM_JOB_ID:-$$}"
    mkdir -p "$LOCAL_BIN"
    cp "$TRITON_BIN"/* "$LOCAL_BIN"/
    chmod +x "$LOCAL_BIN"/*
    export TRITON_PTXAS_PATH="$LOCAL_BIN/ptxas"
    [ -f "$LOCAL_BIN/ptxas-blackwell" ] && export TRITON_PTXAS_BLACKWELL_PATH="$LOCAL_BIN/ptxas-blackwell"
    [ -f "$LOCAL_BIN/cuobjdump" ] && export TRITON_CUOBJDUMP_PATH="$LOCAL_BIN/cuobjdump"
    [ -f "$LOCAL_BIN/nvdisasm" ] && export TRITON_NVDISASM_PATH="$LOCAL_BIN/nvdisasm"
    if "$TRITON_PTXAS_PATH" --version >/dev/null 2>&1; then
      echo "Triton tools: using copies in $LOCAL_BIN"
    else
      echo "Triton tools: copies in $LOCAL_BIN are not executable either; GPU kernels will fail to compile"
    fi
  fi
fi
