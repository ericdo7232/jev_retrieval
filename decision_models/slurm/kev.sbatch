#!/bin/bash
# Test 1: Kev-27B. Submit from decision_models/:   sbatch slurm/kev.sbatch
# Or, inside an interactive GPU session:           bash slurm/kev.sbatch
#SBATCH --job-name=kev-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Uncomment and set for your cluster. The GPU needs >= 80 GB (A100-80GB, H100, H200).
##SBATCH --partition=YOUR_PARTITION
##SBATCH --constraint=YOUR_80GB_GPU_FEATURE
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source ./env.sh
mkdir -p logs results
JOB="${SLURM_JOB_ID:-local}"
PY="$WORK_DIR/kev/.venv/bin/python"

setsid bash kev/serve.sh > "logs/kev-server-$JOB.log" 2>&1 &
PID=$!
"$PY" bench.py --model kev --url "http://127.0.0.1:$KEV_PORT" --tag "${TAG:-default}" \
  --wait 1800 --server-pid "$PID" || echo "bench.py exited with status $?"
kill -TERM -- "-$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
