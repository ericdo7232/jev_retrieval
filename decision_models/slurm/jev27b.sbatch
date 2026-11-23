#!/bin/bash
# Test 2: JEV-27B, run twice: with prefix caching (all tests) and without (smoke + scale, capped).
# Submit from decision_models/:          sbatch slurm/jev27b.sbatch
# Or, inside an interactive GPU session: bash slurm/jev27b.sbatch
#SBATCH --job-name=jev27b-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Uncomment and set for your cluster. The GPU needs >= 80 GB (A100-80GB, H100, H200).
##SBATCH --partition=YOUR_PARTITION
##SBATCH --constraint=YOUR_80GB_GPU_FEATURE
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source ./env.sh
mkdir -p logs results
JOB="${SLURM_JOB_ID:-local}"
PY="$WORK_DIR/jev27b/.venv/bin/python"
URL="http://127.0.0.1:$JEV_PORT"

run_phase() {
  local tag=$1 caching=$2
  shift 2
  PREFIX_CACHING=$caching setsid bash jev27b/serve.sh > "logs/jev27b-server-$tag-$JOB.log" 2>&1 &
  local pid=$!
  "$PY" bench.py --model jev27b --url "$URL" --tag "$tag" --wait 1800 --server-pid "$pid" "$@" \
    || echo "bench.py ($tag) exited with status $?"
  kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 20  # let the GPU memory free up before the next server starts
}

# With caching: send up to 640 questions per workload (enough to measure throughput) and extrapolate the rest.
run_phase prefix_on 1 --question-cap 640
# Without caching every request recomputes the whole state, so send 64 questions per workload and extrapolate.
run_phase prefix_off 0 --tests smoke,scale --question-cap 64
