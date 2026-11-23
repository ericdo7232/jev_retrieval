# Settings shared by the setup, serve and sbatch scripts. Edit for your cluster.

# Where code, virtual environments and model weights go. Needs about 120 GB free
# (Kev-27B ~51 GB, JEV-27B ~55 GB, plus Python environments). Use scratch, not your home directory.
export WORK_DIR="${WORK_DIR:-$HOME/scratch/decision_models}"

# Hugging Face cache (Kev-27B weights are downloaded here).
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"

# pplx-decider-v1.2 checkpoint (~52 GB; pplx/setup.sh downloads it here).
export PPLX_CKPT="${PPLX_CKPT:-$WORK_DIR/pplx/pplx-decider-v1.2}"

export KEV_PORT="${KEV_PORT:-8008}"
export JEV_PORT="${JEV_PORT:-8000}"

# Maximum context for JEV-27B. Stage-3 workloads are ~40k tokens, so keep this above 45k.
export JEV_MAX_LEN="${JEV_MAX_LEN:-65536}"
