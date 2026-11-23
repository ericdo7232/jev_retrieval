# Paths for decision_rag scripts. Edit for your cluster.

# Large data (corpus, embeddings, trees, Python environment, model cache). About 25 GB.
export DRAG_DIR="${DRAG_DIR:-$HOME/scratch/decision_rag}"

# Hugging Face cache for e5-base-v2 and Qwen3-8B (~17 GB).
export HF_HOME="${HF_HOME:-$DRAG_DIR/hf_cache}"
