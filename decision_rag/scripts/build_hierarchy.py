"""Step 2: e5 page embeddings, k-means trees for every seed, Qwen3-8B cluster descriptions.
One GPU. Run with slurm/hierarchy.sub.

  --steps embed,kmeans,describe   any subset; each step reuses the previous steps' files
  --seeds 0,1,2                   default: config hierarchy.kmeans.seeds
  --backend vllm|transformers     description backend (default: config)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_rag.config import load_config  # noqa: E402
from decision_rag.data import load_pages  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="embed,kmeans,describe")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    cfg = load_config()
    steps = args.steps.split(",")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["hierarchy"]["kmeans"]["seeds"]

    import torch
    print(f"torch {torch.__version__}, CUDA available {torch.cuda.is_available()}", flush=True)
    if "describe" in steps and not torch.cuda.is_available():
        sys.exit("No GPU visible: the description step needs one.")

    pages = load_pages()
    if "embed" in steps or "kmeans" in steps:
        from decision_rag.hierarchy import build_tree, page_embeddings
        x = page_embeddings(cfg, pages)
        if "kmeans" in steps:
            for s in seeds:
                build_tree(cfg, pages, x, s)
    if "describe" in steps:
        from decision_rag.describe import describe_trees
        describe_trees(cfg, seeds, backend=args.backend)
