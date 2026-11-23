# DecisionRAG on Natural Questions

The plan, and every decision the paper leaves open, are in [IMPLEMENTATION.md](IMPLEMENTATION.md). Settings are in [config.yaml](config.yaml).

## Running on the cluster (Unicorn, partition `lil-interns`)

From `decision_rag/` on the cluster, after `git pull`:

```bash
P=$(sbatch --parsable slurm/prepare.sub)                                          # step 1: CPU
sbatch --dependency=afterok:$P --kill-on-invalid-dep=yes slurm/hierarchy.sub      # step 2: one A6000
squeue -u $USER
```

| Job | What it does | Writes |
|---|---|---|
| `prepare.sub` (CPU, ~20–40 min) | Python environment (`setup.sh`), DAPR NQ download (~0.8 GB), page and paragraph tables, first sentences, the 500-question sample | `$DRAG_DIR/data/nq/`, `splits/nq_test500.txt` (commit it) |
| `hierarchy.sub` (1 × A6000, ~30–60 min) | e5-base-v2 embeddings of 108,626 pages, k-means (k = 1,000) for seeds 0, 1 and 2, Qwen3-8B descriptions of all 3,000 clusters | `$DRAG_DIR/trees/nq_seed{0,1,2}/tree.json` and intermediate files |

`$DRAG_DIR` defaults to `~/scratch/decision_rag` (about 25 GB; see `env.sh`). Logs are in `logs/`.

To rerun part of step 2, pass arguments through, for example `sbatch slurm/hierarchy.sub --steps describe --seeds 0`. Each step reuses the files of the steps before it.

**What to check in the logs:**
- **`prepare`:** the corpus counts match the paper (108,626 pages, 2,682,017 paragraphs), gold is missing for 0 questions, and the example first sentences look right.
- **`hierarchy`:** cluster sizes per seed against the paper's 15–350 (median 106), and the example descriptions.

**If vLLM fails** to install or start, descriptions fall back to transformers automatically (slower). If torch can't see the GPU because the driver is too old, rebuild the environment without vLLM:
```bash
rm -rf ~/scratch/decision_rag/.venv
sbatch --wrap 'DRAG_VLLM=0 TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash setup.sh' --partition=lil-interns --cpus-per-task=8 --mem=16G --time=01:00:00 --output=logs/drag-setup-%j.out
```

## Layout

```
config.yaml                 settings
env.sh                      DRAG_DIR, HF_HOME
setup.sh                    Python environment
src/decision_rag/data.py    download, pages, paragraphs, first sentences, test sample, loaders
src/decision_rag/hierarchy.py  e5 embeddings, k-means, typical titles
src/decision_rag/describe.py   Qwen3-8B descriptions, tree.json
scripts/prepare_data.py     step 1
scripts/build_hierarchy.py  step 2 (--steps embed,kmeans,describe --seeds 0,1,2)
slurm/prepare.sub, slurm/hierarchy.sub
splits/nq_test500.txt       the committed test sample (written by step 1)
```
