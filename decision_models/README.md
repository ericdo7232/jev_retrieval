# Decision model for DecisionRAG: pplx-decider-v1.2 with forked scoring

DecisionRAG asks a yes/no question about every candidate (cluster, page, paragraph) in one shared context. Jev reads the context once and answers every question independently. We can't use Jev yet, so we use [`perplexity-ai/pplx-decider-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) as a stand-in. It is a 27B Qwen3.5 model and the public checkpoint needs no token.

## The problem with the model's normal (joint) mode

The model puts the state and all the questions in one sequence, with one readout per question. Its 16 full-attention layers are noncausal, so every question sees every other question. Answers then depend on which other candidates are in the request: the model card measures −2.9 to −4.8 points when competing candidates are grouped. DecisionRAG merges P(yes) across requests and multiplies page by paragraph scores, so it needs scores that don't depend on the rest of the request.

## Forked scoring (`pplx/scorer.py`)

The state is computed once and cached. Each question then runs as its own branch that attends to the cached state and to itself only, at the positions and with the tokens it has in a request of its own:
- **Gated DeltaNet layers:** each branch starts from the state's final recurrent state and convolution window.
- **Full-attention layers:** each branch attends to the state's cached keys and values plus its own; the two parts are merged by their log-sum-exp.
- **Cost:** the same as joint mode, about 50 tokens per question on top of the state.

**The one approximation:** sent alone, a question is also seen by the state's tokens in the full-attention layers. Here the state is encoded once without seeing any question. The query is in the state, so the state still knows what is being asked; it just doesn't know which candidate each question is about.

Checked on CPU against the unmodified Transformers model, using a tiny random model with the same architecture, in float32:
- forked matches Transformers run with the same attention pattern (max difference 5e-6);
- forked answers don't change when other questions are added, removed or reordered (5e-7);
- joint answers do change (0.23).

The scorer also has `exact` (each question alone, the state recomputed every time), `joint`, and `official` / `official_masked` (the Transformers model, for checking).

**GPUs:** the weights are about 52 GB in bf16, so the 64 layers are split across two A6000s: 31 on the first GPU with the embeddings, 33 on the second. The A6000 doesn't support FP8, so there is no quantization, and answers match the model card's bf16 numbers.

## Running on the cluster (Unicorn, partition `lil-interns`)

From `decision_models/` on the cluster, after `git pull`:

```bash
SETUP=$(sbatch --parsable slurm/setup.sub)                                          # install + download (~52 GB), CPU only
sbatch --dependency=afterok:$SETUP --kill-on-invalid-dep=yes slurm/pplx-smoke.sub   # smoke test on 2 x A6000
squeue -u $USER
```

Logs: `logs/pplx-setup-<job>.out` and `logs/pplx-smoke-<job>.out`. Results go to `results/pplx_smoke_<time>.json`.

## What the smoke test checks (`pplx/smoke.py`)

| Step | Pass if |
|---|---|
| `load` | Model loads and is split across both GPUs (prints weights per GPU) |
| `kernels` | Forked scoring with the GPU kernels (flash-linear-attention, varlen FlashAttention) agrees with the Transformers model run with the same attention pattern: mean \|Δp\| < 0.01, max < 0.05 |
| `smoke` | P(refund) > 0.5 > P(late delivery) on an obvious customer message |
| `needle` | Gold page in the top 5 of 200 titles for at least 5 of the 6 queries |
| `isolation` | Re-asking 29 questions in another subset and order changes P(yes) by less than 0.02 |
| `timing` | Information only: seconds, tokens/s and peak memory on stage-sized workloads, and an estimate per DecisionRAG query |

## If something fails

- **`CUDA available False`** in the log: the PyPI torch build needs a newer NVIDIA driver than the node has. Reinstall with an older CUDA build:
  ```bash
  sbatch --wrap 'TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash pplx/setup.sh' --partition=lil-interns --cpus-per-task=8 --mem=32G --time=01:00:00 --output=logs/pplx-setup-%j.out
  ```
- **`kernels` fails but the other steps pass:** a GPU kernel disagrees with Transformers. Paste the log.
- **"varlen attention returns log-sum-exp: no"** in the setup log: forked scoring still works, but attention over the shared state runs in plain PyTorch (slower).
- **Out of memory:** lower `--budget` (tokens per forward pass, default 16384) in `slurm/pplx-smoke.sub`.

## Files

```
env.sh                  paths (WORK_DIR, PPLX_CKPT)
common.py               workloads (stage-shaped, needle questions)
pplx/scorer.py          forked / exact / joint scoring, layers split across GPUs
pplx/smoke.py           cluster smoke test
pplx/setup.sh           environment + checkpoint download
slurm/setup.sub         runs pplx/setup.sh (CPU job)
slurm/pplx-smoke.sub    smoke test on 2 x A6000
results/, logs/         outputs (results are committed, logs are not)
```

**Earlier attempt, not used:** `kev/`, `jev27b/`, `bench.py`, `clients.py`, `compare.py`, `slurm/kev.sub` and `slurm/jev27b.sub` tested Kev-27B and JEV-27B, and both failed to run. They can be deleted.
