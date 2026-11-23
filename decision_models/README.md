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

**GPUs:** in bf16 the weights are about 52 GB, so the 64 layers are split across two A6000s: 31 on the first GPU with the embeddings, 33 on the second; answers match the model card's bf16 numbers. With `--fp8` (RTX 6000 Ada only, see below) the model fits on one GPU.

## Making it faster

The bf16 run on 2 × A6000 took about 15 minutes per DecisionRAG question (~1.3M tokens through a 27B model), about a week for 500 questions. pplx is a stand-in until Jev API access, and final numbers (and latency) will come from the Jev API, so we trade a little fidelity for speed, and measure what each shortcut costs.

| Speed-up | Gain | Fidelity cost | Where | Status |
|---|---|---|---|---|
| FP8 on one RTX 6000 Ada | FP8 tensor cores; one GPU per model copy instead of two | FP8 rounding; `reference` step compares with bf16 | `pplx/fp8.py`, `--fp8` | Built, awaiting the Ada smoke test |
| Fast mode: shorter question branches | 1.84x fewer tokens per question (1.31M → 0.71M) | Full-attention layers only; `fast` step compares with forked | `score(..., mode="fast")` | Built, awaiting the Ada smoke test |
| MLP in chunks of 8,192 tokens | Lower peak memory on long states (needed to fit one 48 GB GPU) | None (per-token operation) | `Scorer(mlp_chunk=8192)` | Built |
| Timing without compilation | A fair estimate: each workload runs twice, the second is used (the first call of a new shape spent ~20 s compiling kernels) | None | `pplx/smoke.py` | Built |
| Keep 10% of clusters instead of 15% | Stage 2 about ⅓ smaller (~0.5M tokens per question in total) | The paper's ablation shows recall levels off above 10% | DecisionRAG navigator | Planned |
| Questions split across GPU jobs, resuming after preemption | Linear in the number of GPUs | None | DecisionRAG navigator | Planned |

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
| `load` | Model loads (prints precision, layers and weights per GPU) |
| `kernels` | Forked scoring with the GPU kernels (flash-linear-attention, varlen FlashAttention) agrees with the Transformers model run with the same attention pattern: mean \|Δp\| < 0.01, max < 0.05 (FP8: < 0.02 and < 0.1, since FP8 turns tiny round-off differences into occasional rounding flips) |
| `smoke` | P(refund) > 0.5 > P(late delivery) on an obvious customer message |
| `needle` | Gold page in the top 5 of 200 titles for at least 5 of the 6 queries |
| `isolation` | Re-asking 29 questions in another subset and order changes P(yes) by less than 0.02 |
| `fast` | Fast mode against forked on the 6 needle workloads (1,200 questions): gold page still in the top 5 for at least 5 of 6, mean \|Δp\| < 0.05; also prints the token counts |
| `timing` | Information only: seconds, tokens/s and peak memory on stage-sized workloads, for forked and fast (each run twice; the second, without kernel compilation, is used), and an estimate per DecisionRAG query |
| `reference` | Only with `--reference old.json`: answers on the same workloads agree with an earlier run (e.g. FP8 against bf16): mean \|Δp\| < 0.02, max < 0.15, needle gold ranks unchanged |

## FP8 on one RTX 6000 Ada (`--fp8`, `pplx/fp8.py`)

The 2 × A6000 bf16 run measured about 1,000–1,400 tokens/s, roughly 15 minutes per DecisionRAG question. The A6000 (Ampere) has no FP8 tensor cores; the RTX 6000 Ada does. With `--fp8`:

- The large projections (attention, Gated DeltaNet in/out, MLP) run as FP8 matrix multiplies: weights quantized at load with one scale per output channel, activations at run time with one scale per token. Everything else (norms, convolutions, the recurrent and attention kernels, embeddings, readout) stays bf16.
- Scales are per token, not per tensor, so a question's answer does not depend on the other questions packed with it (a per-tensor scale would break forked scoring's isolation).
- The model is about 27 GB and fits on one 48 GB card, so each GPU runs its own copy on its own share of the questions instead of two GPUs sharing one copy.

```bash
sbatch slurm/pplx-smoke-ada.sub --reference results/pplx_smoke_<bf16 run>.json
```

## Fast mode (`score(..., mode="fast")`)

pplx adds a 48-token template to every yes/no question, and only the page number differs between candidates: about 85% of the tokens in stage 2. Fast mode is forked scoring with two shortcuts, cutting a branch from ~51 to ~19 tokens and a DecisionRAG question from ~1.31M to ~0.71M tokens (1.84x):

- The question text up to the first digit ("Question 0: Is the information … on Page ") is computed once as part of the shared state.
- Branches end at the readout marker; the 9 chat-template tokens after it are dropped.

Both shortcuts only change the 16 full-attention layers (shared tokens don't see the branch; the marker doesn't see the dropped tokens). The shared part depends only on the question template, so answers still don't depend on the other questions. The smoke test's `fast` step measures the agreement with forked on 1,200 needle questions. Final numbers come from the Jev API; this is for the pplx proof of concept.

`default_partition` is preemptible; resubmit if the job is preempted. CPU tests of the FP8 layer and the scorer paths: `PPLX_TEST_CKPT=<dir with the checkpoint's small files> python -m pytest -q tests`.

## If something fails

- **`CUDA available False`** in the log: the PyPI torch build needs a newer NVIDIA driver than the node has. Reinstall with an older CUDA build:
  ```bash
  sbatch --wrap 'TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash pplx/setup.sh' --partition=lil-interns --cpus-per-task=8 --mem=32G --time=01:00:00 --output=logs/pplx-setup-%j.out
  ```
- **`kernels` fails but the other steps pass:** a GPU kernel disagrees with Transformers. Paste the log.
- **"varlen attention returns log-sum-exp: no"** in the setup log: forked scoring still works, but attention over the shared state runs in plain PyTorch (slower).
- **Out of memory:** lower `--budget` (tokens per forward pass, default 16384), passed after the script, e.g. `sbatch slurm/pplx-smoke-ada.sub --budget 8192 ...`.
- **`load` says `FP8 rowwise_emulated`:** this torch build has no rowwise FP8 kernel for the GPU; the same numbers are computed with an extra pass (slightly slower).

## Files

```
env.sh                  paths (WORK_DIR, PPLX_CKPT)
common.py               workloads (stage-shaped, needle questions)
pplx/scorer.py          forked / fast / exact / joint scoring, layers split across GPUs, optional FP8
pplx/fp8.py             FP8 linear layers (per-channel weights, per-token activations)
pplx/smoke.py           cluster smoke test
tests/test_fp8.py       CPU tests: FP8 layer, FP8, fast-mode and chunked-MLP scorer paths
pplx/setup.sh           environment + checkpoint download
slurm/setup.sub         runs pplx/setup.sh (CPU job)
slurm/pplx-smoke.sub    smoke test on 2 x A6000 (bf16)
slurm/pplx-smoke-ada.sub  smoke test on 1 x RTX 6000 Ada (FP8), compared with a bf16 run
results/, logs/         outputs (results are committed, logs are not)
```

**Earlier attempt, not used:** `kev/`, `jev27b/`, `bench.py`, `clients.py`, `compare.py`, `slurm/kev.sub` and `slurm/jev27b.sub` tested Kev-27B and JEV-27B, and both failed to run. They can be deleted.
