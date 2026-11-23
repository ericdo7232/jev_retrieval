# Decision-model tests: Kev-27B and JEV-27B

DecisionRAG scores every candidate at a navigation stage with yes/no questions in one shared context. That is cheap with Jev, because the context is read once. We can't use Jev for now, and GLiDE-27B rereads the whole context for every question, so we are choosing between two open models:

| Model | Why it is a candidate | What could go wrong |
|---|---|---|
| **Kev-27B** (`jaredpalmer/kev-27b`) | Reads the context once and answers any number of isolated yes/no questions on it in one request. Decision Index close to Jev's (59.6 vs 60.2) | Memory: its card reports 78.7 GB peak at a 32k-token context, and our largest stage is about 40k tokens |
| **JEV-27B** (`autotrust/JEV-27B`) | Distilled from Jev 1.13, the version DecisionRAG used; vendor reports KL 0.017 and 90.5% top-choice agreement with Jev | One question per request. Cheap only if vLLM's prefix caching reuses the shared context across requests, which is unproven for this hybrid architecture |

These tests answer the questions that decide between them, on the same GPU and identical inputs.

## What the tests measure

| Test | What it does | What to look for |
|---|---|---|
| `smoke` | Two obvious questions on a short text; prints a raw response | P(yes) high for "refund", low for "late delivery". Confirms the request format works |
| `needle` | 6 real questions; the gold Wikipedia page and 5 related pages are hidden among 200 page titles; every page gets a yes/no question | Gold page ranked first in most of the 6. Also gives both models' full rankings on identical inputs for `compare.py` |
| `position` | Gold page placed at position 1, 625 and 1,250 of 1,250 titles | Similar P(yes) at every position. A big drop at some position means position bias |
| `consistency` | Repeats a 50-question request. Kev: compares one joint request with one-question-at-a-time (`/separate`) and with shuffled question order. JEV-27B: compares concurrent with sequential requests | Differences near 0. Kev's questions should not affect each other; JEV-27B's answers should not depend on batching |
| `scale` | Stage-shaped workloads (table below); time, tokens, peak GPU memory. JEV-27B also times a cold first request against a warm second request on the same context | Kev: time per stage, and whether `stage3_230` fits in memory. JEV-27B: warm request much faster than cold means prefix caching works |

Scale workloads (synthetic text sized like DecisionRAG's stages on Natural Questions):

| Workload | Context | Questions | Paper's equivalent |
|---|---|---|---|
| `stage1_125` | 125 cluster descriptions, ~5k tokens | 125 | One of the 8 cluster requests |
| `stage2_625`, `stage2_1250`, `stage2_2500` | Page titles grouped under clusters, ~5k / ~10k / ~20k tokens | 625 / 1,250 / 2,500 | Page requests of up to 20k tokens |
| `stage3_115`, `stage3_230` | Paragraphs grouped under pages, ~20k / ~40k tokens | 115 / 230 | The one paragraph request (~230 paragraphs) |

The text in `scale` is synthetic, so it tests speed and memory only. `needle` and `position` use real questions and page titles, but they are a sanity check, not a retrieval evaluation.

## Running on the cluster

Requirements: one GPU with at least 80 GB (A100-80GB, H100 or H200), about 120 GB of scratch space, `git`, and [`uv`](https://docs.astral.sh/uv/).

1. **Copy this folder to the cluster** (for example by cloning the repo) and `cd` into `decision_models/`.
2. **Edit `env.sh`.** Set `WORK_DIR` to a scratch directory with about 120 GB free.
3. **Edit the `#SBATCH` lines** in `slurm/kev.sbatch` and `slurm/jev27b.sbatch`: partition, and a constraint that gets an 80 GB GPU. They assume SLURM.
4. **Install and download**, once, on a login node or CPU job (downloads ~51 GB and ~55 GB):
   ```bash
   bash kev/setup.sh
   bash jev27b/setup.sh
   ```
   `jev27b/setup.sh` ends with a check that the installed vLLM has what JEV-27B's server needs. JEV-27B was tested on an unreleased September 2026 vLLM build, so if the check fails, rerun with a nightly build:
   ```bash
   VLLM_SPEC="vllm --pre --extra-index-url https://wheels.vllm.ai/nightly" bash jev27b/setup.sh
   ```
5. **Run the tests:**
   ```bash
   sbatch slurm/kev.sbatch      # ~30 min including model load
   sbatch slurm/jev27b.sbatch   # two server starts (with and without prefix caching), 1-3 h
   ```
   In an interactive GPU session, `bash slurm/kev.sbatch` works too. Each script starts the server, waits for it, runs `bench.py`, and stops the server. Server output goes to `logs/`.
6. **Compare:**
   ```bash
   python3 compare.py results/kev_default_*.json results/jev27b_prefix_on_*.json
   python3 compare.py results/jev27b_prefix_on_*.json results/jev27b_prefix_off_*.json
   ```
7. **Commit and push `results/`** (and the server logs if something failed) so the results can be analyzed.

## Reading the results

**Kev-27B is usable for DecisionRAG if:**
- `smoke` passes and `needle` ranks the gold page first in most queries.
- `consistency` shows near-zero differences, so batching questions doesn't change answers.
- `stage2_2500` takes a few seconds or less. The paper's whole query takes 1.7 s with Jev, with stage 2 split into parallel requests.
- `stage3_230` runs without an out-of-memory error. If it fails, the paragraph stage needs two smaller requests, or a GPU with more memory.

**JEV-27B is usable if:**
- `smoke` and `needle` pass as above.
- With prefix caching on, `warm_over_cold` is well below 1 (for example under 0.2) for the large workloads. With it off, it should be near 1. The `prefix_cache_metrics_delta` hit counters, if vLLM exposes them, confirm the same thing.
- `est_full_s` for `stage2_2500` with caching on is in the same range as Kev's.

**Between the two:**
- If both are usable, `compare.py` shows how far their rankings agree on the `needle` queries.
- JEV-27B is the closer imitation of Jev, so it wins if it is fast enough.
- If its prefix caching doesn't work, Kev is the choice.

## Known uncertainties

- **JEV-27B's vLLM version** is not published. `jev27b/setup.sh` checks compatibility and suggests a nightly build if needed.
- **`--trust-request-chat-template`**: JEV-27B's server script asks for this flag in its docstring, but its README command leaves it out. If the server rejects requests, rerun with `TRUST_REQUEST_CHAT_TEMPLATE=1` set in the environment.
- **`thinking: "off"`** is sent with JEV-27B requests so that no reasoning is generated. If the server rejects the field, set `JEV_THINKING=""`.
- **Kev's fused kernels** need `flash-linear-attention` at the version pinned in Kev's source. `kev/setup.sh` reads and installs it. Without it the server still runs, but slower.
- **GPU memory** comes from `nvidia-smi`. vLLM reserves most of the GPU at startup, so JEV-27B's memory numbers are not informative. Kev's are.

## Files

```
env.sh               paths and ports (edit this)
bench.py             runs the tests against a running server; writes results/<model>_<tag>_<time>.json
compare.py           side-by-side comparison of two result files
clients.py           HTTP clients: Kev /v1/systemone, JEV-27B /v1/decide
common.py            workloads, needle questions, GPU monitor, HTTP helpers
kev/setup.sh         clone Kev, install, download weights
kev/serve.sh         start Kev's server
jev27b/setup.sh      install vLLM, download weights, check compatibility
jev27b/serve.sh      start JEV-27B's server (PREFIX_CACHING=1 or 0)
slurm/*.sbatch       one job per model
results/, logs/      outputs (results are committed, logs are not)
```

`bench.py` and `compare.py` use only the Python standard library.
