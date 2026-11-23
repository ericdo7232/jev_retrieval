# Reproducing DecisionRAG on Natural Questions

Implementation plan for rebuilding DecisionRAG ([Wan, Jünger, Du et al., 2026](https://arxiv.org/abs/2610.35868)) on Natural Questions. This is milestone 1 of the [Navigator-Built Hierarchies proposal](../Navigator-Built%20Hierarchies%20Retrieval%20Indexes%20Constructed%20Entirely%20by%20a%20Decision.md). The rebuilt hierarchy and navigator are the baseline the proposal's Jev-built trees are compared against.

## Scope

- **Dataset:** Natural Questions, DAPR version. This is the only benchmark where DecisionRAG builds its own hierarchy. ConditionalQA and MTRAG come later.
- **Decision model:** pplx-decider-v1.2 (27B), self-hosted on two A6000s with forked scoring (see `../decision_models/README.md`), as a stand-in until the Jev API (1.13) can be used. GLiDE 27B, Kev-27B and JEV-27B were considered first and dropped.
- **Cluster descriptions:** Qwen3-8B, self-hosted on our GPU.
- **Authors' code and artifacts:** not used. Everything is rebuilt from the paper, and every unstated detail is decided below and recorded in config.

Because the decision model differs from the paper's, results are not expected to match the published numbers exactly (see [Success criteria](#success-criteria)).

## What the paper specifies

### Corpus and test set

- DAPR NQ corpus: 108,626 Wikipedia pages and 2,682,017 paragraphs.
- Test set: a fixed random sample of 500 of the 3,610 DAPR test questions, which come from NQ dev. Gold evidence is 1–3 paragraphs on one page.
- Some analyses use only the first 100 of the 500.

### Hierarchy (cluster → page → paragraph)

- Each page is represented by its title and first sentence, embedded with **e5-base-v2**.
- **k-means into 1,000 clusters**, holding 15–350 pages each (median 106).
- Each cluster gets a one-sentence description of at most 30 words from **Qwen3-8B**. The description is written from up to 300 of the cluster's page titles, most typical first, and the prompt asks for the main kinds of subjects, including less typical ones (our wording is under Gaps and decisions).

### Navigation

| Stage | Candidates | Scoring | Kept |
|---|---|---|---|
| 1. Clusters | 1,000 cluster descriptions | Binary P(yes), 8 parallel requests | Top 150 (15%) |
| 2. Pages | About 17,500 page titles under the kept clusters | Binary P(yes), requests of at most 20k tokens | Top 30 |
| 3. Paragraphs | About 230: all paragraphs of the top 3 pages, the first 10 of pages ranked 4–10, the first 3 of pages ranked 11–30 | Binary P(yes) in one shared context, multiplied by the page's P(yes) | Ranked list |
| 4. Rerank | Top 5 paragraphs | Multiple choice over the 5; choice probabilities reorder them, the rest keep binary order | Final ranking |

- **Binary judgment:** P(Yes | shared context, candidate i), normalized over Yes and No only. Scores need not sum to 1.
- **Multiple choice:** probabilities over the options, summing to 1 (Jev's limit is 255 options).
- **Prompt context:** the query, then the candidates, numbered and in document order, grouped under their parents. Each stage builds a new context, and earlier probabilities are not shown to the model.
- **Pruning is final:** a dropped branch is never revisited for that query.
- **Question wording (App. B.1):**
  - Pages: "Is the information needed to answer the user's question on Page nn?"
  - Paragraphs: "Does Paragraph nn contain information needed to answer the user's question?"

### Published targets (Jev 1.13, 500 test questions)

| Metric | DecisionRAG | Dense + Qwen3-Reranker-8B |
|---|---|---|
| Hit@1 | 65.4 | 48.0 |
| R@10 | 86.7 | 89.0 |
| nDCG@10 | 76.0 | 69.0 |
| Median latency | 1.73 s | 5.8 s |

Cluster budget ablation (first 100 questions): keeping 15% of clusters gives R@10 86.2, against 84.5 for flat scoring of all 108,626 page titles. Recall levels off above about 10% and drops below 5%; at 1%, R@10 is 63.7.

## Gaps and decisions

These details are not given in the paper. Each is set in `config.yaml` so it can be changed and rerun.

### Data and evaluation

| Gap | Decision | Why |
|---|---|---|
| Which 500 of the 3,610 test questions | Uniform random sample (`random.Random(0).sample` of the sorted IDs), saved in sampled order to `splits/nq_test500.txt` and committed; "the first 100" means its first 100 lines | The paper's sample is not released; a fixed, committed list makes our runs repeatable |
| How R@10 is computed with several gold paragraphs | Fraction of gold paragraphs in the top 10, averaged over questions (DAPR/BEIR convention). Hit@10 (any gold in the top 10) is also reported | The standard convention; reporting both covers the other reading |
| Hit@1 and nDCG@10 | Hit@1: the top paragraph is gold. nDCG@10: binary relevance, computed with `pytrec_eval` | Standard definitions |

### Hierarchy

| Gap | Decision | Why |
|---|---|---|
| How the "first sentence" is extracted | Citation markers like `[1]` removed, then the first sentence (`pysbd`) of the page's first paragraph. If it has fewer than 4 words (captions, fragments), the next of the first 3 paragraphs is tried; capped at 100 words | A robust default for Wikipedia text; about 7% of first paragraphs are under 8 words |
| Input format for e5 | `passage: {title}. {first sentence}`, embeddings L2-normalized | e5 requires the `passage: ` prefix; normalizing makes k-means behave like cosine clustering |
| k-means settings | faiss k-means, k = 1,000, spherical, 25 iterations, 5 restarts, no subsampling | Fast and deterministic. We build seeds 0, 1 and 2 to measure how much the random clustering moves results |
| e5 pooling | Mean pooling over tokens (e5's own), fp16 on GPU, max 512 tokens | e5's documented usage |
| What "most typical" titles means | Titles sorted by cosine similarity to the cluster centroid, highest first; at most 300 | The natural reading of "most typical" for k-means |
| Exact Qwen prompt and generation settings | A prompt written from the paper's description (below), greedy decoding, thinking off. It asks for 25 words so outputs land under the paper's 30. Outputs over 30 words are shortened by the model with a second prompt (below), and only cut if still too long | The paper's prompt is not released; greedy decoding makes the descriptions repeatable. A first draft that said "including less typical ones" made the model echo "typical … less typical" in its output, focus on the first titles, and overrun 30 words in 5–7% of clusters, which were then cut mid-sentence |

Description prompt:

> Below are the titles of Wikipedia pages in one group, most typical first. Write one sentence of at most 25 words naming the kinds of subjects in the group. Cover the whole list, including kinds that only appear further down, not just the first titles. Name kinds of subjects rather than individual titles, and do not use the words "typical" or "less typical". Output only the sentence.
>
> {titles, one per line}

Shortening prompt (only for outputs over 30 words):

> Shorten this description of a group of Wikipedia pages to one complete sentence of at most 25 words, keeping the kinds of subjects it names. Output only the sentence.
>
> {description}

### Navigation

| Gap | Decision | Why |
|---|---|---|
| Full prompts (only per-stage questions are given) | The paper's questions verbatim, inside a short fixed instruction plus the query and numbered candidates. Templates live in `prompts/` | Keeps the published wording where it exists |
| Question for the cluster stage (none given for NQ) | "Is the information needed to answer the user's question in Group nn?" | Adapted from the page-stage question |
| How the 1,000 clusters are split into 8 requests | 125 clusters per request, in cluster-ID order | Even split; the paper gives no order |
| How pages are shown | Title only, grouped under their cluster's description, clusters in rank order, pages by cluster then corpus order; split into requests of at most 20k tokens without breaking up a cluster where possible | The paper says "page titles" and grouping under parents |
| How paragraphs are shown | Full paragraph text, grouped under the page title, pages in rank order and paragraphs in document order | Matches "grouped under parents, in document order" |
| Very long paragraphs | Truncated at 512 tokens | Keeps the shared context bounded; affects few paragraphs |
| Final rerank format | The 5 paragraphs as options, each shown with its page title; context limited to the options | The paper says the context is restricted to the options |
| Ties | Broken by document order | Deterministic |
| Pages with multiple clusters | Not applicable: k-means gives each page exactly one cluster | |

### Specific to GLiDE

The paper's scoring calls are Jev's. Before writing the client, we check the GLiDE model card for the following, and record the answers here:

- How the binary yes/no probability is returned, and whether it is normalized over Yes and No only.
- Whether many candidates can be scored in one shared context.
- The option limit for multiple choice.
- The maximum context length. This decides whether 20k-token page requests fit, or whether pages need smaller requests.
- How its `/v1/systemone` interface (reported as compatible with Jev's) differs from Jev's, if at all.

If GLiDE cannot do something the paper relies on, the workaround is written into this document as a deviation.

## Implementation

### Layout

```
decision_rag/
  IMPLEMENTATION.md      # this file
  config.yaml            # every decision above
  prompts/               # cluster, page, paragraph, rerank and description templates
  src/decision_rag/
    data.py              # load DAPR NQ; pages, paragraphs, queries, gold labels; test sample
    hierarchy.py         # first sentences, e5 embeddings, k-means, typical titles
    describe.py          # Qwen3-8B cluster descriptions via vLLM
    client.py            # decision-model client: binary() and choice(), with an on-disk cache
    navigate.py          # the four stages
    metrics.py           # Hit@1, R@10, Hit@10, nDCG@10, latency
  scripts/               # one entry point per step, e.g. build_tree.py, run_nav.py, evaluate.py
  data/                  # gitignored: corpus, embeddings, trees
  runs/                  # gitignored: per-run outputs, metrics and config snapshots
```

### Decision-model client

- Two calls:
  - `binary(context, candidates, question)` returns P(yes) for each candidate.
  - `choice(context, options)` returns a probability per option.
- Backend chosen by base URL in config: GLiDE now, the Jev API or JEV-27B later, with no other code changes.
- **Cache:** every request and response is stored in SQLite, keyed by a hash of the model, the prompt and the settings. Reruns and analyses then cost no GPU time, and the stored probabilities can be reused for the proposal's offline tuning.
- **Prefix caching:** shared context (instructions, query) goes first in each request, so vLLM can reuse it across the parallel requests of a stage.
- Each call logs its latency and token counts.

### Steps

1. **Project setup.** `uv` project, config, `.gitignore` for `data/` and `runs/`.
2. **Data.** Check file sizes first (the Mac has about 45 GB free), then download DAPR NQ from Hugging Face (`UKPLab/dapr`). Build page-to-paragraph maps, load queries and gold labels, and write the 500-question sample.
3. **Hierarchy.** Extract first sentences and embed them with e5-base-v2 (on the Mac). Run k-means for seeds 0, 1 and 2, and save the clusters and typical titles.
4. **Descriptions.** Serve Qwen3-8B with vLLM on the GPU, generate 1,000 descriptions per tree, and save each tree as JSON.
5. **Navigator and metrics.** Written and tested on the Mac against a stub client that returns fixed scores, so the pipeline is checked before any GPU time is spent.
6. **Serve GLiDE.** vLLM on the GPU; confirm the GLiDE details above.
7. **Runs.**
   - About 20 questions as a sanity check.
   - The first 100 questions.
   - All 500 questions, on the seed-0 tree.
   - The first 100 questions on the seed-1 and seed-2 trees.
   - Cluster budget sweep on the first 100 questions: keep 1%, 5%, 10%, 15% and 25% of clusters, plus flat scoring of all page titles.
8. **Baselines, after navigation works:**
   - BM25.
   - Dense retrieval with Qwen3-Embedding-8B over the 2.68M paragraphs, run on the GPU.
   - Dense retrieval reranked by Qwen3-Reranker-8B.

## Success criteria

The numbers will not match the paper exactly, because of a different decision model (GLiDE instead of Jev 1.13), a different test sample, a different tree and different prompt wording. The reproduction counts as successful if the paper's main patterns hold:

1. Hit@1 is within a few points of 65.4, and above the dense-plus-reranker baseline run on our own sample.
2. Keeping 15% of clusters gives about the same R@10 as flat scoring of all page titles.
3. R@10 drops sharply below 5% of clusters.
4. The gap across the three tree seeds is reported, so it is clear how much of any difference from the paper comes from the tree alone.

A gap from the paper's numbers that remains after this is attributed to the decision model, and stays open until the Jev API can be used.

## Outputs per run

Each run writes to `runs/<timestamp>_<name>/`:

- `config.yaml`: a snapshot of the settings used.
- `rankings.jsonl`: per question, the clusters, pages and paragraphs kept at each stage, with scores.
- `metrics.json`: Hit@1, R@10, Hit@10, nDCG@10, per-stage gold recall (was the gold cluster or page kept?), median and p95 latency, and model calls and tokens per query.

Per-stage gold recall shows where routing fails. It is also the main diagnostic for the proposal's later work.

## Open items

- [ ] Read the GLiDE model card and fill in [Specific to GLiDE](#specific-to-glide).
- [ ] Confirm the DAPR NQ download size before downloading.
- [ ] Set up vLLM on the GPU for Qwen3-8B and GLiDE 27B.
- [ ] Check GLiDE's license terms for research use (the proposal says it is free below $10M annual revenue).
