# Navigator-Built Hierarchies: Retrieval Indexes Constructed Entirely by a Decision Model

Nov 18, 2026 · @eric

## Summary

We propose a retrieval index whose structure is decided by the decision model that navigates it (Jev). The central claim: a hierarchy whose membership Jev decides, and which is tuned on Jev's own routing errors, is easier for Jev to route through than one built by embeddings.

Recent work shows decision models can retrieve by walking a corpus hierarchy in one to two seconds per query, beating dense retrieval with a reranker. But where no hierarchy exists, as on Wikipedia, that work builds one with embedding k-means and one-shot LLM descriptions, and names uninformative descriptions as a limitation.

The method has two parts. **Hierarchy initialization:** categories are proposed either by Jev-only random pivots and merging, or by a small LLM; in both cases Jev decides which documents belong where, allowing documents in several categories. **Post-initialization tuning:** a per-node routing loss on training queries locates failures and drives group-level edits (regraft, relabel, merge), handing off to construction when a fix needs a new category. We evaluate on Natural Questions (2.68M paragraphs) against the DecisionRAG hierarchy, holding navigation fixed. Development uses two open-weight decision models, GLiDE 27B and AutoTrust's Jev-distilled JEV-27B; reported results use the Jev API.

## Background and related work

LLM-guided hierarchical search replaces vector search with a tree of natural-language nodes that a model navigates. Existing systems differ in three ways: the navigator, where the hierarchy comes from, and whether the tree is tuned to the navigator. No prior system tunes the tree to its navigator.

| System | Navigator | Hierarchy source | Tuned to navigator |
| --- | --- | --- | --- |
| [DecisionRAG](https://arxiv.org/abs/2610.35868) (2026) | Decision model (Jev) | Existing structure; e5 k-means + Qwen labels on Wikipedia | No |
| [LATTICE](https://arxiv.org/abs/2510.13217) (2025–26) | Generative LLM | LLM-built top-down; embedding first pass at NQ scale | No |
| RAPTOR (2024) | Embeddings over summaries | Embedding clustering + LLM summaries | No |
| PageIndex, MemWalker | Generative LLM | Document table of contents | No |
| This proposal | Decision model (Jev) | Categories from Jev pivots or a small LLM; membership decided by Jev | Yes |

**DecisionRAG** is the closest work. Jev scores all children of a node in one shared context, keeps the top k, and descends. On Natural Questions it beats dense retrieval with an 8B reranker by 17.4 Hit@1 points at 1.7 seconds per query. Its Wikipedia hierarchy is 1,000 k-means clusters over e5 embeddings of titles and first sentences, each described by one Qwen3-8B sentence written from page titles alone. Keeping 15% of clusters matches flat scoring; below 5%, recall falls sharply. The authors list uninformative node descriptions as a limitation.

**LATTICE** builds trees top-down with an LLM and beats embedding-built trees by 2 to 12 nDCG@10 points. It falls back to an embedding first pass at Natural Questions scale, and its generative navigation takes tens of seconds per query.

**Description optimization** for routing exists outside retrieval: iterative refinement of class definitions from misclassified examples, automated tuning of agent skill descriptions, and routing-stability checks during hierarchy construction (CRAFT).

**Oracle clustering** gives theory for clustering with a noisy "same cluster?" oracle (Ashtiani et al. 2016; Mazumdar and Saha 2017), and the pivot algorithm for correlation clustering (Ailon et al. 2008).

## Problem and research questions

In current decision-model retrieval, the tree is built by one model (an embedder) and walked by another (Jev). Clusters that are close in embedding space need not be ones Jev can tell apart, and descriptions written once, without seeing sibling clusters, may not separate them. Routing errors at the top level are unrecoverable, so systems compensate by keeping many clusters per query, which raises cost.

We ask:

1. **Structure.** Does a hierarchy whose membership Jev decides route queries more accurately than an embedding-built hierarchy, under identical navigation?
2. **Initialization.** Do categories proposed by Jev-only pivots and merging, or by a small LLM, give better routing? Can exemplar labels, with no generated text, serve as navigable descriptions?
3. **Membership.** Is it better for Jev to place documents directly, or to place them where questions about them route? Does allowing multiple memberships improve recall?
4. **Tuning.** How much does loss-based tuning improve routing over the initial hierarchy, and at which levels?
5. **Efficiency and scale.** Does better routing let Jev keep fewer nodes per level at the same recall, and can the hierarchy be built over all of Wikipedia without an embedding fallback?

## Part 1: Hierarchy initialization

One principle governs initialization: whatever proposes categories, Jev decides document membership. The tree is built top-down, one node at a time, so every Jev question stays within its limits and topics absorbed at one level can separate at the next.

### Parameters

| Symbol | Meaning | Initial value |
| --- | --- | --- |
| r | Random pivots per node (option A) | 200 |
| b | Target children per node | 50 |
| L | Maximum documents in a leaf | 50 |
| e | Exemplars per label (option A) | 5 |
| δ | Log-probability gap for extra memberships (per-document rule) | log 2 |
| α | Extra memberships as a fraction of documents (global budget) | 0.2 |
| m | Maximum memberships per document | 3 |
| n\_q | Questions per document (membership mode 2) | 3 |

In Jev contexts, each document appears as its title plus first sentence.

### Proposing categories at a node

**Option A: Jev-only pivots and merging.** No embeddings and no generated text.

1. **Seed.** Pick r random documents from the node as pivots.
2. **Assign.** Each document picks its most similar pivot in one multiple-choice question.
3. **Represent.** For each cluster, Jev picks the member that best represents the group (sampled to 255 members if larger). The top choice becomes the pivot; the top e become the label.
4. **Merge.** With all clusters listed by exemplars in one context, each picks its most similar other cluster, or none. Mutual pairs merge and are re-represented. Repeat until about b clusters remain.
5. **Clean up.** Fold very small clusters into their nearest neighbor.

The resulting pivots and exemplar labels become the node's categories. This is the original pivot-and-merge idea, and on its own, recursed down the tree, it is a complete method needing only Jev; it is the first version to build. Seeding can optionally use a k-means++ analogue: sample each new pivot in proportion to Jev's "none of these" probability against current pivots.

**Option B: small LLM categories.** A small LLM reads the titles of a sample of the node's documents and proposes up to b named categories, each with a one-line description, plus an "Other" category. If "Other" receives many documents, the LLM proposes further categories from those documents only. Labels are the category names and descriptions.

### Deciding membership

Categories from either option are then filled by Jev. Two membership modes are compared:

- **Mode 1, routing documents.** One multiple-choice call per document over the node's categories. Jev's probability distribution over categories is then turned into memberships by one of the overlap rules below.
- **Mode 2, routing questions about documents.** A small LLM writes n\_q questions each document answers. Each question is routed among the node's categories exactly as a query would be, and the document joins the categories its questions reach. This costs more but places documents where queries will look.

A single-membership variant (top category only) serves as an ablation for both modes.

### Overlap rules

Every document first joins its top category. The two options differ in who decides on extra memberships, and by comparing against what. Both treat −log p as a distance, which ties them to overlapping k-means methods.

**Option 1: per-document dropoff.** Each document decides from its own probabilities only. Sorting them as p₁ ≥ p₂ ≥ …, it also joins category j while log p₁ − log p\_j ≤ δ (δ = log 2 means at least half as likely as the top), up to m. Decisions are independent across documents, and each comparison stays within one Jev call. Total overlap is not controlled: it depends on how many documents happen to be ambiguous. This is the relative distance-threshold heuristic from soft clustering, and walking down sorted categories mirrors overlapping k-means (OKM).

**Option 2: global budget.** The total number of extra memberships for the node is fixed at αn, and documents compete for them. All documents' runner-up (document, category) pairs are pooled, ranked together, and the best αn become extra memberships, up to m per document. This follows NEO-K-Means with β = 0, since every document must stay retrievable. It gives exact control over overlap, leaf size and query cost, but ranks probabilities from different Jev calls against each other, assuming they are on the same scale. A middle variant ranks by each document's ratio p\_j / p₁ instead of raw probability, keeping comparisons within each call while still fixing the total.

In both options, a near-flat distribution means a document fits nothing well, not that it fits everything. Such documents go to "Other" (option B) or are flagged for re-proposed categories rather than given extra memberships. Because probabilities are stored, δ and α are tuned offline against routing recall and average memberships per document.

Paired with re-picking pivots from each cluster's overlapping members (option A), and repeated for two or three rounds, option 1 is an OKMED-style method and option 2 a NEO-K-Means-style method, with Jev in place of distances. The convergence guarantees of those algorithms do not carry over, since Jev's probabilities depend on the whole pivot set; the objective J = Σ −log p over all memberships is tracked each round, and the best round is kept.

### Recursion and stopping

Start with all documents at the root and split any node with more than L documents, recursing into each child. Stop early and make a node a leaf if Jev cannot divide it: low-confidence membership, or merging that collapses nearly everything into one category. For Wikipedia, leaves contain pages and each page keeps its paragraphs below it. With b = 50, 108k pages need about two levels of splitting.

## Part 2: Post-initialization tuning

Tuning treats the initial hierarchy like a decision tree to be pruned and corrected, guided by a routing loss on training queries. This part is still being designed; the loss and the edit set below are the current plan.

### Routing loss

For a training query q with gold path root = v₀ → … → v\_D (the gold leaf), Jev gives each child c of node v a probability p(c | q, v), from multiple-choice mode or normalized yes-probabilities. The path loss is the negative log probability of the gold path, as in hierarchical softmax:

```latex
\mathcal{L}(q) = -\sum_{d=0}^{D-1} \log p(v_{d+1} \mid q, v_d)
```

It is computed with teacher forcing (every node on the gold path is evaluated, even past an earlier wrong turn), so it splits into a loss per node, summed over the queries passing through it. Because navigation keeps the top k children, a beam variant penalizes the gold child falling below the k-th best sibling by less than a margin γ. Errors can also be weighted by the depth of the first wrong turn, so early mistakes cost most.

### Division of labor with construction

Construction determines the tree's shape; tuning only corrects it. Tuning never splits a node to go deeper, since depth and granularity are set by L and the recursion in Part 1, and letting training queries reshape them would overfit.

| Phase | Allowed to |
| --- | --- |
| Construction (Part 1) | Create categories, split, set depth, assign memberships |
| Tuning (Part 2) | Move or regraft groups between existing nodes, relabel, merge |
| Handoff to construction | When a fix needs a new category or a new level, rebuild that node's split with Part 1 |

### Edits

Edits act on groups, and each needs support across many queries. Most Natural Questions gold documents have only one or two training questions, so an edit justified by a single query fits that query, not the corpus.

**Prune and regraft a subset.** At node v, take the queries whose gold child is A but which route elsewhere; their gold documents form a subset S inside A.

1. **Coherence check.** Jev picks a pivot from S, then assigns A's documents between that pivot and A's current pivot. If S does not separate confidently from the rest of A, the failures are scattered and no edit is made.
2. **Destination.** If S's queries concentrate on one sibling B, move S under B, or add B as an extra membership drawing on the overlap budget. If they scatter across many siblings, no existing category covers S: hand off to construction and rebuild v's split with S's pivot among the seeds.

**Single-document move.** Allowed only with evidence independent of the training query: Jev places the document itself in the new node when asked about the document directly. This corrects placement mistakes without fitting individual queries.

**Relabel.** Swap a node's pivot or exemplars (or revise a category name in option B) when its documents are well placed but its label fails to attract the questions they answer.

**Merge siblings.** For siblings A and B, average the routing distributions of training queries whose gold child is A, giving p̄\_A, and likewise p̄\_B. If the two are close (small Jensen-Shannon divergence), Jev cannot tell A-questions from B-questions. Merge if Jev also cannot separate A's and B's documents. If the documents do separate, the right fix is grouping A and B under a new shared parent, which adds a level and so is handed off to construction. This mirrors label trees in large-scale classification, which group classes a classifier confuses into the same subtree.

After regrafts and merges, a leaf may exceed L documents; the construction rule then splits it again.

| Edit | Trigger | Overfitting guard |
| --- | --- | --- |
| Prune and regraft a subset | Failing queries' gold documents form a group inside a child | Coherence check; minimum number of supporting queries |
| Single-document move | Jev places the document elsewhere when asked directly | Evidence independent of the training query |
| Relabel | Well-placed documents, but their questions route elsewhere | Held-out confirmation |
| Merge siblings | p̄\_A ≈ p̄\_B and documents inseparable | Divergence measured over many queries |
| Rebuild (handoff) | Fix needs a new category or a new level | Uses Part 1's own rules |

### Search and acceptance

Nodes are examined top-down, level by level: an edit at a parent changes the routing distributions below it, so fixing children first can be wasted work. Within a level, the highest-loss node comes first. Each candidate edit is scored on only the queries passing through the node, with a cost term as in cost-complexity pruning:

```latex
\text{score} = \Delta\mathcal{L} + \alpha \cdot \Delta(\text{query cost})
```

The best edit is applied if it beats a minimum improvement and holds on held-out queries; otherwise the node is marked done. Sweeping α yields a family of trees along the recall-vs-cost curve. Tuning stops when no node has an acceptable edit or a call budget runs out. Still open: whether to batch edits at unrelated nodes, the exact thresholds, and the minimum number of queries per node needed to trust an edit.

### Tuning during construction

The same loss can validate each level as it is built: route training queries through a new split before recursing, and retry the split if it routes poorly. This catches bad top-level splits, the most costly ones, early. Deep nodes see few training queries, so loss there is noisy; options are synthetic queries from the small LLM or restricting tuning to the top levels.

### Hyperparameters

The membership threshold τ and navigation budgets k are tuned offline from stored Jev probabilities and query runs, with no rebuilds. Only r, b and L need rebuilds; these are searched with multi-fidelity methods (successive halving or Hyperband, using corpus size as fidelity) and Bayesian optimization, on held-out training questions.

## Navigation and implementation

### Navigation

Navigation follows DecisionRAG unchanged. At each level, Jev answers a yes/no relevance question for every child in one shared context, keeps the top k, and descends. Paragraphs of the top pages are scored together, and a final multiple-choice question reranks the top candidates. Adaptations for this tree: a budget k per level, question wording suited to exemplar or named labels, path scores multiplied down the tree (tested against page-and-paragraph only), and page-level deduplication for documents with several memberships.

### Models

Three decision models are used: two open-weight models for development, **GLiDE 27B** (Fastino) and **JEV-27B** (AutoTrust), and the **Jev API** for the DecisionRAG reproduction and all reported results.

- **GLiDE 27B** is the strongest well-tested open model. It is second on the community Decision Index (73.37, tied with first) and fourth on Artificial Analysis v2 (73), against 60.24 and 66 for Jev 1.14. It serves Jev's /v1/systemone interface, so the same client code runs on it and on Jev by changing the base URL. Its license requires a commercial agreement only above $10M annual revenue, which research use does not reach.
- **JEV-27B** is the open model most likely to behave like Jev. It is a Qwen3.8-27B student whose decision behavior is distilled from Jev 1.13, the version DecisionRAG used, and AutoTrust reports it as the closest open model to Jev 1.13 at the level of output distributions (a vendor claim, not yet independently verified). It is Apache-2.0 and served on vLLM through its own /v1/decide endpoint, so it needs a thin adapter to the Jev client. Because it is trained on Jev's outputs, TypeSafe's terms on such models are checked before publication.
- **Jev** is pinned to version 1.13, as in DecisionRAG, for the reproduction and final comparison, so baseline numbers match the published ones.

The two open models represent two different bets: GLiDE is more accurate on general benchmarks, while JEV-27B should make the same mistakes Jev makes, which matters because design choices must transfer to Jev. Neither has been measured on retrieval relevance, and Jev may be stronger on knowledge and calibration. Before building, about 200 real calls from the pipeline (construction and navigation questions on Natural Questions) go to all three models, measuring agreement with Jev on top choices, KL divergence from Jev's probability distributions, and Hit@1. Development proceeds on whichever open model agrees with Jev more, preferring GLiDE if they are close; the other is kept as a cross-check on key pilot results, and Kev 27B is a further fallback. Agreement figures are reported, since they show how far development results can be trusted to predict Jev's.

Self-hosted throughput, not per-request latency, limits development: latencies are similar (about 112 ms for GLiDE 27B against 92 ms for Jev on benchmark inputs; JEV-27B's latency is unpublished and is measured in the agreement test), but one GPU cannot match Jev's parallel serving. Shared context such as pivot lists is placed first in each request to benefit from prefix caching, and query latency is reported on the Jev API.

### Cost

Initialization with membership mode 1 costs about one Jev call per document per level: roughly 217k calls for 108k pages over two levels. Mode 2 costs about n\_q calls per document per level, plus question generation. Option A adds a few hundred calls per node for representing and merging; option B adds one small-LLM call per node. Tuning costs one routing pass per training query per evaluated edit, limited to queries through the edited node. All calls at a level are independent and run in parallel.

## Experimental plan

The main experiment swaps DecisionRAG's Natural Questions hierarchy for ours and changes nothing else. Navigation (Jev, DecisionRAG-style) is held fixed in every tree-based row, so differences come from the tree.

### Datasets

- **Natural Questions, DAPR version (primary):** 108,626 Wikipedia pages, 2.68M paragraphs, DecisionRAG's fixed 500-question test sample. The only benchmark where DecisionRAG builds its own hierarchy, and the one that carries the main claim.
- **BRIGHT (secondary, optional):** dropped; transfer is tested on ConditionalQA and MTRAG instead.
- **ConditionalQA (transfer):** 652 UK government guides, 58,483 paragraphs, 271 labeled development questions. DecisionRAG uses the guides' heading structure.
- **MTRAG, IBM Cloud subset (transfer):** 8,578 documentation pages in 172 URL-derived products, 72,442 passages, 188 labeled conversational turns.

### Data splits and transfer protocol

Natural Questions is used for all design and tuning; ConditionalQA and MTRAG test whether the result transfers with every design choice frozen.

**Natural Questions splits.** NQ training questions whose gold page lies in the 108k-page corpus are randomly divided into tuning queries (used by Part 2 to compute loss and choose edits) and development queries (used to choose hyperparameters and design options). DecisionRAG's 500 test questions are run once, after the configuration is frozen. The tree is built over the same corpus throughout; the test is whether new questions route well through a tree shaped by other questions.

**Transfer.** Hyperparameters and design choices are frozen from Natural Questions. Two variants are reported on each transfer dataset:

1. **Construction only.** Part 1 with Natural Questions' settings and no tuning; only the target's documents are used.
2. **Construction plus in-domain tuning.** Same frozen settings, with Part 2 tuning on the target's own questions: ConditionalQA training questions, and for MTRAG, which has no training split, DecisionRAG's design set of synthetic conversations and MTRAG-UN turns.

Each is compared with the dataset's human-made hierarchy, an embedding k-means tree and our tree, with DecisionRAG's navigation configuration reused unchanged (including its conversational notepad on MTRAG).

**Expectations.** On MTRAG, b = 50 gives about 170 top-level clusters, close to the 172 URL-derived products, a direct test of whether Jev-built categories match a human taxonomy. ConditionalQA is small (about 13 clusters at b = 50, where DecisionRAG scores all pages in one request), so it mainly checks that the method does no harm. With 271 and 188 labeled questions, confidence intervals will be wide; paired bootstrap intervals are reported, and Natural Questions carries the main claim.

### Baselines

| Row | Tree | Question it answers |
| --- | --- | --- |
| DecisionRAG tree | e5 k-means, Qwen sentence labels | Main baseline |
| Full method | Best initialization + tuning | Proposed system |
| Random tree | Random membership, same shape | Sanity check: does structure matter at all? |
| LATTICE-style tree (optional) | LLM-built, corpus subset | Does Jev-decided membership match LLM construction? |
| Oracle routing | Forced down the gold path | Ceiling: routing vs. ranking errors |

Reference points without a tree: flat Jev scoring of all 108,626 page titles, BM25, dense retrieval (Qwen3-Embedding-8B), dense with Qwen3-Reranker-8B, and dense with Jev reranking. Rerun at least dense retrieval on our exact question sample.

### Ablations

Each factor is varied with the others fixed at their best setting.

| Factor | Options |
| --- | --- |
| Category initialization | Jev pivots + merge vs. small LLM categories |
| Pivot seeding (option A) | Random vs. k-means++ analogue vs. embedding seeds |
| Membership decided by | Routing documents vs. routing questions about documents |
| Overlap | Single membership vs. per-document dropoff vs. global budget |
| Labels | Exemplars vs. LLM category names, on the same tree |
| Tuning | None vs. during construction vs. post hoc vs. both |
| Judge | GLiDE 27B vs. JEV-27B vs. Jev API |

The labels row separates structure quality from label quality, the main uncertainty in option A.

### Metrics

- **Retrieval:** Recall@10 and Hit@1, plotted against the fraction of nodes kept at each level. The curve is the main result, not a single operating point.
- **Query cost:** Jev calls, API cost and median latency per query.
- **Construction cost:** total Jev calls, cost and wall-clock time to build the tree.
- **Answer quality:** exact match from a fixed generator (Qwen3.5-27B, as in DecisionRAG) given each system's top five passages.
- **Diagnostics:** routing accuracy and routing loss per level against the gold path (before and after tuning), memberships per document, and cluster size distribution.

## Hypotheses and expected results

The headline hypothesis is that our recall-vs-budget curve sits above DecisionRAG's, with the largest gap at small budgets (1–5% of nodes kept), where their routing breaks down.

- **H1 (structure).** Jev-decided membership beats the embedding tree under the same labels. LATTICE's finding that LLM-built trees beat embedding trees by 2 to 12 points suggests the direction; the size is unknown.
- **H2 (initialization and labels).** Small-LLM categories route queries better than exemplar labels, because named categories resemble questions more than lists of titles do. Jev-only pivots remain competitive on structure.
- **H3 (membership).** Routing questions about documents beats routing documents directly, and multiple memberships raise recall at small cost.
- **H4 (tuning).** Loss-based tuning improves routing most at the top levels, and its gains carry over to held-out queries.
- **H5 (efficiency and scale).** Matching DecisionRAG's recall requires keeping substantially fewer nodes, and construction over all 108k pages needs no embedding step.

Useful negative results are also possible. If H1 holds but exemplar labels fail, the method still contributes a Jev-decided structure, with option B's labels as the fix. If tuning gains vanish on held-out queries, that bounds what routing-based optimization can do.

## Risks and mitigations

The biggest risks are that labels route queries poorly and that results on GLiDE do not transfer to Jev. The pilot tests both first.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Exemplar labels route queries poorly | Weak recall despite good clusters | Labels ablation isolates it; option B named categories; question-based membership |
| GLiDE behaves differently from Jev | Design choices tuned on GLiDE fail with Jev | Agreement test before building; repeat the pilot comparison with the Jev API early; JEV-27B as a second, Jev-distilled development model |
| LLM categories miss the corpus's real topics | Large "Other" category, poor coverage | Propose from sampled documents; re-propose from "Other" |
| Multiple memberships bloat leaves | Larger leaves, higher query cost | Cap at m memberships; deduplicate at retrieval |
| Top-level misplacement | Documents unreachable by correctly routed queries | Multiple memberships; level-wise validation during construction |
| Too few training queries at deep nodes | Noisy loss, unreliable edits | Synthetic queries; restrict tuning to top levels |
| Tuning overfits training queries | Gains vanish on test | Held-out confirmation for every edit; minimum-improvement threshold |
| Position bias in multiple choice | Early-listed options attract more | Shuffle option order per question |
| Concurrent work | Reduced novelty | DecisionRAG is weeks old; monitor its authors' follow-ups |

## Extensions

Two follow-ons build on the method once it works. Routing-based optimization, an extension in the first draft, is now part of the core method (Part 2).

**Question exemplars as labels.** A node's label could include a few training questions that route correctly to it, alongside or instead of document exemplars. This targets the gap between how documents and queries look, and choosing which questions to show is still a selection problem Jev can do.

**Distilling Jev into an embedder.** Collect Jev's membership decisions on a sample, fine-tune an embedding model so its clusters match them, and cluster the full corpus at embedding cost, sending only boundary documents back to Jev. This is the most plausible path to billions of documents. ClusterLLM (Zhang et al. 2023) distills generative-LLM judgments into an embedder for clustering, but not for navigation.

## Milestones

The work runs in five phases, each with an exit check before the next begins. Dates are open until scope and compute are set.

1. **Reproduce DecisionRAG on Natural Questions.** Rebuild its e5 k-means hierarchy and navigation. Exit: Hit@1 and Recall@10 within a few points of the published 65.4 and 86.7.
2. **Pilot initialization on 10k pages (GLiDE 27B).** Build trees with options A and B and both membership modes; read labels for coherence; route held-out training questions. Exit: one initialization beats the embedding tree on routing, or the failure is traced to structure or labels.
3. **Jev API check.** Repeat the pilot's main comparison with the Jev API. Exit: the ranking of methods matches the GLiDE results, or the plan is adjusted.
4. **Tuning on the pilot.** Implement the routing loss, diagnostics and edits; measure gains on held-out questions. Exit: tuning helps held-out routing, or the negative result is documented.
5. **Full build, experiments and write-up.** Construct over all 108k pages with the Jev API; run baselines, ablations and recall-vs-budget curves; transfer to ConditionalQA and MTRAG; optional LATTICE comparison; paper draft.

## To-dos

### Research

- [x] Overlapping (multi-membership) clustering: fuzzy c-means, overlapping k-means (OKM), NEO-K-Means. How they choose the number of memberships per item and handle items that fit nowhere.
- [ ] Check the Jev API: whether several multiple-choice questions can share one context, batching several documents per call, option limits, pricing and rate limits.
- [ ] Run the \~200-call agreement test of GLiDE 27B and JEV-27B against the Jev API (agreement, KL divergence, Hit@1, latency); check TypeSafe's terms on JEV-27B.
- [ ] Read up on loss design: hierarchical softmax and tree models trained under beam search.
- [ ] Verify the references cited from memory and add links.
- [ ] Watch for follow-ups from the DecisionRAG and LATTICE authors.

### Build

- [ ] Reproduce DecisionRAG on Natural Questions (milestone 1).
- [ ] Implement the original pivot-and-merge method (option A) end to end, Jev only.
- [ ] Implement small-LLM categories (option B).
- [ ] Implement membership modes 1 and 2, with multiple memberships.
- [ ] Implement the routing loss and per-node diagnostics.
- [ ] Implement tuning edits (move, relabel, merge, split, restructure) and subtree rebuild.

### Experiments

- [ ] Set up GLiDE 27B and JEV-27B on an 80 GB GPU (adapter for JEV-27B's /v1/decide endpoint), then run the 10k-page pilot with the better-agreeing one.
- [ ] Repeat the pilot's main comparison with the Jev API.
- [ ] Tune hyperparameters on the pilot.
- [ ] Full build, baselines and ablations.

### Decisions

- [ ] Obtain ConditionalQA training questions and MTRAG design conversations (DecisionRAG's synthetic set and MTRAG-UN).
- [ ] Check how many NQ training questions have their gold page in the DAPR corpus, then split them into tuning and development sets.
- [ ] Set compute budget and milestone dates.

## References

- Wan, Junger et al. [Retriever-Free Retrieval: Navigating Documents with Decision Models](https://arxiv.org/pdf/2610.35868) (DecisionRAG), 2026.
- Gupta et al. [LLM-guided Hierarchical Search for End-to-end Reasoning Intensive Retrieval](https://arxiv.org/abs/2510.13217) (LATTICE), 2025–26. [Code](https://github.com/nilesh2797/llm-guided-hierarchical-search).
- TypeSafe AI. [Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), 2026.
- Zhong et al. [Explaining Datasets in Words: Statistical Models with Natural Language Parameters](https://arxiv.org/abs/2409.08466), 2024.
- Rehmat et al. [Iterative Definition Refinement for Zero-Shot Classification](https://arxiv.org/abs/2604.27335), 2026.
- [A Single Rewrite Suffices: Empirical Lessons from Production Skill Description Optimization](https://arxiv.org/pdf/2606.30775), 2026.
- [CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities](https://arxiv.org/pdf/2607.16122), 2026.
- [Semantic Compression Trees: Multi-Resolution Knowledge Retrieval via Hierarchical Semantic Residuals](https://arxiv.org/pdf/2608.21610), 2026.
- Whang, Dhillon and Gleich. [Non-exhaustive, Overlapping k-means](https://www.cs.purdue.edu/homes/dgleich/publications/Whang%202015%20-%20neokm.pdf) (NEO-K-Means), SDM 2015.
- Cleuziou. [An Extended Version of the k-Means Method for Overlapping Clustering](https://www.univ-orleans.fr/lifo/Members/cleuziou/papers/C_ICPR_08.pdf) (OKM), ICPR 2008. Medoid variant OKMED: Cleuziou 2009.
- Fastino. [GLiDE-27B](https://huggingface.co/fastino/GLiDE-27B), open-weight decision model, 2026.
- AutoTrust AI. [JEV-27B](https://huggingface.co/autotrust/JEV-27B), open-weight decision model distilled from Jev 1.13, 2026.
- Pinggy. [Best Open Source Jev Alternatives](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/), leaderboard summary as of November 18, 2026.

Cited from memory, links to be added: Sarthi et al., RAPTOR (ICLR 2024); Ashtiani, Kushagra and Ben-David, Clustering with Same-Cluster Queries (NeurIPS 2016); Mazumdar and Saha, Clustering with Noisy Queries (NeurIPS 2017); Ailon, Charikar and Newman, KwikCluster (2008); Zhang et al., ClusterLLM (EMNLP 2023); Rodriguez and Laio, density-peaks clustering (2014); Morin and Bengio, hierarchical softmax (2005); Breiman et al., CART and cost-complexity pruning (1984); Bengio, Weston and Grangier, Label Embedding Trees for Large Multi-class Tasks (NeurIPS 2010) and Griffin and Perona, taxonomies from confusion matrices (2008), both to verify; Li et al., Hyperband (2017). Bezdek, fuzzy c-means (1981).
