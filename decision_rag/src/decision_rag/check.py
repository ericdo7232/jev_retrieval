"""Quick check of the trees without the decision model: can e5 route the test questions to the gold page's cluster?

For each test question (e5 embedding with the "query: " prefix), per seed:
  centroid recall@b     the gold page's cluster is in the top b% of the 1,000 clusters ranked by similarity to the
                        cluster centroids: is the grouping itself findable?
  description recall@b  the same, ranked by similarity to the cluster descriptions (what the decision model reads in
                        stage 1): do the descriptions keep what the grouping has?
  pages: flat vs tree   gold page in the top 30 pages by e5, over all 108,626 pages (flat) or only over pages in the
                        top 15% of clusters by description (tree). The paper's 15%-vs-flat ablation, with e5 in place
                        of Jev: if tree is close to flat, pruning clusters loses little.
e5 is much weaker than the decision model, so these are floors, for comparing seeds and description versions and for
spotting a broken tree; they do not predict the final scores.

Writes $DRAG_DIR/trees/nq_seed{s}/check.json and prints a table and the worst-routed questions.
"""
import json
import math

import numpy as np
import pandas as pd

from .config import nq_dir, tree_dir
from .data import load_paragraphs, load_test
from .hierarchy import embed


def gold_pages(tests, para_page):
    """Gold page per question (the paper: gold evidence is on one page; several pages keep the best-ranked)."""
    return [sorted({int(para_page[g]) for g in gold}) for _, _, gold in tests]


def best_rank(scores, golds):
    """1-based rank of the best-scoring gold item in each row of scores [questions, items]."""
    out = np.empty(len(golds), dtype=int)
    for i, g in enumerate(golds):
        top = scores[i, g].max()
        out[i] = 1 + int((scores[i] > top).sum())
    return out


def recall(ranks, n, budgets):
    return {f"{round(b * 100)}%": float(np.mean(ranks <= math.ceil(b * n))) for b in budgets}


def check_tree(cfg, seed, q, golds, page_x, embed_fn, texts):
    c = cfg["check"]
    d = tree_dir(seed)
    clusters = pd.read_parquet(d / "clusters.parquet").sort_values("page_idx")
    page_cluster = clusters["cluster"].to_numpy()
    k = int(page_cluster.max()) + 1
    gold_clusters = [sorted({int(page_cluster[p]) for p in g}) for g in golds]

    centroids = np.load(d / "centroids.npy")
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    cen_rank = best_rank(q @ centroids.T, gold_clusters)
    out = {"seed": seed, "questions": len(golds), "clusters": k,
           "centroid_recall": recall(cen_rank, k, c["budgets"]), "centroid_median_rank": float(np.median(cen_rank))}

    tree_path = d / "tree.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
        descs = [cl["description"] for cl in sorted(tree["clusters"], key=lambda cl: cl["id"])]
        desc_scores = q @ embed_fn([cfg["hierarchy"]["embed"]["prefix"] + t for t in descs]).T
        desc_rank = best_rank(desc_scores, gold_clusters)
        out["description_recall"] = recall(desc_rank, k, c["budgets"])
        out["description_median_rank"] = float(np.median(desc_rank))

        # Pages: flat over everything vs only inside the top clusters by description.
        page_scores = q @ page_x.T
        flat = best_rank(page_scores, golds)
        kept = np.argsort(-desc_scores, axis=1)[:, :math.ceil(c["tree_budget"] * k)]
        inside = np.zeros_like(page_scores, dtype=bool)
        for i, ks in enumerate(kept):
            inside[i] = np.isin(page_cluster, ks)
        tree_ranks = best_rank(np.where(inside, page_scores, -np.inf), golds)
        tree_ranks[[not inside[i, g].any() for i, g in enumerate(golds)]] = np.iinfo(tree_ranks.dtype).max  # pruned
        top = c["pages_kept"]
        out["pages"] = {f"flat_recall@{top}": float(np.mean(flat <= top)),
                        f"tree_recall@{top}": float(np.mean(tree_ranks <= top)),
                        "tree_budget": c["tree_budget"]}
        worst = np.argsort(-desc_rank)[:c["show_worst"]]
        out["worst"] = [{"question": texts[i], "gold_cluster": gold_clusters[i][0],
                         "gold_cluster_rank": int(desc_rank[i]),
                         "gold_cluster_description": descs[gold_clusters[i][0]],
                         "top_cluster_description": descs[int(np.argmax(desc_scores[i]))]} for i in worst]
    (d / "check.json").write_text(json.dumps(out, indent=1))
    return out


def check_trees(cfg, seeds, embed_fn=None):
    e = cfg["hierarchy"]["embed"]
    if embed_fn is None:
        def embed_fn(texts):
            return embed(texts, e["model"], e["batch_size"], e["max_length"])
    tests = load_test()
    paras = load_paragraphs(columns=["_id", "page_idx"])
    golds = gold_pages(tests, dict(zip(paras["_id"], paras["page_idx"])))
    del paras
    q = embed_fn([cfg["check"]["query_prefix"] + t for _, t, _ in tests])
    page_x = np.load(nq_dir() / "page_embeddings.npy")
    texts = [t for _, t, _ in tests]
    results = [check_tree(cfg, s, q, golds, page_x, embed_fn, texts) for s in seeds]

    budgets = list(results[0]["centroid_recall"])
    print(f"\nGold cluster in the top b% of clusters, e5 routing ({len(tests)} test questions):")
    print(f"{'seed':>4}  {'by':<12} " + " ".join(f"{b:>6}" for b in budgets) + "  median rank")
    for r in results:
        for name in ("centroid", "description"):
            if f"{name}_recall" in r:
                print(f"{r['seed']:>4}  {name:<12} " + " ".join(f"{r[f'{name}_recall'][b]:>6.3f}" for b in budgets)
                      + f"  {r[f'{name}_median_rank']:>6.0f}")
    for r in results:
        if "pages" in r:
            p = r["pages"]
            print(f"seed {r['seed']}: gold page in e5 top {cfg['check']['pages_kept']}: flat "
                  f"{list(p.values())[0]:.3f}, inside top {round(p['tree_budget'] * 100)}% clusters "
                  f"{list(p.values())[1]:.3f}")
    if "worst" in results[0]:
        print(f"\nWorst-routed questions (seed {results[0]['seed']}, by description rank):")
        for w in results[0]["worst"]:
            print(f"  [{w['gold_cluster_rank']}] {w['question']}\n      gold cluster: {w['gold_cluster_description']}"
                  f"\n      top cluster:  {w['top_cluster_description']}")
    return results
