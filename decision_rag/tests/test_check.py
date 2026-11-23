"""The e5 routing check on a synthetic tree with known answers (a keyword embedding stands in for e5)."""
import json

import numpy as np
import pandas as pd

from decision_rag import check
from decision_rag.config import nq_dir, splits_dir, tree_dir

TOPICS = ["alpha", "beta", "gamma"]


def fake_embed(texts):
    """One axis per topic word; anything else is a small constant so vectors are never zero."""
    out = np.full((len(texts), 3), 0.01, dtype=np.float32)
    for i, t in enumerate(texts):
        for j, w in enumerate(TOPICS):
            if w in t:
                out[i, j] = 1.0
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def write_inputs(cfg):
    # 9 pages, 3 per cluster; page 3*c is its cluster's most central page.
    rng = np.random.default_rng(0)
    cluster = np.repeat([0, 1, 2], 3)
    x = np.eye(3, dtype=np.float32)[cluster] + 0.1 * rng.random((9, 3), dtype=np.float32) * (np.arange(9) % 3 > 0)[:, None]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    nq_dir().mkdir(parents=True)
    np.save(nq_dir() / "page_embeddings.npy", x)
    pd.DataFrame({"_id": [f"p{i}-0" for i in range(9)], "page_idx": range(9)}).to_parquet(
        nq_dir() / "paragraphs.parquet")
    # q1 asks about alpha and its gold page is in the alpha cluster; q2 asks about beta but its gold is in gamma.
    pd.DataFrame({"_id": ["q1", "q2"], "text": ["alpha question", "beta question"]}).to_parquet(
        nq_dir() / "queries_test.parquet")
    pd.DataFrame({"query_id": ["q1", "q2"], "corpus_id": ["p0-0", "p7-0"], "score": [1, 1]}).to_parquet(
        nq_dir() / "qrels_test.parquet")
    splits_dir().mkdir(parents=True, exist_ok=True)
    (splits_dir() / "nq_test500.txt").write_text("q1\nq2\n")
    d = tree_dir(0)
    d.mkdir(parents=True)
    pd.DataFrame({"page_idx": range(9), "cluster": cluster, "sim": 1.0}).to_parquet(d / "clusters.parquet")
    np.save(d / "centroids.npy", np.eye(3, dtype=np.float32) * 2)  # not normalized: check must normalize
    (d / "tree.json").write_text(json.dumps({"clusters": [
        {"id": c, "description": f"Pages about {TOPICS[c]}."} for c in (2, 0, 1)]}))  # out of order on purpose


def test_check_reports_known_ranks(drag_dir, cfg):
    write_inputs(cfg)
    cfg["check"].update(budgets=[0.01, 1.0], tree_budget=0.15, pages_kept=30, show_worst=1)
    (r,) = check.check_trees(cfg, [0], embed_fn=fake_embed)
    # 3 clusters: 1% keeps 1 cluster. q1's gold cluster ranks 1st, q2's (gamma, asked about beta) 2nd or 3rd.
    assert r["centroid_recall"] == {"1%": 0.5, "100%": 1.0}
    assert r["description_recall"] == {"1%": 0.5, "100%": 1.0}
    # Flat search finds both gold pages in the top 30 of 9; keeping 15% (1 cluster) prunes q2's gold cluster.
    assert r["pages"]["flat_recall@30"] == 1.0 and r["pages"]["tree_recall@30"] == 0.5
    assert r["worst"][0]["question"] == "beta question"
    assert r["worst"][0]["gold_cluster_description"] == "Pages about gamma."
    assert r["worst"][0]["top_cluster_description"] == "Pages about beta."
    assert json.loads((tree_dir(0) / "check.json").read_text())["questions"] == 2
