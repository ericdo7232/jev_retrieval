import json

import numpy as np
import pandas as pd
import pytest

from decision_rag import hierarchy
from decision_rag.config import tree_dir


def blobs(n_per=40, k=3, d=16, seed=0):
    """k well-separated directions on the unit sphere, with small noise."""
    rng = np.random.default_rng(seed)
    centers = np.linalg.qr(rng.normal(size=(d, k)))[0].T  # orthonormal
    x = np.concatenate([c + 0.05 * rng.normal(size=(n_per, d)) for c in centers]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.repeat(np.arange(k), n_per)


def pages_for(n):
    return pd.DataFrame({"page_idx": np.arange(n), "title": [f"T{i}" for i in range(n)],
                         "first_sentence": [f"Sentence {i}." for i in range(n)]})


def test_page_texts():
    p = pd.DataFrame({"title": ["Canberra", "Stub"], "first_sentence": ["Canberra is a city.", ""]})
    assert hierarchy.page_texts(p, "passage: ") == ["passage: Canberra. Canberra is a city.", "passage: Stub"]


def test_kmeans_recovers_blobs(cfg):
    x, truth = blobs()
    assign, sims, centroids, stats = hierarchy.kmeans(cfg, x, seed=0, threads=1)
    # each true blob maps to exactly one cluster
    for b in range(3):
        assert len(set(assign[truth == b])) == 1
    assert len(set(assign)) == 3
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1, atol=1e-4)   # spherical
    np.testing.assert_allclose(sims, (x * centroids[assign]).sum(1), atol=1e-5)
    assert sims.min() > 0.9
    assert stats["size_min"] == stats["size_max"] == 40 and stats["empty_clusters"] == 0


def test_kmeans_is_deterministic_per_seed(cfg):
    x, _ = blobs(seed=3)
    a1, *_ = hierarchy.kmeans(cfg, x, seed=5, threads=1)
    a2, *_ = hierarchy.kmeans(cfg, x, seed=5, threads=1)
    np.testing.assert_array_equal(a1, a2)


def test_build_tree_and_typical_titles(drag_dir, cfg):
    x, truth = blobs()
    pages = pages_for(len(x))
    hierarchy.build_tree(cfg, pages, x, seed=0)
    d = tree_dir(0)
    clusters = pd.read_parquet(d / "clusters.parquet")
    assert sorted(clusters["page_idx"]) == list(range(len(x)))
    assert json.loads((d / "stats.json").read_text())["k"] == 3
    titles = hierarchy.typical_titles(pages, clusters, k=3, limit=4)
    assert [len(t) for t in titles] == [4, 4, 4]
    for c, t in enumerate(titles):  # most similar to the centroid first, all from the cluster
        members = clusters[clusters["cluster"] == c].sort_values("sim", ascending=False)
        assert t == [f"T{i}" for i in members["page_idx"][:4]]


@pytest.fixture
def tiny_encoder(tmp_path):
    """A small random BERT saved locally, standing in for e5-base-v2 (same pooling code path)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "passage", ":", ".", "canberra", "is", "a", "city",
             "stub", "the", "capital", "of", "australia", "song"]
    d = tmp_path / "enc"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64)).save_pretrained(d)
    return str(d)


def test_embed_pooling_and_order(tiny_encoder):
    texts = ["passage: canberra", "passage: canberra is a city . the capital of australia . a city . a city",
             "passage: stub", "passage: song"]
    x = hierarchy.embed(texts, tiny_encoder, batch_size=2, max_length=64)
    assert x.shape == (4, 32) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
    # rows follow the input order despite length-sorted batching, and padding does not change a text's embedding
    for i, t in enumerate(texts):
        np.testing.assert_allclose(hierarchy.embed([t], tiny_encoder, batch_size=1, max_length=64)[0], x[i],
                                   atol=1e-5)
