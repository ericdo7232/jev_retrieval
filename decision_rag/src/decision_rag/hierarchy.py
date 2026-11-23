"""Page embeddings (e5-base-v2) and k-means clusters.

Files:
  $DRAG_DIR/data/nq/page_embeddings.npy      float32 [pages, 768], L2-normalized
  $DRAG_DIR/trees/nq_seed{s}/clusters.parquet page_idx, cluster, sim (cosine to its centroid)
  $DRAG_DIR/trees/nq_seed{s}/centroids.npy
  $DRAG_DIR/trees/nq_seed{s}/stats.json       cluster sizes against the paper's (15-350, median 106)
"""
import json
import time

import numpy as np

from .config import nq_dir, tree_dir


def page_texts(pages, prefix):
    out = []
    for title, sent in zip(pages["title"], pages["first_sentence"]):
        out.append(f"{prefix}{title}. {sent}" if sent else f"{prefix}{title}")
    return out


def load_model(cls, name, dtype):
    """from_pretrained with `dtype` (transformers >= 4.56) or `torch_dtype` (older, e.g. as pinned by vLLM)."""
    try:
        return cls.from_pretrained(name, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype)


def embed(texts, model_name, batch_size, max_length):
    """Mean-pooled, L2-normalized e5 embeddings (fp16 on GPU), longest texts first for tight batches."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = load_model(AutoModel, model_name, torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device).eval()
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        batch = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**batch).last_hidden_state.float()
        m = batch["attention_mask"][..., None].float()
        out[idx] = F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
        if (i // batch_size) % 50 == 0:
            print(f"  embedded {i + len(idx)}/{len(texts)} ({time.time() - t0:.0f} s)", flush=True)
    return out


def page_embeddings(cfg, pages):
    path = nq_dir() / "page_embeddings.npy"
    if path.exists():
        x = np.load(path)
        if len(x) == len(pages):
            print(f"using cached embeddings {path}")
            return x
    e = cfg["hierarchy"]["embed"]
    x = embed(page_texts(pages, e["prefix"]), e["model"], e["batch_size"], e["max_length"])
    np.save(path, x)
    return x


def kmeans(cfg, x, seed, threads=8):
    """Spherical faiss k-means. Returns (assignment, cosine similarity to own centroid, centroids, stats)."""
    import faiss
    faiss.omp_set_num_threads(threads)
    k = cfg["hierarchy"]["kmeans"]
    km = faiss.Kmeans(x.shape[1], k["k"], niter=k["niter"], nredo=k["nredo"], spherical=k["spherical"],
                      seed=seed, verbose=False, max_points_per_centroid=len(x))
    t0 = time.time()
    km.train(x)
    _, assign = km.index.search(x, 1)
    assign = assign[:, 0]
    sims = (x * km.centroids[assign]).sum(1)
    sizes = np.bincount(assign, minlength=k["k"])
    stats = {"seed": seed, "k": k["k"], "seconds": round(time.time() - t0, 1),
             "objective": float(km.obj[-1]) if len(getattr(km, "obj", [])) else None,
             "size_min": int(sizes.min()), "size_median": float(np.median(sizes)), "size_max": int(sizes.max()),
             "empty_clusters": int((sizes == 0).sum()),
             "paper": "15-350 pages per cluster, median 106"}
    return assign, sims, km.centroids, stats


def build_tree(cfg, pages, x, seed):
    import pandas as pd
    d = tree_dir(seed)
    d.mkdir(parents=True, exist_ok=True)
    assign, sims, centroids, stats = kmeans(cfg, x, seed)
    pd.DataFrame({"page_idx": pages["page_idx"], "cluster": assign, "sim": sims}).to_parquet(
        d / "clusters.parquet", index=False)
    np.save(d / "centroids.npy", centroids)
    (d / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"seed {seed}: {json.dumps(stats)}", flush=True)
    return stats


def typical_titles(pages, clusters, k, limit):
    """Per cluster: up to `limit` titles, most similar to the centroid first."""
    df = clusters.merge(pages[["page_idx", "title"]], on="page_idx")
    df = df.sort_values(["cluster", "sim"], ascending=[True, False], kind="stable")
    by = df.groupby("cluster")["title"].apply(lambda s: s.head(limit).tolist())
    return [by.get(c, []) for c in range(k)]
