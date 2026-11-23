"""DAPR Natural Questions: download, page and paragraph tables, first sentences, test sample, loaders.

Files under $DRAG_DIR/data/nq/:
  raw/...                    the DAPR parquet files as downloaded
  pages.parquet              page_idx, doc_id, title, n_paragraphs, first_sentence  (page_idx = corpus order)
  paragraphs.parquet         _id, page_idx, paragraph_no, text  (sorted by page_idx, paragraph_no)
  queries_test.parquet       _id, text  (3,610 questions)
  qrels_test.parquet         query_id, corpus_id, score  (gold paragraphs)
  stats.json
"""
import json
import random
import re
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

from .config import nq_dir, splits_dir

_SEG = None
_CFG = None


def download(cfg):
    from huggingface_hub import hf_hub_download
    raw = nq_dir() / "raw"
    sub = cfg["data"]["subset"]
    files = [f"{sub}/corpus/0000.parquet", f"{sub}/queries/test.parquet", f"{sub}/qrels/test.parquet"]
    return {f.split("/")[1]: hf_hub_download(cfg["data"]["hf_repo"], f, repo_type="dataset", local_dir=raw)
            for f in files}


def _clean(text):
    return re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", text or "")).strip()


def _init(cfg):
    global _SEG, _CFG
    import pysbd
    _SEG = pysbd.Segmenter(language="en", clean=False)
    _CFG = cfg["hierarchy"]["first_sentence"]


def _first_sentence(paragraphs):
    """First sentence of the first paragraph that has one of at least min_words words; else the first found."""
    fallback = ""
    for p in paragraphs:
        p = _clean(p)
        if not p:
            continue
        sents = [s.strip() for s in _SEG.segment(p[:2000]) if s.strip()]
        if not sents:
            continue
        words = sents[0].split()
        s = " ".join(words[:_CFG["max_words"]])
        if len(words) >= _CFG["min_words"]:
            return s
        fallback = fallback or s
    return fallback


def build_tables(corpus, cfg, workers=8):
    """Corpus rows (_id, doc_id, title, paragraph_no, text) -> (pages, paragraphs sorted by page then paragraph).

    page_idx numbers pages in order of first appearance in the corpus file ("corpus order").
    """
    corpus = corpus.copy()
    corpus["page_idx"] = pd.factorize(corpus["doc_id"])[0]
    corpus = corpus.sort_values(["page_idx", "paragraph_no"], kind="stable").reset_index(drop=True)
    pages = corpus.groupby("page_idx", sort=True).agg(doc_id=("doc_id", "first"), title=("title", "first"),
                                                      n_paragraphs=("_id", "size")).reset_index()
    k = cfg["hierarchy"]["first_sentence"]["paragraphs_to_try"]
    heads = corpus[corpus.groupby("page_idx").cumcount() < k].groupby("page_idx", sort=True)["text"].apply(list)
    print(f"first sentences for {len(heads)} pages with {workers} workers ...", flush=True)
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
            pages["first_sentence"] = pool.map(_first_sentence, heads.tolist(), chunksize=500)
    else:
        _init(cfg)
        pages["first_sentence"] = [_first_sentence(h) for h in heads.tolist()]
    return pages, corpus[["_id", "page_idx", "paragraph_no", "text"]]


def prepare(cfg, workers=8, paths=None):
    """Download (unless `paths` gives local corpus/queries/qrels files) and write everything under nq_dir()."""
    out = nq_dir()
    out.mkdir(parents=True, exist_ok=True)
    paths = paths or download(cfg)

    corpus = pq.read_table(paths["corpus"], columns=["_id", "doc_id", "title", "paragraph_no", "text"]).to_pandas()
    pages, paragraphs = build_tables(corpus, cfg, workers)
    del corpus
    n_pages, n_paras = len(pages), len(paragraphs)
    print(f"corpus: {n_pages} pages, {n_paras} paragraphs "
          f"(paper: {cfg['data']['expected_pages']}, {cfg['data']['expected_paragraphs']})", flush=True)
    pages.to_parquet(out / "pages.parquet", index=False)
    paragraphs.to_parquet(out / "paragraphs.parquet", index=False)

    queries = pd.read_parquet(paths["queries"])
    qrels = pd.read_parquet(paths["qrels"])
    qrels = qrels[qrels["score"] > 0]
    queries.to_parquet(out / "queries_test.parquet", index=False)
    qrels.to_parquet(out / "qrels_test.parquet", index=False)
    corpus = paragraphs

    # Gold evidence should sit on one page per question (paper: 1-3 paragraphs on one page).
    para_page = dict(zip(corpus["_id"], corpus["page_idx"]))
    missing = int((~qrels["corpus_id"].isin(para_page)).sum())
    pages_per_q = qrels.assign(page=qrels["corpus_id"].map(para_page)).groupby("query_id")["page"].nunique()

    test = cfg["data"]["test_sample"]
    sample = sample_test(queries["_id"].tolist(), min(test["size"], len(queries)), test["seed"])
    splits_dir().mkdir(exist_ok=True)
    (splits_dir() / "nq_test500.txt").write_text("\n".join(sample) + "\n")

    empty = int((pages["first_sentence"] == "").sum())
    stats = {"pages": n_pages, "paragraphs": n_paras, "queries": len(queries), "gold_paragraphs": len(qrels),
             "gold_missing_from_corpus": missing, "questions_with_gold_on_several_pages": int((pages_per_q > 1).sum()),
             "pages_without_first_sentence": empty, "test_sample": len(sample),
             "first_sentence_words_median": float(pages["first_sentence"].str.split().str.len().median())}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("examples:")
    for _, r in pages.sample(min(5, len(pages)), random_state=0).iterrows():
        print(f"  {r.title!r}: {r.first_sentence!r}")
    return stats


def sample_test(ids, size, seed):
    """Uniform sample of `size` question IDs, in sampled order (so the first 100 lines are a random 100)."""
    return random.Random(seed).sample(sorted(ids), size)


# ---------------------------------------------------------------- loaders (used by later steps)

def load_pages():
    return pd.read_parquet(nq_dir() / "pages.parquet")


def load_paragraphs(columns=None):
    return pd.read_parquet(nq_dir() / "paragraphs.parquet", columns=columns)


def load_test(n=None):
    """[(query_id, text, set of gold paragraph ids)] for the committed 500-question sample (first n)."""
    ids = (splits_dir() / "nq_test500.txt").read_text().split()[:n]
    q = pd.read_parquet(nq_dir() / "queries_test.parquet").set_index("_id")["text"]
    gold = pd.read_parquet(nq_dir() / "qrels_test.parquet").groupby("query_id")["corpus_id"].apply(set)
    return [(i, q[i], gold[i]) for i in ids]
