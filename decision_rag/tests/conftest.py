"""CPU-only tests. Large outputs and the test split go to a temporary directory, never the repo or $DRAG_DIR."""
import copy
import os
import sys

# torch and faiss-cpu each bundle libomp on macOS; loading both aborts (and hangs) without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_rag.config import load_config  # noqa: E402


@pytest.fixture
def drag_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAG_DIR", str(tmp_path / "drag"))
    monkeypatch.setenv("DRAG_SPLITS_DIR", str(tmp_path / "splits"))
    return tmp_path


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["data"]["test_sample"]["size"] = 3
    c["hierarchy"]["kmeans"].update(k=3, niter=10, nredo=2, seeds=[0])
    c["hierarchy"]["typical_titles"] = 4
    return c


# Synthetic DAPR-shaped corpus. Docs appear out of ID order and paragraphs out of order, to test sorting.
PARAGRAPHS = [
    # (doc_id, title, paragraph_no, text)
    ("d-9", "Hey Jude", 1, "Hey Jude is a song by the Beatles.[1] It was released in 1968."),
    ("d-9", "Hey Jude", 0, "Hey Jude"),                                  # caption-like first paragraph
    ("d-9", "Hey Jude", 2, "The song was written by Paul McCartney."),
    ("d-2", "Canberra", 0, "Canberra is the capital city of Australia. It lies in the ACT."),
    ("d-2", "Canberra", 1, "The city was founded in 1913."),
    ("d-5", "Stub", 0, ""),                                               # empty, then too short everywhere
    ("d-5", "Stub", 1, "Short one."),
]


@pytest.fixture
def corpus():
    rows = [{"_id": f"{d}-{n}", "doc_id": d, "title": t, "paragraph_no": n, "text": x}
            for d, t, n, x in PARAGRAPHS]
    return pd.DataFrame(rows)


@pytest.fixture
def raw_files(tmp_path, corpus):
    """Local files in the layout prepare() reads, instead of downloading."""
    d = tmp_path / "raw"
    d.mkdir()
    corpus.to_parquet(d / "corpus.parquet")
    pd.DataFrame({"_id": ["q1", "q2", "q3", "q4"],
                  "text": ["who sang hey jude", "capital of australia", "when was canberra founded", "x"]}
                 ).to_parquet(d / "queries.parquet")
    pd.DataFrame({"query_id": ["q1", "q2", "q3", "q4"],
                  "corpus_id": ["d-9-1", "d-2-0", "d-2-1", "d-5-1"], "score": [1, 1, 1, 1]}
                 ).to_parquet(d / "qrels.parquet")
    return {k: str(d / f"{k}.parquet") for k in ("corpus", "queries", "qrels")}
