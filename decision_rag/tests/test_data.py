import json

import pandas as pd

from decision_rag import data
from decision_rag.config import nq_dir, splits_dir


def test_first_sentence_rules(cfg):
    data._init(cfg)
    # citation markers removed; first sentence only
    assert data._first_sentence(["Hey Jude is a song.[1] It was released in 1968."]) == "Hey Jude is a song."
    # a too-short first paragraph falls through to the next one
    assert data._first_sentence(["Hey Jude", "Hey Jude is a song by the Beatles."]) == \
        "Hey Jude is a song by the Beatles."
    # empty paragraphs are skipped; if nothing is long enough, the first short sentence is kept
    assert data._first_sentence(["", "Short one."]) == "Short one."
    assert data._first_sentence(["", "  "]) == ""
    # run-on text is capped at max_words
    long = " ".join(["word"] * 300) + "."
    assert len(data._first_sentence([long]).split()) == cfg["hierarchy"]["first_sentence"]["max_words"]


def test_build_tables_orders_pages_and_paragraphs(cfg, corpus):
    pages, paras = data.build_tables(corpus, cfg, workers=1)
    # pages in order of first appearance in the file, not by ID
    assert pages["doc_id"].tolist() == ["d-9", "d-2", "d-5"]
    assert pages["page_idx"].tolist() == [0, 1, 2]
    assert pages["n_paragraphs"].tolist() == [3, 2, 2]
    assert pages["first_sentence"].tolist() == ["Hey Jude is a song by the Beatles.",
                                                 "Canberra is the capital city of Australia.", "Short one."]
    # paragraphs sorted by page, then paragraph number
    assert paras["_id"].tolist() == ["d-9-0", "d-9-1", "d-9-2", "d-2-0", "d-2-1", "d-5-0", "d-5-1"]
    assert list(paras.columns) == ["_id", "page_idx", "paragraph_no", "text"]


def test_build_tables_parallel_matches_serial(cfg, corpus):
    serial, _ = data.build_tables(corpus, cfg, workers=1)
    parallel, _ = data.build_tables(corpus, cfg, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_sample_is_deterministic_and_ordered():
    ids = [f"q{i}" for i in range(50)]
    a = data.sample_test(ids, 10, seed=0)
    assert a == data.sample_test(list(reversed(ids)), 10, seed=0)  # independent of input order
    assert len(set(a)) == 10 and a != sorted(a)                   # kept in sampled order
    assert a != data.sample_test(ids, 10, seed=1)


def test_prepare_end_to_end(drag_dir, cfg, raw_files):
    stats = data.prepare(cfg, workers=1, paths=raw_files)
    assert stats["pages"] == 3 and stats["paragraphs"] == 7 and stats["queries"] == 4
    assert stats["gold_missing_from_corpus"] == 0
    assert stats["questions_with_gold_on_several_pages"] == 0
    for f in ["pages.parquet", "paragraphs.parquet", "queries_test.parquet", "qrels_test.parquet", "stats.json"]:
        assert (nq_dir() / f).exists(), f
    assert json.loads((nq_dir() / "stats.json").read_text())["test_sample"] == 3
    ids = (splits_dir() / "nq_test500.txt").read_text().split()
    assert len(ids) == 3
    test = data.load_test()
    assert [t[0] for t in test] == ids
    gold = {q: g for q, _, g in test}
    assert all(isinstance(g, set) and g for g in gold.values())
    assert data.load_test(n=2) == test[:2]
