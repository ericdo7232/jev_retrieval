import json

import numpy as np
import pandas as pd

from decision_rag import describe, hierarchy
from decision_rag.config import nq_dir, tree_dir


def test_clean():
    assert describe.clean('"Pages about rivers."', 30) == ("Pages about rivers.", False)
    assert describe.clean("<think>\n\n</think>\n\nPages about rivers.\nExtra line", 30) == ("Pages about rivers.", False)
    out, cut = describe.clean(" ".join(f"w{i}," for i in range(40)), 30)
    assert cut and len(out.split()) == 30 and out.endswith("w29.")
    assert describe.clean("", 30) == ("", False)


class FakeTokenizer:
    """Chat template that keeps the user text, so a fake generator can read the prompt."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text):
        return {"input_ids": text.split()}


def fake_generate(texts):
    """Description prompts: answers with the first title listed, so each description can be traced to its cluster,
    padded past the word limit when that title's number is even. Shorten prompts: keeps the first 4 words."""
    out = []
    for t in texts:
        body = t.split("<assistant>")[0].split("\n\n", 1)[1]
        if t.startswith("<user>Shorten"):
            out.append(" ".join(body.split()[:4]))
            continue
        first = next(l for l in body.splitlines() if l[:1] == "T" and l[1:].isdigit())
        pad = " and more" * 20 if int(first[1:]) % 2 == 0 else ""
        out.append(f"<think>\n\n</think>\n\nGroup led by {first}.{pad}")
    return out


def test_describe_trees_writes_consistent_tree(drag_dir, cfg):
    rng = np.random.default_rng(0)
    centers = np.linalg.qr(rng.normal(size=(16, 3)))[0].T
    x = np.concatenate([c + 0.05 * rng.normal(size=(30, 16)) for c in centers]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    pages = pd.DataFrame({"page_idx": np.arange(90), "doc_id": [f"d{i}" for i in range(90)],
                          "title": [f"T{i}" for i in range(90)], "n_paragraphs": 1,
                          "first_sentence": [f"S{i}." for i in range(90)]})
    nq_dir().mkdir(parents=True)
    pages.to_parquet(nq_dir() / "pages.parquet")
    for seed in (0, 1):
        hierarchy.build_tree(cfg, pages, x, seed)

    describe.describe_trees(cfg, [0, 1], tok=FakeTokenizer(), generate=fake_generate)

    for seed in (0, 1):
        tree = json.loads((tree_dir(seed) / "tree.json").read_text())
        clusters = pd.read_parquet(tree_dir(seed) / "clusters.parquet")
        assert tree["seed"] == seed and tree["k"] == 3 and len(tree["clusters"]) == 3
        all_pages = [p for c in tree["clusters"] for p in c["pages"]]
        assert sorted(all_pages) == list(range(90))                       # every page exactly once
        titles = hierarchy.typical_titles(pages, clusters, 3, cfg["hierarchy"]["typical_titles"])
        for c in tree["clusters"]:
            assert c["pages"] == sorted(c["pages"])                        # corpus order inside a cluster
            assert c["size"] == len(c["pages"])
            assert set(c["pages"]) == set(clusters[clusters["cluster"] == c["id"]]["page_idx"])
            assert c["description"] == f"Group led by {titles[c['id']][0]}."  # matched to the right cluster
        rows = [json.loads(l) for l in (tree_dir(seed) / "descriptions.jsonl").read_text().splitlines()]
        assert [r["cluster"] for r in rows] == [0, 1, 2]
        for r in rows:                                                     # long ones shortened, none cut
            even = int(titles[r["cluster"]][0][1:]) % 2 == 0
            assert (r["shortened"] is not None) == even and not r["truncated"]
