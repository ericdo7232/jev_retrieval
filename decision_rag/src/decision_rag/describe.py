"""Cluster descriptions with Qwen3-8B (greedy, thinking off), and the final tree file.

Files under $DRAG_DIR/trees/nq_seed{s}/:
  descriptions.jsonl   one line per cluster: cluster, description, raw output, shortened output (or null), truncated
  tree.json            {seed, k, clusters: [{id, description, size, pages (page_idx, corpus order)}]}
"""
import json
import time

import pandas as pd

from .config import tree_dir
from .data import load_pages
from .hierarchy import typical_titles


def chat(tok, content):
    msg = [{"role": "user", "content": content}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompts(cfg, tok, titles_per_cluster):
    tmpl = cfg["describe"]["prompt"]
    return [chat(tok, tmpl.format(titles="\n".join(titles))) for titles in titles_per_cluster]


def shorten_prompts(cfg, tok, descriptions):
    tmpl = cfg["describe"]["shorten_prompt"]
    return [chat(tok, tmpl.format(description=d)) for d in descriptions]


def first_line(text):
    """First non-empty line, without quotes or a leftover think block."""
    text = text.split("</think>")[-1]
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return line.strip('"“”\' ')


def clean(text, max_words):
    """First non-empty line, without quotes or a leftover think block, cut to max_words words."""
    line = first_line(text)
    words = line.split()
    if len(words) <= max_words:
        return line, False
    return " ".join(words[:max_words]).rstrip(",;:") + ".", True


def vllm_generator(cfg):
    """Loads the model once; returns generate(texts) -> outputs."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=cfg["describe"]["model"], dtype="bfloat16", max_model_len=8192, seed=0,
              gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=cfg["describe"]["max_new_tokens"])
    return lambda texts: [o.outputs[0].text for o in llm.generate(texts, params)]


def transformers_generator(cfg, tok, batch_size=16):
    """Loads the model once; returns generate(texts) -> outputs."""
    import torch
    from transformers import AutoModelForCausalLM
    from .hierarchy import load_model
    model = load_model(AutoModelForCausalLM, cfg["describe"]["model"], torch.bfloat16).to("cuda").eval()
    tok.padding_side = "left"

    def generate(texts):
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = [None] * len(texts)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=cfg["describe"]["max_new_tokens"], do_sample=False)
            for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                out[i] = tok.decode(g, skip_special_tokens=True)
            print(f"  generated {b + len(idx)}/{len(texts)}", flush=True)
        return out
    return generate


def describe_trees(cfg, seeds, backend=None, tok=None, generate=None):
    """Descriptions for every seed's clusters in one model load (3 x 1,000 prompts).

    tok, generate: injected tokenizer and generate(texts) -> outputs, for tests.
    """
    pages = load_pages()
    k, limit = cfg["hierarchy"]["kmeans"]["k"], cfg["hierarchy"]["typical_titles"]
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["describe"]["model"])
    clusters = {s: pd.read_parquet(tree_dir(s) / "clusters.parquet") for s in seeds}
    texts, keys = [], []
    for s in seeds:
        for c, titles in enumerate(typical_titles(pages, clusters[s], k, limit)):
            texts.append(prompts(cfg, tok, [titles])[0])
            keys.append((s, c))
    lens = [len(tok(t)["input_ids"]) for t in texts[:50]]
    print(f"{len(texts)} prompts, ~{sum(lens) // len(lens)} tokens each", flush=True)

    backend = backend or cfg["describe"]["backend"]
    t0 = time.time()
    if generate is None and backend == "vllm":
        try:
            generate = vllm_generator(cfg)
        except Exception as e:  # missing or incompatible vLLM: fall back rather than lose the GPU slot
            print(f"vLLM failed ({type(e).__name__}: {e}); falling back to transformers", flush=True)
    if generate is None:
        generate = transformers_generator(cfg, tok)
    raw = generate(texts)
    print(f"generated in {time.time() - t0:.0f} s", flush=True)

    # Over-long descriptions: ask the model for a shorter complete sentence instead of cutting mid-sentence.
    max_words = cfg["describe"]["max_words"]
    long = [i for i, r in enumerate(raw) if len(first_line(r).split()) > max_words]
    short = dict(zip(long, generate(shorten_prompts(cfg, tok, [first_line(raw[i]) for i in long])))) if long else {}
    print(f"{len(long)} descriptions over {max_words} words were shortened by the model", flush=True)

    for s in seeds:
        d = tree_dir(s)
        rows = []
        for i, ((seed, c), r) in enumerate(zip(keys, raw)):
            if seed == s:
                desc, cut = clean(short.get(i, r), max_words)
                rows.append({"cluster": c, "description": desc, "raw": r, "shortened": short.get(i),
                             "truncated": cut})
        with open(d / "descriptions.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        members = clusters[s].sort_values("page_idx").groupby("cluster")["page_idx"].apply(
            lambda p: [int(i) for i in p])
        tree = {"seed": s, "k": k, "description_model": cfg["describe"]["model"],
                "clusters": [{"id": c, "description": rows[c]["description"],
                              "size": len(members.get(c, [])), "pages": members.get(c, [])} for c in range(k)]}
        (d / "tree.json").write_text(json.dumps(tree))
        n_short = sum(r["shortened"] is not None for r in rows)
        n_cut = sum(r["truncated"] for r in rows)
        n_empty = sum(not r["description"] for r in rows)
        n_typical = sum("typical" in r["description"].lower() for r in rows)
        print(f"seed {s}: wrote {d / 'tree.json'}; {n_short} shortened by the model, {n_cut} still cut to "
              f"{max_words} words, {n_empty} empty, {n_typical} mention 'typical'")
        for row in rows[:3]:
            print(f"  cluster {row['cluster']}: {row['description']}")
