"""HTTP clients with one shared call: noul(state, questions) -> P(yes) for each question.

Kev-27B   POST /v1/systemone   one request: the state plus every question (state computed once, questions isolated).
JEV-27B   POST /v1/decide      one request per question; requests share the state as a prompt prefix, so they
                               only share work if vLLM's prefix caching reuses it.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from common import http_json, http_text


def _trim(resp, key, keep=3):
    """Copy of a response with a long answers dict cut down, for storing as a sample."""
    out = dict(resp)
    if isinstance(out.get(key), dict) and len(out[key]) > keep:
        out[key] = dict(list(out[key].items())[:keep])
    return out


class KevClient:
    name = "kev"

    def __init__(self, url, timeout=1800, **_):
        self.url = url.rstrip("/")
        self.timeout = timeout
        key = os.environ.get("KEV_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def info(self):
        return http_json("GET", self.url + "/v1/models", timeout=30, headers=self.headers)

    def metrics(self):
        return None

    def noul(self, state, questions, endpoint="/v1/systemone", **_):
        ids = [f"q{i:05d}" for i in range(len(questions))]
        payload = {
            "model": "kev-latest",
            "state": state,
            "questions": {qid: {"type": "noul", "instructions": q} for qid, q in zip(ids, questions)},
        }
        t = time.perf_counter()
        r = http_json("POST", self.url + endpoint, payload, self.timeout, self.headers)
        wall = time.perf_counter() - t
        answers = r.get("answers") or {}
        usage = r.get("usage") or {}
        latency_ms = r.get("latency_ms")
        return {
            "probs": [(answers.get(qid) or {}).get("noul") for qid in ids],
            "wall_s": wall,
            "input_tokens": usage.get("input_tokens"),
            "server_s": latency_ms / 1000 if latency_ms is not None else None,
            "n_requests": 1,
            "raw": _trim(r, "answers"),
        }


class Jev27bClient:
    name = "jev27b"

    def __init__(self, url, timeout=1800, concurrency=32, **_):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.concurrency = concurrency
        # serve_decide.py accepts thinking=off; set JEV_THINKING="" to leave the field out.
        self.thinking = os.environ.get("JEV_THINKING", "off")

    def info(self):
        try:
            return http_json("GET", self.url + "/v1/decide/info", timeout=30)
        except Exception:
            return http_json("GET", self.url + "/v1/models", timeout=30)

    def metrics(self):
        """Sum of vLLM's prefix-cache counters from /metrics, or None if not exposed."""
        try:
            text = http_text(self.url + "/metrics")
        except Exception:
            return None
        out = {}
        for line in text.splitlines():
            if line.startswith("vllm:prefix_cache"):
                name, _, val = line.rpartition(" ")
                key = name.split("{")[0]
                try:
                    out[key] = out.get(key, 0.0) + float(val)
                except ValueError:
                    pass
        return out

    def _one(self, state, question):
        payload = {"kind": "noul", "state": state, "question": question}
        if self.thinking:
            payload["thinking"] = self.thinking
        r = http_json("POST", self.url + "/v1/decide", payload, self.timeout)
        options = r.get("options") or ["false", "true"]
        probs = r.get("probabilities") or []
        i = options.index("true") if "true" in options else 1
        return (probs[i] if i < len(probs) else None), r

    def noul(self, state, questions, concurrency=None, **_):
        workers = max(1, min(concurrency or self.concurrency, len(questions)))
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda q: self._one(state, q), questions))
        wall = time.perf_counter() - t
        return {
            "probs": [p for p, _ in results],
            "wall_s": wall,
            "input_tokens": sum((r.get("usage") or {}).get("prompt_tokens") or 0 for _, r in results),
            "server_s": sum(r.get("elapsed_seconds") or 0 for _, r in results),
            "n_requests": len(questions),
            "raw": results[0][1] if results else None,
        }


def make(model, url, **kw):
    return {"kev": KevClient, "jev27b": Jev27bClient}[model](url, **kw)
