#!/usr/bin/env python3
"""Run DecisionRAG-shaped tests against a running Kev-27B or JEV-27B server.

Run from decision_models/:
    python bench.py --model kev    --url http://127.0.0.1:8008 --tag default
    python bench.py --model jev27b --url http://127.0.0.1:8000 --tag prefix_on

Tests (see README.md):
    smoke        two obvious questions; prints a raw response
    needle       6 real questions, gold page hidden among 200 titles; rank of the gold page
    position     gold page at the start, middle and end of 1,250 titles; its P(yes)
    consistency  repeat a request; Kev: joint vs one-question-at-a-time and question order;
                 JEV-27B: concurrent vs sequential
    scale        stage-shaped workloads; time, tokens, GPU memory (JEV-27B: cold vs warm request)

Results are written to results/<model>_<tag>_<time>.json after every test.
"""
import argparse
import datetime
import json
import os
import random
import socket
import time

import clients
import common

ALL_TESTS = ["smoke", "needle", "position", "consistency", "scale"]


def max_abs_diff(a, b):
    d = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
    return max(d) if d else None


def wait_ready(client, wait_s, server_pid):
    deadline = time.time() + wait_s
    while True:
        try:
            return client.info()
        except Exception as e:
            last = e
        if server_pid is not None:
            try:
                os.kill(server_pid, 0)
            except OSError:
                raise RuntimeError(f"server process {server_pid} exited before it was ready; see its log")
        if time.time() > deadline:
            raise RuntimeError(f"server not ready after {wait_s:.0f}s: {last}")
        time.sleep(10)


def t_smoke(c, args):
    state = "The customer says they were charged twice for order 1182 and asks for one of the charges to be refunded."
    qs = ["Is the customer asking for a refund?", "Is the customer asking about a late delivery?"]
    r = c.noul(state, qs)
    p = r["probs"]
    ok = None not in p and p[0] > 0.5 > p[1]
    print(f"  P(yes): refund={p[0]}  late delivery={p[1]}  -> {'PASS' if ok else 'CHECK'}")
    print(f"  raw response sample: {json.dumps(r['raw'])[:800]}")
    return {"probs": p, "pass": ok, "wall_s": r["wall_s"], "raw_sample": r["raw"]}


def t_needle(c, args):
    rows = []
    for i, nd in enumerate(common.NEEDLES):
        w = common.needle_workload(nd, n=200, seed=100 + i)
        r = c.noul(w["state"], w["questions"])
        p = r["probs"]
        gp = p[w["gold_index"]]
        rank = None if gp is None else 1 + sum(1 for x in p if x is not None and x > gp)
        order = sorted(range(len(p)), key=lambda j: -(p[j] if p[j] is not None else -1.0))
        top5 = [[w["titles"][j], p[j]] for j in order[:5]]
        print(f"  {nd['query'][:48]:48s} gold rank {rank}  P={gp}  top: {top5[0][0]}")
        rows.append({"query": nd["query"], "gold": nd["gold"], "gold_rank": rank, "gold_p": gp, "top5": top5,
                     "titles": w["titles"], "probs": p, "wall_s": r["wall_s"], "input_tokens": r["input_tokens"]})
    rank1 = sum(1 for q in rows if q["gold_rank"] == 1)
    top3 = sum(1 for q in rows if q["gold_rank"] is not None and q["gold_rank"] <= 3)
    print(f"  gold page ranked 1st: {rank1}/{len(rows)}, in top 3: {top3}/{len(rows)}")
    return {"queries": rows, "gold_rank1": rank1, "gold_top3": top3, "n": len(rows)}


def t_position(c, args):
    n, rows = 1250, []
    for i, nd in enumerate(common.NEEDLES[:2]):
        for pos in (1, n // 2, n):
            w = common.needle_workload(nd, n=n, seed=200 + i, gold_pos=pos)
            r = c.noul(w["state"], [common.Q_PAGE.format(n=pos)])
            print(f"  {nd['query'][:40]:40s} gold at {pos:4d}/{n}: P(yes)={r['probs'][0]}")
            rows.append({"query": nd["query"], "gold_pos": pos, "gold_p": r["probs"][0],
                         "wall_s": r["wall_s"], "input_tokens": r["input_tokens"]})
    return {"n_candidates": n, "rows": rows}


def t_consistency(c, args):
    w = common.needle_workload(common.NEEDLES[0], n=200, seed=100)
    state, qs = w["state"], w["questions"][:50]
    a = c.noul(state, qs)
    b = c.noul(state, qs)
    out = {"n_questions": len(qs), "repeat_max_abs_diff": max_abs_diff(a["probs"], b["probs"])}
    if args.model == "kev":
        s = c.noul(state, qs, endpoint="/v1/systemone/separate")
        out["joint_vs_separate_max_abs_diff"] = max_abs_diff(a["probs"], s["probs"])
        perm = list(range(len(qs)))
        random.Random(0).shuffle(perm)
        shuffled = c.noul(state, [qs[i] for i in perm])
        unshuffled = [None] * len(qs)
        for k, i in enumerate(perm):
            unshuffled[i] = shuffled["probs"][k]
        out["question_order_max_abs_diff"] = max_abs_diff(a["probs"], unshuffled)
    else:
        s = c.noul(state, qs, concurrency=1)
        out["concurrent_vs_sequential_max_abs_diff"] = max_abs_diff(a["probs"], s["probs"])
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def _scale_kev(c, name):
    runs = []
    for seed in (1, 2):
        w = common.scale_workload(name, seed)
        with common.GpuMonitor() as g:
            r = c.noul(w["state"], w["questions"])
        runs.append({"seed": seed, "wall_s": r["wall_s"], "server_s": r["server_s"],
                     "input_tokens": r["input_tokens"], "peak_gpu_mib": g.peak_mib,
                     "missing_answers": sum(1 for p in r["probs"] if p is None)})
    return {
        "n_questions": len(w["questions"]),
        "runs": runs,
        "est_full_s": sum(x["wall_s"] for x in runs) / len(runs),
        "peak_gpu_mib": max((x["peak_gpu_mib"] or 0) for x in runs) or None,
    }


def _scale_jev(c, name, cap):
    w = common.scale_workload(name, 1)
    state, qs = w["state"], w["questions"]
    n = len(qs)
    k = max(3, min(n, cap or n))
    m0 = c.metrics()
    with common.GpuMonitor() as g:
        cold = c.noul(state, [qs[0]], concurrency=1)   # first request with this state
        warm = c.noul(state, [qs[1]], concurrency=1)   # same state, new question: fast only if the prefix is reused
        rest = c.noul(state, qs[2:k])                  # the remaining questions, sent concurrently
    m1 = c.metrics()
    prefix = None
    if m0 is not None and m1 is not None:
        prefix = {key: m1[key] - m0.get(key, 0.0) for key in m1}
    return {
        "n_questions": n,
        "questions_sent": k,
        "state_tokens": cold["input_tokens"],
        "cold_s": cold["wall_s"],
        "warm_s": warm["wall_s"],
        "warm_over_cold": warm["wall_s"] / cold["wall_s"] if cold["wall_s"] else None,
        "batch_wall_s": rest["wall_s"],
        "batch_questions": k - 2,
        "est_full_s": cold["wall_s"] + warm["wall_s"] + rest["wall_s"] * (n - 2) / (k - 2),
        "extrapolated": k < n,
        "input_tokens": cold["input_tokens"] + warm["input_tokens"] + rest["input_tokens"],
        "peak_gpu_mib": g.peak_mib,
        "prefix_cache_metrics_delta": prefix,
    }


def t_scale(c, args):
    out = {}
    for name in args.scale_workloads.split(","):
        try:
            res = _scale_kev(c, name) if args.model == "kev" else _scale_jev(c, name, args.question_cap)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        out[name] = res
        if "error" in res:
            print(f"  {name:12s} ERROR {res['error'][:300]}")
        else:
            line = f"  {name:12s} {res['n_questions']:5d} questions  est. {res['est_full_s']:7.2f} s  peak GPU {res['peak_gpu_mib']} MiB"
            if args.model == "jev27b":
                line += (f"  state {res['state_tokens']} tok  cold {res['cold_s']:.3f} s  warm {res['warm_s']:.3f} s"
                         + ("  (extrapolated)" if res["extrapolated"] else ""))
            print(line, flush=True)
    return {"workloads": out}


TEST_FUNCS = {"smoke": t_smoke, "needle": t_needle, "position": t_position,
              "consistency": t_consistency, "scale": t_scale}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["kev", "jev27b"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--tests", default=",".join(ALL_TESTS))
    ap.add_argument("--scale-workloads", default=",".join(common.SCALE_WORKLOADS))
    ap.add_argument("--question-cap", type=int, default=None,
                    help="JEV-27B: send at most this many questions per scale workload and extrapolate the rest")
    ap.add_argument("--concurrency", type=int, default=32, help="JEV-27B: requests in flight")
    ap.add_argument("--timeout", type=float, default=1800, help="per HTTP request, seconds")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for the server to come up")
    ap.add_argument("--server-pid", type=int, default=None, help="stop waiting early if this process exits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    c = clients.make(args.model, args.url, timeout=args.timeout, concurrency=args.concurrency)
    print(f"Waiting for {args.model} at {args.url} ...", flush=True)
    info = wait_ready(c, args.wait, args.server_pid)
    print("Server ready.", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.out, f"{args.model}_{args.tag}_{stamp}.json")
    results = {"model": args.model, "tag": args.tag, "url": args.url, "started": stamp,
               "host": socket.gethostname(), "gpu": common.gpu_info(), "server_info": info,
               "args": vars(args), "tests": {}}

    def save():
        with open(path, "w") as f:
            json.dump(results, f, indent=1)

    save()
    for name in args.tests.split(","):
        print(f"\n== {name}", flush=True)
        t = time.perf_counter()
        try:
            out = TEST_FUNCS[name](c, args)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
            print(f"  ERROR {out['error'][:500]}")
        out["test_wall_s"] = round(time.perf_counter() - t, 2)
        results["tests"][name] = out
        save()
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
