#!/usr/bin/env python3
"""Compare two bench.py result files side by side.

    python3 compare.py results/kev_default_*.json results/jev27b_prefix_on_*.json

Needle: rank agreement between the two models on identical inputs (Spearman, same top page, top-10 overlap).
Position: gold-page P(yes) at the start, middle and end of 1,250 titles.
Scale: estimated time and peak GPU memory per workload.
"""
import json
import sys


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def top(probs, k):
    return set(sorted(range(len(probs)), key=lambda i: -(probs[i] if probs[i] is not None else -1.0))[:k])


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    la, lb = f"{A['model']}:{A['tag']}", f"{B['model']}:{B['tag']}"
    print(f"A = {la}  ({A.get('gpu')})\nB = {lb}  ({B.get('gpu')})")

    na = {q["query"]: q for q in A["tests"].get("needle", {}).get("queries", [])}
    nb = {q["query"]: q for q in B["tests"].get("needle", {}).get("queries", [])}
    common = [q for q in na if q in nb]
    if common:
        print("\nNeedle (200 titles, identical inputs)")
        print(f"  {'query':44s} {'gold A':>6s} {'gold B':>6s} {'rho':>6s} {'same top':>8s} {'top10 overlap':>13s}")
        rhos = []
        for q in common:
            a, b = na[q], nb[q]
            rho = spearman(a["probs"], b["probs"])
            rhos.append(rho)
            same = top(a["probs"], 1) == top(b["probs"], 1)
            ov = len(top(a["probs"], 10) & top(b["probs"], 10))
            print(f"  {q[:44]:44s} {fmt(a['gold_rank']):>6s} {fmt(b['gold_rank']):>6s} {fmt(rho, 2):>6s} "
                  f"{str(same):>8s} {ov:>10d}/10")
        valid = [r for r in rhos if r is not None]
        if valid:
            print(f"  mean Spearman: {sum(valid) / len(valid):.2f}")

    pa_rows = A["tests"].get("position", {}).get("rows", [])
    pb_rows = {(r["query"], r["gold_pos"]): r for r in B["tests"].get("position", {}).get("rows", [])}
    if pa_rows:
        print("\nPosition (gold P(yes) by position among 1,250 titles)")
        for r in pa_rows:
            rb = pb_rows.get((r["query"], r["gold_pos"]), {})
            print(f"  {r['query'][:40]:40s} pos {r['gold_pos']:4d}:  A {fmt(r['gold_p'])}  B {fmt(rb.get('gold_p'))}")

    sa = A["tests"].get("scale", {}).get("workloads", {})
    sb = B["tests"].get("scale", {}).get("workloads", {})
    names = list(dict.fromkeys(list(sa) + list(sb)))
    if names:
        print("\nScale (estimated seconds for the whole workload / peak GPU MiB)")
        for n in names:
            a, b = sa.get(n, {}), sb.get(n, {})
            def cell(r):
                if "error" in r:
                    return "ERROR"
                if not r:
                    return "-"
                return f"{r['est_full_s']:.2f} s / {r.get('peak_gpu_mib')}" + (" (extrap.)" if r.get("extrapolated") else "")
            print(f"  {n:12s}  A {cell(a):28s}  B {cell(b)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
