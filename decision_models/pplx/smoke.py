"""Cluster smoke test for pplx-decider-v1.2 with forked scoring. Writes results/pplx_smoke_<time>.json.

Steps (each prints PASS, FAIL or INFO; the exit code is 1 if any step fails):
  load       load the checkpoint and split its layers across the visible GPUs; weights per GPU
  kernels    forked scoring (flash-linear-attention + varlen FlashAttention) against the unmodified Transformers
             model run with the same attention pattern, on 12 questions: must agree to bf16 noise
  smoke      two obvious questions: P(yes) high for the relevant one, low for the other
  needle     6 real questions, gold page hidden among 200 titles: rank of the gold page
  isolation  the same questions asked in a different subset and order: answers must not change
  timing     stage-shaped workloads, each run twice: the first call includes kernel compilation for new shapes,
             the second is the steady-state speed used for the per-query estimate; tokens/s, peak GPU memory
  reference  with --reference: this run's answers against an earlier run's (e.g. FP8 against bf16) on the same
             deterministic workloads: |dp| on the kernels, smoke and needle questions, needle ranks and top 5

Usage: python pplx/smoke.py --checkpoint $PPLX_CKPT [--devices cuda:0,cuda:1] [--fp8] [--reference old.json]
                            [--skip timing]
"""
import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from common import NEEDLES, needle_workload, scale_workload  # noqa: E402
from scorer import Scorer  # noqa: E402

# Paper's per-query workload on Natural Questions: 1,000 clusters in 8 requests of ~125, ~17,500 titles in
# requests of <= 20k tokens (~7 x 2,500), ~230 paragraphs in one request.
TIMING = ["stage1_125", "stage2_625", "stage2_2500", "stage3_230"]
PER_QUERY = {"stage1_125": 8, "stage2_2500": 7, "stage3_230": 1}


def gold_rank(margins, gold):
    return 1 + sum(m > margins[gold] for m in margins)


def peak_gib(scorer):
    return {str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in scorer.devices if d.type == "cuda"}


def reset_peaks(scorer):
    for d in scorer.devices:
        if d.type == "cuda":
            torch.cuda.reset_peak_memory_stats(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--devices", default=None, help="comma-separated; default: every visible GPU")
    ap.add_argument("--budget", type=int, default=16384, help="tokens per forward pass for the question branches")
    ap.add_argument("--fp8", action="store_true", help="FP8 projections (Ada/Hopper); fits one 48 GB GPU")
    ap.add_argument("--reference", default=None, help="an earlier results JSON to compare answers against")
    ap.add_argument("--skip", default="", help="comma-separated steps to skip")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None

    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "fp8": args.fp8,
           "reference": args.reference, "steps": {}}
    path = Path(args.out) / f"pplx_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = []

    def report(step, ok, summary, **data):
        status = "INFO" if ok is None else "PASS" if ok else "FAIL"
        if ok is False:
            failed.append(step)
        print(f"[{status}] {step}: {summary}", flush=True)
        out["steps"][step] = {"status": status, "summary": summary, **data}
        path.write_text(json.dumps(out, indent=1))

    # ---- load
    t0 = time.perf_counter()
    try:
        devices = args.devices.split(",") if args.devices else None
        s = Scorer(args.checkpoint, devices=devices, budget=args.budget, fp8=args.fp8)
    except Exception:
        report("load", False, "could not load the model", error=traceback.format_exc())
        print(out["steps"]["load"]["error"])
        return 1
    weights = {str(d): round(torch.cuda.memory_allocated(d) / 2**30, 1) for d in s.devices if d.type == "cuda"}
    precision = f"FP8 {s.fp8_mode}, {s.fp8_layers} projections" if args.fp8 else "bf16"
    report("load", True, f"{time.perf_counter() - t0:.0f} s; {precision}; layers per device {s.split_summary}; "
                         f"weights GiB {weights}; kernel notes: {s.kernels.notes or 'none'}",
           layers=s.split_summary, weights_gib=weights, notes=s.kernels.notes, precision=precision)

    def step(name, fn):
        if name in skip:
            return
        try:
            fn()
        except Exception:
            report(name, False, "raised an exception", error=traceback.format_exc())
            print(out["steps"][name]["error"], flush=True)

    # ---- kernels: forked vs Transformers with the same attention pattern (state does not see the question)
    def kernels():
        w = needle_workload(NEEDLES[1], 50, seed=7)
        qs = w["questions"][:12]
        f = s.score(w["state"], qs, "forked")
        ref = s.score(w["state"], qs, "official_masked")
        dp = [abs(a - b) for a, b in zip(f["p"], ref["p"])]
        dm = [abs(a - b) for a, b in zip(f["margin"], ref["margin"])]
        # FP8 turns round-off differences between the two code paths into occasional rounding flips: looser bound.
        mean_max, max_max = (0.02, 0.1) if args.fp8 else (0.01, 0.05)
        ok = max(dp) < max_max and statistics.mean(dp) < mean_max
        report("kernels", ok, f"vs Transformers: mean |dp| {statistics.mean(dp):.4f}, max |dp| {max(dp):.4f}, "
                              f"max |d margin| {max(dm):.3f} (want mean < {mean_max}, max < {max_max})",
               mean_abs_dp=statistics.mean(dp), max_abs_dp=max(dp), max_abs_dmargin=max(dm),
               forked=f["p"], reference=ref["p"])

    # ---- smoke
    def smoke():
        state = ("Customer message: I was charged twice for my order #4471 last week. Please refund the duplicate "
                 "charge. The package itself arrived on time and in good condition.")
        r = s.score(state, ["Is the customer asking for a refund?",
                            "Is the customer complaining about a late delivery?"], "forked")
        ok = r["p"][0] > 0.5 > r["p"][1]
        report("smoke", ok, f"P(refund) {r['p'][0]:.3f} (want > 0.5), P(late delivery) {r['p'][1]:.3f} (want < 0.5)",
               p=r["p"], margin=r["margin"])

    # ---- needle
    def needle():
        rows = []
        for i, nd in enumerate(NEEDLES):
            w = needle_workload(nd, 200, seed=i)
            r = s.score(w["state"], w["questions"], "forked")
            m = r["margin"]
            top = sorted(range(len(m)), key=lambda j: -m[j])[:5]
            rows.append({"query": nd["query"], "gold_rank": gold_rank(m, w["gold_index"]),
                         "gold_p": r["p"][w["gold_index"]], "top5": [w["titles"][j] for j in top],
                         "seconds": r["seconds"], "tokens": r["tokens"], "p": r["p"]})
            print(f"    {nd['query']!r}: gold rank {rows[-1]['gold_rank']}, P(gold) {rows[-1]['gold_p']:.3f}, "
                  f"top 5 {rows[-1]['top5']}", flush=True)
        top5 = sum(r["gold_rank"] <= 5 for r in rows)
        report("needle", top5 >= 5, f"gold page in the top 5 for {top5}/6 queries "
                                    f"(ranks {[r['gold_rank'] for r in rows]})", queries=rows)

    # ---- isolation: a subset in another order must give the same answers
    def isolation():
        w = needle_workload(NEEDLES[2], 200, seed=11)
        full = s.score(w["state"], w["questions"], "forked")
        idx = list(range(0, 200, 7))[::-1]
        sub = s.score(w["state"], [w["questions"][i] for i in idx], "forked")
        dp = [abs(full["p"][i] - p) for i, p in zip(idx, sub["p"])]
        ok = max(dp) < 0.02
        report("isolation", ok, f"{len(idx)} questions re-asked in another subset and order: "
                                f"mean |dp| {statistics.mean(dp):.4f}, max |dp| {max(dp):.4f} (want ~0, bf16 noise)",
               mean_abs_dp=statistics.mean(dp), max_abs_dp=max(dp))

    # ---- timing
    def timing():
        rows = {}
        for name in TIMING:
            w = scale_workload(name, seed=1)
            reset_peaks(s)
            first = s.score(w["state"], w["questions"], "forked")
            r = s.score(w["state"], w["questions"], "forked")
            rows[name] = {"seconds": round(r["seconds"], 2), "first_call_seconds": round(first["seconds"], 2),
                          "tokens": r["tokens"], "tokens_per_s": round(r["tokens"] / r["seconds"]),
                          "peak_gib": peak_gib(s)}
            print(f"    {name}: {rows[name]}", flush=True)
        est = sum(rows[k]["seconds"] * n for k, n in PER_QUERY.items())
        gpus = len(set(s.layer_devices))
        report("timing", None, f"estimated {est:.0f} s per DecisionRAG query on {gpus} GPU(s), steady state "
                               f"(8 x stage1_125 + 7 x stage2_2500 + stage3_230; paper: 1.7 s with Jev)",
               workloads=rows, est_seconds_per_query=est, gpus=gpus)

    # ---- reference: the same deterministic workloads in an earlier run (e.g. bf16 on A6000s)
    def compare():
        ref, now = reference["steps"], out["steps"]
        pairs = {}  # name -> (reference p's, this run's p's)
        if "kernels" in ref and "kernels" in now:
            pairs["kernels"] = (ref["kernels"]["forked"], now["kernels"]["forked"])
        if "smoke" in ref and "smoke" in now:
            pairs["smoke"] = (ref["smoke"]["p"], now["smoke"]["p"])
        rows = []
        if "needle" in ref and "needle" in now:
            for a, b in zip(ref["needle"]["queries"], now["needle"]["queries"]):
                if "p" in a:  # all 200 answers (runs from this version on)
                    pairs[a["query"]] = (a["p"], b["p"])
                else:         # older runs kept only the gold page's answer
                    pairs[a["query"]] = ([a["gold_p"]], [b["gold_p"]])
                rows.append({"query": a["query"], "gold_rank": (a["gold_rank"], b["gold_rank"]),
                             "top5_overlap": len(set(a["top5"]) & set(b["top5"]))})
        dp = [abs(x - y) for a, b in pairs.values() for x, y in zip(a, b)]
        if not dp:
            raise ValueError("no steps in common with the reference run")
        same_gold = sum(r["gold_rank"][0] == r["gold_rank"][1] for r in rows)
        overlap = statistics.mean(r["top5_overlap"] for r in rows) if rows else float("nan")
        ok = statistics.mean(dp) < 0.02 and max(dp) < 0.15 and same_gold >= len(rows) - 1
        report("reference", ok, f"vs {Path(args.reference).name}: {len(dp)} answers, mean |dp| "
                                f"{statistics.mean(dp):.4f}, max |dp| {max(dp):.4f}; needle gold rank unchanged "
                                f"{same_gold}/{len(rows)}, top-5 overlap {overlap:.1f}/5 "
                                f"(want mean < 0.02, max < 0.15)",
               mean_abs_dp=statistics.mean(dp), max_abs_dp=max(dp), answers=len(dp), needle=rows,
               per_set={k: max(abs(x - y) for x, y in zip(a, b)) for k, (a, b) in pairs.items()})

    # Kernel compilation for a few shapes, not timed (packs of 1 and of several sequences, odd and even lengths).
    s.score("Warm-up state.", ["Is this a warm-up?"], "forked")
    s.score("Warm-up state number two, a little longer. " * 20, [f"Is this warm-up {i}?" for i in range(37)],
            "forked")
    steps = [("kernels", kernels), ("smoke", smoke), ("needle", needle), ("isolation", isolation),
             ("timing", timing)]
    if reference is not None:
        steps.append(("reference", compare))
    for name, fn in steps:
        step(name, fn)

    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["failed"] = failed
    path.write_text(json.dumps(out, indent=1))
    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'All checks passed.'}  Results: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
