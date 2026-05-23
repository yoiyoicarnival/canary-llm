#!/usr/bin/env python3
"""
routing_benchmark.py — RAG routing cost-reduction benchmark

Measures how much Canary reduces RAG invocations while maintaining
coverage of unknown questions.

Metric that matters for product:
  - RAG reduction %: fraction of factual questions correctly routed to DIRECT
  - Unknown coverage %: fraction of unknown questions correctly sent to RAG

Run:
    OMP_NUM_THREADS=1 python3 benchmark/routing_benchmark.py
"""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

# ── Load TruthfulQA questions (from datasets or local cache) ──────────────────

def load_truthfulqa():
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation", split="validation")
        return [row["question"] for row in ds]
    except Exception:
        return []


# ── Simple factual questions (known-domain, should route DIRECT) ──────────────

FACTUAL_KNOWN = [
    "What is the capital of France?",
    "What is the chemical formula for water?",
    "Who invented the telephone?",
    "What year did World War II end?",
    "What is the speed of light in km/s?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the solar system?",
    "How many bones are in the adult human body?",
    "What is the boiling point of water in Celsius?",
    "What does DNA stand for?",
    "What is the capital of Japan?",
    "Who discovered gravity?",
    "What is the square root of 144?",
    "What is the largest ocean?",
    "What is the atomic number of carbon?",
    "Who wrote Romeo and Juliet?",
    "What is the tallest mountain?",
    "What is the most spoken language?",
    "What is the hardest natural substance?",
    "How many planets are in the solar system?",
]


def run_routing_benchmark(threshold: float = 0.35, fast: bool = True):
    import canary

    print(f"\n{'='*60}")
    print(f"Canary Routing Benchmark  (threshold={threshold}, fast={fast})")
    print(f"{'='*60}")

    # ── Part 1: known-domain routing ─────────────────────────────────────────
    print("\n[1/2] Known-domain questions (should → DIRECT_ANSWER)...")
    n_correct_direct = 0
    for q in FACTUAL_KNOWN:
        r = canary.route(q, threshold=threshold, fast=fast)
        if r.action == "DIRECT_ANSWER":
            n_correct_direct += 1

    direct_rate = n_correct_direct / len(FACTUAL_KNOWN)
    print(f"  Routed to DIRECT: {n_correct_direct}/{len(FACTUAL_KNOWN)} ({direct_rate:.1%})")

    # ── Part 2: unknown-domain routing ───────────────────────────────────────
    print("\n[2/2] TruthfulQA questions (should → USE_RAG)...")
    tqa_questions = load_truthfulqa()
    if not tqa_questions:
        print("  [skip] TruthfulQA not available (pip install datasets)")
        rag_rate = None
    else:
        n_correct_rag = 0
        for q in tqa_questions:
            r = canary.route(q, threshold=threshold, fast=fast)
            if r.action == "USE_RAG":
                n_correct_rag += 1
        rag_rate = n_correct_rag / len(tqa_questions)
        print(f"  Routed to USE_RAG: {n_correct_rag}/{len(tqa_questions)} ({rag_rate:.1%})")

    # ── Cost reduction model ──────────────────────────────────────────────────
    print("\n── Cost Reduction Estimate ──────────────────────────────")
    for known_frac in [0.50, 0.70, 0.80, 0.90]:
        unk_frac   = 1.0 - known_frac
        # baseline: RAG on every question
        # with Canary: RAG only on unknown + misrouted known
        if rag_rate is not None:
            rag_calls = known_frac * (1 - direct_rate) + unk_frac * rag_rate
        else:
            rag_calls = known_frac * (1 - direct_rate) + unk_frac * 1.0
        reduction = (1.0 - rag_calls) * 100
        print(f"  {known_frac:.0%} known / {unk_frac:.0%} unknown  "
              f"→ RAG reduction: {reduction:.0f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "threshold": threshold,
        "fast": fast,
        "known_domain": {
            "n": len(FACTUAL_KNOWN),
            "direct_rate": round(direct_rate, 4),
        },
    }
    if rag_rate is not None:
        results["unknown_domain"] = {
            "n": len(tqa_questions),
            "rag_rate": round(rag_rate, 4),
        }
    out = os.path.join(os.path.dirname(__file__), "routing_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")
    return results


if __name__ == "__main__":
    run_routing_benchmark(threshold=0.35, fast=True)
