#!/usr/bin/env python3
"""canary CLI — pre-flight hallucination risk scorer"""
import argparse, sys, json


def main():
    p = argparse.ArgumentParser(
        prog="canary",
        description="Pre-flight hallucination risk scorer for LLM prompts.",
    )
    p.add_argument("question", nargs="?", help="Prompt to score")
    p.add_argument("--answer",  default="", metavar="TEXT", help="LLM answer (optional)")
    p.add_argument("--fast",    action="store_true", help="Skip entropy computation")
    p.add_argument("--json",    action="store_true", help="JSON output")
    p.add_argument("--build",   action="store_true", help="Rebuild knowledge bank")
    p.add_argument("--demo",    action="store_true", help="Run demo on 8 example prompts")
    args = p.parse_args()

    if args.build:
        from canary import build_bank
        build_bank(force=True)
        print("Bank rebuilt.")
        return

    if args.demo or not args.question:
        _demo(args.fast, args.json)
        return

    from canary import score
    r = score(args.question, args.answer, fast=args.fast)

    if args.json:
        print(json.dumps({
            "risk": round(r.risk, 4),
            "label": r.label,
            "gamma_h": round(r.gamma_h, 4),
            "d_min": round(r.d_min, 2),
            "entropy": round(r.entropy, 3),
            "nearest_q": r.nearest_q,
        }))
    else:
        print(r)


def _demo(fast: bool, as_json: bool):
    from canary import score
    prompts = [
        ("What is the capital of France?",                   "safe"),
        ("Who discovered penicillin?",                       "safe"),
        ("In what year did World War II end?",               "safe"),
        ("What was Einstein's 1931 quantum biology paper?",  "risky"),
        ("Describe the grammar of the Elvish language Sindarin.", "risky"),
        ("What is the cuisine of the lost city of Atlantis?","risky"),
        ("What is the most effective treatment for cancer?", "borderline"),
        ("Who won the 2031 Nobel Peace Prize?",              "risky"),
    ]
    print("\n  CANARY — Hallucination Risk Demo")
    print("  " + "─" * 60)
    print(f"  {'Prompt':44s}  {'Risk':>5}  {'Label'}")
    print("  " + "─" * 60)
    results = []
    for q, tier in prompts:
        r = score(q, fast=fast)
        bar   = "█" * int(r.risk * 16) + "░" * (16 - int(r.risk * 16))
        short = q[:42] + ("…" if len(q) > 42 else "")
        color = "\033[91m" if r.risk > 0.65 else ("\033[93m" if r.risk > 0.35 else "\033[92m")
        reset = "\033[0m"
        print(f"  {short:44s}  {color}{r.risk*100:>4.0f}%  {r.label}{reset}")
        results.append({"question": q, "tier": tier, "risk": r.risk, "label": r.label})
    print("  " + "─" * 60)
    if as_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
