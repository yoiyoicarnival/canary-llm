#!/usr/bin/env python3
"""canary CLI — pre-RAG routing layer"""
import argparse, sys, json


def main():
    p = argparse.ArgumentParser(
        prog="canary",
        description="Pre-RAG routing layer. Decides whether a question needs retrieval.",
    )
    p.add_argument("question", nargs="?", help="Question to evaluate")
    p.add_argument("--route",     action="store_true", help="Show routing decision (default when question given)")
    p.add_argument("--score",     action="store_true", help="Show raw risk score instead of routing")
    p.add_argument("--threshold", type=float, default=0.35, metavar="T",
                   help="Routing threshold (default 0.35)")
    p.add_argument("--fast",   action="store_true", help="Skip entropy, 2× faster")
    p.add_argument("--json",   action="store_true", help="JSON output")
    p.add_argument("--build",  action="store_true", help="Rebuild knowledge bank")
    p.add_argument("--demo",   action="store_true", help="Run routing demo")
    args = p.parse_args()

    if args.build:
        from canary import build_bank
        build_bank(force=True)
        print("Bank rebuilt.")
        return

    if args.demo or not args.question:
        _demo(args.fast, args.json, args.threshold)
        return

    if args.score:
        from canary import score
        r = score(args.question, fast=args.fast)
        if args.json:
            print(json.dumps({"risk": round(r.risk, 4), "label": r.label,
                               "d_min": round(r.d_min, 2), "entropy": round(r.entropy, 3),
                               "nearest_q": r.nearest_q}))
        else:
            print(r)
    else:
        from canary import route
        r = route(args.question, threshold=args.threshold, fast=args.fast)
        if args.json:
            print(json.dumps({"action": r.action, "confidence": round(r.confidence, 4),
                               "risk": round(r.risk, 4), "nearest_q": r.nearest_q}))
        else:
            icon  = "\033[92m✓\033[0m" if r.action == "DIRECT_ANSWER" else "\033[93m⟳\033[0m"
            color = "\033[92m" if r.action == "DIRECT_ANSWER" else "\033[93m"
            reset = "\033[0m"
            print(f"{icon} {color}{r.action}{reset}  "
                  f"confidence={r.confidence:.2f}  risk={r.risk:.2f}")
            print(f"  nearest: \"{r.nearest_q[:60]}\"")


def _demo(fast: bool, as_json: bool, threshold: float):
    from canary import route
    cases = [
        ("What is the capital of France?",                      "known"),
        ("Who discovered penicillin?",                          "known"),
        ("In what year did World War II end?",                  "known"),
        ("What is the boiling point of water in Celsius?",     "known"),
        ("What was Einstein's 1931 quantum biology paper?",    "unknown"),
        ("Describe the grammar of the Elvish language Sindarin.", "unknown"),
        ("What is the cuisine of the lost city of Atlantis?",  "unknown"),
        ("Who won the 2031 Nobel Peace Prize?",                "unknown"),
    ]
    print("\n  CANARY — Pre-RAG Routing Demo")
    print("  " + "─" * 68)
    print(f"  {'Question':50s}  {'Action':14s}  Conf")
    print("  " + "─" * 68)
    results = []
    for q, tier in cases:
        r = route(q, threshold=threshold, fast=fast)
        short = q[:48] + ("…" if len(q) > 48 else "")
        if r.action == "DIRECT_ANSWER":
            color, icon = "\033[92m", "✓"
        else:
            color, icon = "\033[93m", "⟳"
        reset = "\033[0m"
        print(f"  {short:50s}  {color}{icon} {r.action:13s}{reset}  {r.confidence:.2f}")
        results.append({"question": q, "tier": tier, "action": r.action,
                         "confidence": r.confidence, "risk": r.risk})
    print("  " + "─" * 68)
    direct = sum(1 for x in results if x["action"] == "DIRECT_ANSWER" and x["tier"] == "known")
    rag    = sum(1 for x in results if x["action"] == "USE_RAG"        and x["tier"] == "unknown")
    print(f"\n  Known → DIRECT:  {direct}/{sum(1 for x in results if x['tier']=='known')}")
    print(f"  Unknown → RAG:   {rag}/{sum(1 for x in results if x['tier']=='unknown')}")
    if as_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
