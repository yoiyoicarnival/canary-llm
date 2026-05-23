"""
Canary — Pre-RAG Routing Layer for LLM Applications

Lightweight knowledge boundary estimator. Routes questions outside your
knowledge domain to RAG/search before they reach the LLM.

    import canary

    # Routing API (recommended)
    r = canary.route("What is our refund policy?")
    # RouteResult(✓ DIRECT_ANSWER  confidence=0.89 ...)

    r = canary.route("What did the CEO say last Tuesday?")
    # RouteResult(⟳ USE_RAG  confidence=0.94 ...)

    # Use your own knowledge bank
    canary.load_bank(["Our return policy is 30 days.", ...])

    # Raw risk score
    r = canary.score("What was Einstein's 1931 quantum biology paper?")
    print(r.risk)   # 0.80
"""
from .scorer import score, route, load_bank, build_bank, CanaryResult, RouteResult

__version__ = "0.1.0"
__all__ = ["score", "route", "load_bank", "build_bank", "CanaryResult", "RouteResult"]
