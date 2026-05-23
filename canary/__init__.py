"""
Canary — Pre-flight Hallucination Risk Detector

Scores a prompt *before* sending it to any LLM.
High score = the prompt is likely to cause hallucination.

Usage:
    from canary import score
    result = score("What was Einstein's 1931 quantum biology paper?")
    print(result.risk)   # 0.0 – 1.0

    # With answer (post-generation cross-check):
    result = score(question, answer=llm_response)
"""
from .scorer import score, build_bank, CanaryResult

__version__ = "0.1.0"
__all__ = ["score", "build_bank", "CanaryResult"]
