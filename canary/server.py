"""
FastAPI server — drop-in middleware for LLM pipelines.

    uvicorn canary.server:app --host 0.0.0.0 --port 8080

POST /score
  body: {"question": "...", "answer": ""}
  returns: {"risk": 0.72, "label": "HIGH", "gamma_h": 0.65, ...}

GET  /health   → {"status": "ok"}
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Canary — Hallucination Risk API",
    description="Pre-flight hallucination risk scorer for LLM prompts.",
    version="0.1.0",
)

_scorer_loaded = False


def _ensure():
    global _scorer_loaded
    if _scorer_loaded:
        return
    from canary.scorer import _ensure_model, _ensure_bank, _ensure_pca3d
    _ensure_model()
    _ensure_bank()
    _ensure_pca3d()
    _scorer_loaded = True


class ScoreRequest(BaseModel):
    question: str
    answer:   Optional[str] = ""
    fast:     Optional[bool] = False


class ScoreResponse(BaseModel):
    risk:      float
    label:     str
    gamma_h:   float
    d_min:     float
    entropy:   float
    nearest_q: str


@app.on_event("startup")
async def startup():
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/score", response_model=ScoreResponse)
def score_endpoint(req: ScoreRequest):
    _ensure()
    from canary import score
    r = score(req.question, req.answer or "", fast=req.fast or False)
    return ScoreResponse(
        risk      = round(r.risk, 4),
        label     = r.label,
        gamma_h   = round(r.gamma_h, 4),
        d_min     = round(r.d_min, 2),
        entropy   = round(r.entropy, 3),
        nearest_q = r.nearest_q,
    )
