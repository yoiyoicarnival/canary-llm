# Canary-LLM

[![PyPI version](https://badge.fury.io/py/canary-llm.svg)](https://pypi.org/project/canary-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Lightweight pre-RAG routing for LLM applications.**  
Detects out-of-domain queries before retrieval/tool use.  
No GPU. Sub-100ms. Model-agnostic.

```
User query
    │
    ▼
┌─────────┐
│  Canary │  ~10–50ms, CPU-only
└────┬────┘
     │
     ├─── DIRECT_ANSWER ──→  LLM
     │
     └─── USE_RAG ──────→  retrieval / search / tools → LLM
```

```python
import canary

r = canary.route("What is our return policy?")
r.action      # "DIRECT_ANSWER"
r.confidence  # 0.91

r = canary.route("What did our CEO say last Tuesday?")
r.action      # "USE_RAG"
r.confidence  # 0.88
```

---

## Install

```bash
pip install canary-llm
```

First run downloads GPT-2 (~500MB, one-time).

---

## Quickstart

```python
import canary

# ── routing (recommended) ─────────────────────────────────────
r = canary.route(question)

if r.action == "USE_RAG":
    context = retriever.get(question)
    answer  = llm(question, context)
else:
    answer  = llm(question)

# ── with your own knowledge base ─────────────────────────────
canary.load_bank([
    "Our return policy allows returns within 30 days.",
    "We ship to the US, EU, and Japan.",
    "Enterprise plans start at $499/month.",
    # ... add sentences from your docs/FAQ/wiki
])

canary.route("Do you ship to Brazil?")
# RouteResult(action='USE_RAG', confidence=0.88, ...)

canary.route("How do I return a product?")
# RouteResult(action='DIRECT_ANSWER', confidence=0.93, ...)
```

---

## Result object

```python
r = canary.route("Who was the 44th president?")

r.action      # "DIRECT_ANSWER" | "USE_RAG"
r.confidence  # how certain the routing is   (0 – 1)
r.risk        # raw out-of-domain score       (0 – 1)
r.nearest_q   # closest question in your bank
```

---

## Tune the threshold

```python
# More conservative — minimize missed unknowns
canary.route(q, threshold=0.25)

# More aggressive — maximize RAG savings
canary.route(q, threshold=0.50)

# Fast mode — 2× faster, slightly less accurate
canary.route(q, fast=True)
```

---

## Benchmark — RAG call reduction

Tested on TruthfulQA (817 out-of-domain questions) + 20 factual questions, threshold 0.35:

| | Result |
|--|--|
| Out-of-domain → USE_RAG | **99.6%** |
| In-domain → DIRECT_ANSWER | **90.0%** |

**Estimated savings by workload:**

| Workload (known : unknown) | RAG calls saved |
|---------------------------|-----------------|
| 70% : 30% | **63%** |
| 80% : 20% | **72%** |
| 90% : 10% | **81%** |

---

## Integrations

### LangChain

```python
from langchain_core.runnables import RunnableLambda
import canary

def router(inputs):
    r = canary.route(inputs["question"])
    inputs["use_rag"] = (r.action == "USE_RAG")
    return inputs

chain = RunnableLambda(router) | your_chain
```

### OpenAI

```python
import openai, canary

def chat(question):
    r = canary.route(question)
    msgs = [{"role": "user", "content": question}]
    if r.action == "USE_RAG":
        ctx = retrieve(question)
        msgs[0]["content"] = f"Context:\n{ctx}\n\n{question}"
    return openai.chat.completions.create(model="gpt-4o", messages=msgs) \
                                  .choices[0].message.content
```

### REST API

```bash
pip install "canary-llm[server]"
uvicorn canary.server:app --port 8080
```

```bash
curl -X POST http://localhost:8080/score \
  -d '{"question": "What was the Treaty of Atlantis?"}' \
  -H "Content-Type: application/json"
# {"action": "USE_RAG", "confidence": 0.91, ...}
```

---

## CLI

```bash
canary "Who invented the telephone?"
# ✓ DIRECT_ANSWER  confidence=0.89

canary "What happened at the board meeting yesterday?"
# ⟳ USE_RAG  confidence=0.94

canary --demo        # run routing demo on 8 example queries
canary --json "..."  # JSON output
canary --build       # rebuild bank
```

---

## How it works

Canary measures how far a query sits from your knowledge bank in GPT-2's embedding space. Far = unknown = route to RAG.

- Uses GPT-2 layer-11 hidden states (768-dim)
- Combines geometric distance + generation entropy
- No fine-tuning, no labels, plug-and-play

---

## Roadmap

- [ ] v0.2 — `route_batch()`, auto-grow bank
- [ ] v0.3 — output-side verification
- [ ] v1.0 — hosted API

---

## License

MIT
