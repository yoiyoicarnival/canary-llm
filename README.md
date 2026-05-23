# Canary — Pre-RAG Routing Layer for LLM Applications

**Route questions to RAG only when needed. Reduce retrieval costs by up to 72%.**

```python
import canary

r = canary.route("What is the boiling point of water?")
# RouteResult(✓ DIRECT_ANSWER  confidence=0.91 ...)

r = canary.route("What did our CEO announce last Tuesday?")
# RouteResult(⟳ USE_RAG  confidence=0.94 ...)
```

Canary estimates whether a question falls **inside or outside your knowledge boundary** before sending it to an LLM. Questions outside → go to RAG. Questions inside → answer directly.

---

## The Problem

RAG on every query is slow and expensive.

```
User question → [always RAG] → LLM → Answer
                    ↑
              unnecessary 80% of the time
```

Most production workloads are 70–90% answerable from a stable knowledge base (docs, FAQ, product info). Running full retrieval on those wastes latency and money.

---

## The Solution

```
User question
     ↓
  Canary                          ~10–50ms, CPU-only
     ↓
known domain? ──yes──→ LLM (direct answer)
             └──no──→  RAG → LLM
```

---

## Benchmark — RAG Cost Reduction

Evaluated on TruthfulQA (817 unknown-domain questions) + 20 factual questions, default threshold 0.35:

| Metric | Value |
|--------|-------|
| Unknown questions → RAG | **99.6%** (814 / 817) |
| Known questions → DIRECT | **90.0%** (18 / 20) |

### Estimated RAG reduction by workload mix

| Known : Unknown | RAG calls saved |
|-----------------|-----------------|
| 50% : 50% | **45%** |
| 70% : 30% | **63%** |
| 80% : 20% | **72%** |
| 90% : 10% | **81%** |

---

## Installation

```bash
pip install canary-llm
```

For the REST API server:
```bash
pip install "canary-llm[server]"
```

---

## Usage

### Routing (recommended)

```python
import canary

r = canary.route("What is our return policy?")
# r.action      → "DIRECT_ANSWER" or "USE_RAG"
# r.confidence  → certainty of the decision (0–1)
# r.risk        → raw epistemic distance score (0–1)

if r.action == "USE_RAG":
    answer = rag_pipeline(question)
else:
    answer = llm(question)
```

### Custom knowledge bank

Load your own documents so Canary routes questions **outside your specific knowledge domain**:

```python
import canary

canary.load_bank([
    "Our return policy allows returns within 30 days.",
    "We ship to the US, EU, and Japan.",
    "Enterprise plans start at $499/month.",
    # add hundreds of representative sentences from your docs
])

canary.route("Do you ship to Brazil?")
# RouteResult(⟳ USE_RAG  confidence=0.88 ...)

canary.route("How do I return a product?")
# RouteResult(✓ DIRECT_ANSWER  confidence=0.93 ...)
```

### Threshold tuning

```python
# Conservative — minimize missed unknowns
canary.route(question, threshold=0.25)

# Aggressive — maximize RAG savings
canary.route(question, threshold=0.50)

# Fast mode — skip entropy, 2× faster
canary.route(question, fast=True)
```

### Raw risk score

```python
r = canary.score("What was Einstein's 1931 quantum biology paper?")
print(r.risk)    # 0.80
print(r.label)   # "HIGH"
```

---

## Integrations

### LangChain

```python
from langchain_core.runnables import RunnableLambda
import canary

def canary_router(inputs: dict) -> dict:
    r = canary.route(inputs["question"])
    inputs["use_rag"] = (r.action == "USE_RAG")
    inputs["canary_confidence"] = r.confidence
    return inputs

chain = RunnableLambda(canary_router) | your_rag_or_llm_chain
```

### OpenAI

```python
import openai, canary

def smart_chat(question: str) -> str:
    r = canary.route(question)
    if r.action == "USE_RAG":
        context = retrieve(question)
        messages = [{"role": "user", "content": f"Context: {context}\n\n{question}"}]
    else:
        messages = [{"role": "user", "content": question}]
    return openai.chat.completions.create(
        model="gpt-4o", messages=messages
    ).choices[0].message.content
```

### REST API

```bash
uvicorn canary.server:app --port 8080
```

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Treaty of Atlantis?"}'
# {"risk": 0.89, "label": "HIGH", ...}
```

### Docker

```bash
docker build -t canary-llm .
docker run -p 8080:8080 canary-llm
```

---

## How it works

Canary uses **GPT-2's hidden-state geometry** (layer 11, 768-dim) to measure how far a question is from a bank of known-domain examples. Questions far from the bank surface → need retrieval.

```
risk = σ(A × (d_min − r_c))   # geometric distance signal
     + entropy component       # generation uncertainty signal
```

Key properties:

- **Model-agnostic** — works regardless of which LLM you call
- **GPU-free** — GPT-2 small (117M params) runs on CPU in 10–50ms
- **No retraining** — plug-and-play, no labels needed
- **Vendor-neutral** — OpenAI, Anthropic, Gemini, local models
- **Customizable** — `load_bank()` for domain-specific routing

---

## CLI

```bash
# Route a single question
canary --route "Who wrote Hamlet?"

# Demo
canary --demo

# JSON output
canary --json "What is quantum entanglement?"

# Build bank
canary --build
```

---

## Routing decision guide

| Risk score | Action | Meaning |
|------------|--------|---------|
| 0.0 – 0.35 | DIRECT_ANSWER | Question is within known territory |
| 0.35 – 0.65 | USE_RAG | Borderline — retrieval recommended |
| 0.65 – 1.0 | USE_RAG | Outside known territory — retrieve |

---

## Roadmap

- [ ] v0.2: Batch routing API (`canary.route_batch(questions)`)
- [ ] v0.2: Auto-grow bank from answered queries
- [ ] v0.3: Output-side verification (post-generation)
- [ ] v0.3: HaluEval benchmark
- [ ] v1.0: Hosted API

---

## License

MIT

## Citation

```bibtex
@software{canary2026,
  title  = {Canary: Pre-RAG Routing Layer for LLM Applications},
  year   = {2026},
  url    = {https://github.com/yoiyoicarnival/canary-llm}
}
```
