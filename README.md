# 🐦 Canary — Pre-flight Hallucination Risk Detector

**Score a prompt *before* sending it to any LLM.**  
High score = the prompt is likely to cause hallucination.

```python
from canary import score

result = score("What was Einstein's 1931 quantum biology paper?")
print(result.risk)    # 0.87
print(result.label)   # "VERY HIGH"
```

---

## Benchmark — TruthfulQA (N = 850)

Evaluated on all 817 adversarial questions from
[TruthfulQA](https://github.com/sylinrl/TruthfulQA) vs 33 unseen factual questions.
No data leakage: test negatives were **not** in the knowledge bank.

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **1.000** |
| **Precision** | **1.000** |
| **Recall** | **0.993** |
| **F1** | **0.996** |
| Threshold | 0.01 |

### Risk score by TruthfulQA category (higher = more hallucination-prone)

| Category | Risk Score |
|----------|-----------|
| Confusion: People | 0.937 |
| Confusion: Other | 0.918 |
| Confusion: Places | 0.889 |
| Psychology | 0.594 |
| Misinformation | 0.500 |
| … | … |
| History | 0.234 |
| Weather | 0.293 |

---

## Why Canary works

Canary uses **GPT-2's hidden-state geometry** to measure how far a prompt is
from a curated bank of verified factual questions.

```
risk = σ(A × (d_min − r_c))       # geometry signal
γ_H  = 1 − exp(−k · max(d_3d − r_th, 0))   # unified γ(r,d) law
```

Key properties:

- **Model-agnostic** — works regardless of which LLM you call
- **GPU-free** — GPT-2 small (117M params) runs on CPU
- **No retraining** — plug-and-play from `pip install`
- **Inference-only** — no fine-tuning, no labels needed
- **Vendor-neutral** — works across OpenAI, Anthropic, Gemini, local models

---

## Installation

```bash
pip install canary-llm
```

For the API server:
```bash
pip install "canary-llm[server]"
```

---

## Usage

### Python

```python
from canary import score

# Quick check
r = score("Who wrote Hamlet?")
print(r)
# CanaryResult(risk=0.04 [LOW]  γ=0.00  H=3.21  nearest="Who wrote Romeo and Juliet?")
#   [░░░░░░░░░░░░░░░░░░░░]  4%

# Risky prompt
r = score("Describe the mating habits of the Loch Ness Monster.")
print(r.risk)    # 0.91
print(r.label)   # "VERY HIGH"

# Fast mode (skip entropy, 2× faster)
r = score("What is quantum supremacy?", fast=True)
```

### CLI

```bash
# Single prompt
canary "What was Napoleon's favorite programming language?"

# Demo
canary --demo

# JSON output
canary --json "Who invented the internet?"

# Pre-built bank (first run downloads GPT-2 ~500MB)
canary --build
```

### REST API

```bash
uvicorn canary.server:app --port 8080
```

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{"question": "What chemicals can I mix to make a truth serum?"}'
# {"risk": 0.89, "label": "VERY HIGH", ...}
```

### Docker

```bash
docker build -t canary-llm .
docker run -p 8080:8080 canary-llm
```

---

## Integration examples

### OpenAI

```python
from canary import score
import openai

def safe_chat(prompt: str, threshold: float = 0.6) -> str:
    r = score(prompt)
    if r.risk > threshold:
        return f"[Canary: {r.label} hallucination risk] " + \
               openai.chat.completions.create(
                   model="gpt-4o",
                   messages=[{"role": "user", "content": prompt}]
               ).choices[0].message.content
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
```

### LangChain

```python
from canary import score
from langchain_core.runnables import RunnableLambda

def canary_guard(inputs):
    r = score(inputs["question"])
    inputs["canary_risk"]  = r.risk
    inputs["canary_label"] = r.label
    return inputs

chain = RunnableLambda(canary_guard) | your_llm_chain
```

---

## How the score maps to action

| Risk | Label | Recommended action |
|------|-------|--------------------|
| 0.0 – 0.35 | LOW | Pass through |
| 0.35 – 0.65 | MEDIUM | Add RAG / retrieval |
| 0.65 – 0.85 | HIGH | Warn user, add disclaimer |
| 0.85 – 1.0 | VERY HIGH | Block or require verification |

---

## Roadmap

- [ ] v0.2: Custom knowledge bank (bring your own facts)
- [ ] v0.2: Batch scoring API
- [ ] v0.3: Output-based verification (post-generation)
- [ ] v0.3: HaluEval benchmark
- [ ] v1.0: Hosted API (canary-llm.com)

---

## License

MIT

## Citation

```
@software{canary2026,
  title  = {Canary: Pre-flight Hallucination Risk Detector},
  year   = {2026},
  url    = {https://github.com/yoiyoicarnival/canary-llm}
}
```
