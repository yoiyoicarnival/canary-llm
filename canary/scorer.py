"""
Core scorer.  Lazy-loads GPT-2 on first call (~500 MB, CPU-only).
"""
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import Optional
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ── calibration (GPT-2 small, layer-11, 85-question BUILTIN bank) ──
_A      = 0.1044   # geometry logistic sharpness
_RC     = 45.0     # critical distance r_c  (bank questions cluster d<30; OOD at 46+)
_LAYER  = 11
_K      = 0.405    # γ(r,d) decay  (unified law)
_RTH    = 0.283    # coherence radius in PCA3D

_BANK_DEFAULT = os.path.join(os.path.dirname(__file__), "data", "bank.json")

_state: dict = {}   # lazy cache: model, tokenizer, bank, pca3d, bank_3d


# ── knowledge bank ────────────────────────────────────────────────────────────

BUILTIN_FACTS = [
    # geography
    ("What is the capital of France?",       "Paris"),
    ("What is the capital of Japan?",        "Tokyo"),
    ("What is the capital of Germany?",      "Berlin"),
    ("What is the capital of Italy?",        "Rome"),
    ("What is the capital of Spain?",        "Madrid"),
    ("What is the capital of China?",        "Beijing"),
    ("What is the capital of Russia?",       "Moscow"),
    ("What is the capital of Brazil?",       "Brasilia"),
    ("What is the capital of Australia?",    "Canberra"),
    ("What is the capital of Canada?",       "Ottawa"),
    ("What is the capital of India?",        "New Delhi"),
    ("What is the capital of Mexico?",       "Mexico City"),
    ("What is the capital of South Korea?",  "Seoul"),
    ("What is the capital of Egypt?",        "Cairo"),
    ("What is the capital of Turkey?",       "Ankara"),
    ("What is the capital of Argentina?",    "Buenos Aires"),
    ("What is the capital of South Africa?", "Pretoria"),
    ("What is the capital of Nigeria?",      "Abuja"),
    ("What is the capital of Indonesia?",    "Jakarta"),
    ("What is the capital of Saudi Arabia?", "Riyadh"),
    # science
    ("What is the chemical symbol for gold?",       "Au"),
    ("What is the chemical symbol for silver?",     "Ag"),
    ("What is the chemical symbol for iron?",       "Fe"),
    ("What is the chemical symbol for sodium?",     "Na"),
    ("What is the chemical formula for water?",     "H2O"),
    ("What is the formula for carbon dioxide?",     "CO2"),
    ("What is the boiling point of water in Celsius?",  "100"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the atomic number of hydrogen?",      "1"),
    ("What is the atomic number of carbon?",        "6"),
    ("What is the atomic number of oxygen?",        "8"),
    ("What is the hardest natural substance?",      "diamond"),
    ("What planet is closest to the Sun?",          "Mercury"),
    ("What planet is largest in the solar system?", "Jupiter"),
    ("What planet has rings?",                      "Saturn"),
    ("What is the closest star to Earth?",          "the Sun"),
    ("What is the largest ocean on Earth?",         "Pacific Ocean"),
    ("What is the tallest mountain in the world?",  "Mount Everest"),
    ("What is the deepest lake in the world?",      "Lake Baikal"),
    ("What is the largest country by area?",        "Russia"),
    # history
    ("In what year did World War II end?",                    "1945"),
    ("In what year did World War I begin?",                   "1914"),
    ("In what year did man first land on the Moon?",          "1969"),
    ("What year did the Titanic sink?",                       "1912"),
    ("What year was the Eiffel Tower built?",                 "1889"),
    ("What year did Christopher Columbus reach the Americas?","1492"),
    ("What year did the Soviet Union dissolve?",              "1991"),
    ("In what year did the Berlin Wall fall?",                "1989"),
    # people
    ("Who discovered penicillin?",                   "Alexander Fleming"),
    ("Who invented the telephone?",                  "Alexander Graham Bell"),
    ("Who painted the Mona Lisa?",                   "Leonardo da Vinci"),
    ("Who wrote Romeo and Juliet?",                  "William Shakespeare"),
    ("Who developed the theory of relativity?",      "Albert Einstein"),
    ("Who discovered gravity?",                      "Isaac Newton"),
    ("Who invented the light bulb?",                 "Thomas Edison"),
    ("Who was the first woman to win a Nobel Prize?","Marie Curie"),
    ("Who was the first human to travel to space?",  "Yuri Gagarin"),
    ("Who invented the printing press?",             "Johannes Gutenberg"),
    ("Who founded Microsoft?",                       "Bill Gates"),
    ("Who sang Bohemian Rhapsody?",                  "Queen"),
    # math / misc
    ("What is the value of pi to 2 decimal places?",  "3.14"),
    ("What is the square root of 144?",               "12"),
    ("What is 2 to the power of 10?",                "1024"),
    ("What is the Roman numeral for 10?",             "X"),
    ("How many days are in a leap year?",             "366"),
    ("How many planets are in the solar system?",     "8"),
    ("How many bones are in the adult human body?",   "206"),
    ("What is the speed of light in km/s?",           "299792"),
    ("What does DNA stand for?",                      "Deoxyribonucleic acid"),
    ("What is the most spoken language in the world?","Mandarin Chinese"),
    ("What language is spoken in Brazil?",            "Portuguese"),
    ("What is the normal human body temperature in Celsius?", "37"),
    ("What is the largest organ in the human body?",  "skin"),
    ("How many chromosomes do humans have?",          "46"),
    ("What gas do plants absorb from the atmosphere?","carbon dioxide"),
    ("What vitamin is produced by sunlight?",         "Vitamin D"),
    ("What is the currency of Japan?",                "yen"),
    ("What is the largest continent?",                "Asia"),
    ("What is the longest river in the world?",       "Nile"),
    ("How many colors are in a rainbow?",             "7"),
    ("How many legs does a spider have?",             "8"),
    ("What is the largest animal on Earth?",          "blue whale"),
    ("What is the fastest land animal?",              "cheetah"),
    ("What does CPU stand for?",                      "Central Processing Unit"),
]


def _cache_path() -> str:
    d = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "bank_gpt2.json")


def build_bank(force: bool = False) -> list:
    """Build or load the GPT-2 embedding bank.  Returns list of dicts."""
    path = _cache_path()
    if not force and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        bank = [{"q": d["q"], "emb": np.array(d["emb"])} for d in data]
        return bank

    _ensure_model()
    model, tok = _state["model"], _state["tok"]
    import torch

    bank = []
    print(f"Building bank ({len(BUILTIN_FACTS)} facts)…", flush=True)
    for i, (q, _) in enumerate(BUILTIN_FACTS):
        emb = _embed(q, model, tok)
        bank.append({"q": q, "emb": emb})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(BUILTIN_FACTS)}]", flush=True)

    with open(path, "w") as f:
        json.dump([{"q": b["q"], "emb": b["emb"].tolist()} for b in bank], f)
    print(f"Bank saved → {path}")
    return bank


def _ensure_model():
    if "model" in _state:
        return
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    print("Loading GPT-2 (117M, CPU)…", flush=True)
    tok   = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()
    _state["model"] = model
    _state["tok"]   = tok


def _ensure_bank():
    if "bank" in _state:
        return
    _state["bank"] = build_bank(force=False)


def _ensure_pca3d():
    if "pca3d" in _state:
        return
    from sklearn.decomposition import PCA
    bank = _state["bank"]
    embs = np.array([b["emb"] for b in bank])
    pca  = PCA(n_components=3, random_state=42)
    _state["pca3d"]    = pca
    _state["bank_3d"]  = pca.fit_transform(embs)


def _embed(text: str, model, tok) -> np.ndarray:
    import torch
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    return out.hidden_states[_LAYER][0, -1, :].numpy()


def _entropy(text: str, model, tok, n_tokens: int = 24) -> float:
    """Mean next-token entropy (nats) over n_tokens generated from text."""
    import torch
    prompt = f"Q: {text}\nA:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=100)
    ids    = inputs["input_ids"]
    ents   = []
    with torch.no_grad():
        for _ in range(n_tokens):
            out    = model(ids, output_hidden_states=False)
            logits = out.logits[0, -1, :]
            lp     = torch.log_softmax(logits, dim=-1)
            ent    = float(-torch.sum(torch.exp(lp) * lp))
            ents.append(ent)
            ids = torch.cat([ids, logits.argmax().unsqueeze(0).unsqueeze(0)], dim=1)
    return float(np.mean(ents))


# ── public API ────────────────────────────────────────────────────────────────

@dataclass
class CanaryResult:
    risk:      float
    gamma_h:   float
    d_min:     float
    entropy:   float
    nearest_q: str
    label:     str     # "LOW" / "MEDIUM" / "HIGH" / "VERY HIGH"

    def __repr__(self) -> str:
        bar = "█" * int(self.risk * 20) + "░" * (20 - int(self.risk * 20))
        return (f"CanaryResult(risk={self.risk:.2f} [{self.label}]  "
                f"γ={self.gamma_h:.2f}  H={self.entropy:.2f}  "
                f'nearest="{self.nearest_q[:40]}")\n'
                f"  [{bar}] {self.risk*100:.0f}%")


@dataclass
class RouteResult:
    action:     str    # "DIRECT_ANSWER" | "USE_RAG"
    confidence: float  # certainty of the routing decision (0–1)
    risk:       float  # raw epistemic risk score
    nearest_q:  str    # nearest known question in bank

    def __repr__(self) -> str:
        icon = "✓" if self.action == "DIRECT_ANSWER" else "⟳"
        return (f"RouteResult({icon} {self.action}  "
                f"confidence={self.confidence:.2f}  risk={self.risk:.2f}  "
                f'nearest="{self.nearest_q[:40]}")')


def _compute_risk(question: str, fast: bool) -> tuple:
    """Returns (risk, gamma_h, d_min, entropy, nearest_q)."""
    _ensure_model()
    _ensure_bank()
    _ensure_pca3d()

    model, tok = _state["model"], _state["tok"]
    bank       = _state["bank"]
    pca3d      = _state["pca3d"]
    bank_3d    = _state["bank_3d"]

    emb        = _embed(question, model, tok)
    bank_embs  = np.array([b["emb"] for b in bank])
    dists      = np.linalg.norm(bank_embs - emb, axis=1)
    d_min      = float(dists.min())
    nn_idx     = int(dists.argmin())

    from scipy.special import expit as sigmoid
    risk_d = float(sigmoid(_A * (d_min - _RC)))

    ent    = _entropy(question, model, tok) if not fast else 3.5
    risk_h = float(sigmoid(1.5 * (ent - 3.5)))
    risk   = float(np.clip(0.65 * risk_d + 0.35 * risk_h, 0.0, 1.0))

    q_3d    = pca3d.transform(emb[np.newaxis, :])[0]
    d3d_min = float(np.linalg.norm(bank_3d - q_3d, axis=1).min())
    gamma_h = float(1.0 - np.exp(-_K * max(d3d_min - _RTH, 0.0)))

    return risk, gamma_h, d_min, ent, bank[nn_idx]["q"]


def score(question: str, answer: str = "", *, fast: bool = False) -> CanaryResult:
    """
    Estimate epistemic risk for a question.

    Returns CanaryResult with .risk in [0, 1].
    Low risk  → question is within the known knowledge bank.
    High risk → question is outside known territory; consider RAG or search.
    """
    risk, gamma_h, d_min, ent, nearest_q = _compute_risk(question, fast)

    if   risk < 0.35: label = "LOW"
    elif risk < 0.65: label = "MEDIUM"
    elif risk < 0.85: label = "HIGH"
    else:             label = "VERY HIGH"

    return CanaryResult(
        risk=risk, gamma_h=gamma_h, d_min=d_min,
        entropy=ent, nearest_q=nearest_q, label=label,
    )


def route(question: str, *, threshold: float = 0.35, fast: bool = False) -> RouteResult:
    """
    Decide whether a question needs retrieval (RAG / search / tool) or can be
    answered directly.

    Args:
        question:  The user query.
        threshold: Risk cutoff.  Below → DIRECT_ANSWER; above → USE_RAG.
                   Default 0.35 balances precision and RAG savings.
        fast:      Skip entropy (2× faster, slightly lower accuracy).

    Returns:
        RouteResult with .action and .confidence.

    Example::

        r = canary.route("What is the boiling point of water?")
        if r.action == "USE_RAG":
            answer = rag_pipeline(question)
        else:
            answer = llm(question)
    """
    risk, _, _, _, nearest_q = _compute_risk(question, fast)

    if risk < threshold:
        action     = "DIRECT_ANSWER"
        confidence = round(1.0 - risk / threshold, 3)
    else:
        action     = "USE_RAG"
        confidence = round((risk - threshold) / (1.0 - threshold), 3)

    return RouteResult(
        action=action, confidence=confidence,
        risk=round(risk, 3), nearest_q=nearest_q,
    )


def load_bank(texts: "list[str]", *, save_path: "str | None" = None) -> None:
    """
    Replace the built-in knowledge bank with a custom one.

    Build a bank from your own documents (FAQ, wiki, product docs) so Canary
    routes questions outside *your* knowledge domain to RAG.

    Args:
        texts:     List of representative sentences / questions from your corpus.
        save_path: Optional path to persist the bank as JSON for later reuse.

    Example::

        import canary
        canary.load_bank([
            "Our return policy is 30 days.",
            "We ship to 45 countries.",
            ...
        ])
        r = canary.route("Do you ship to Antarctica?")
        # RouteResult(⟳ USE_RAG  confidence=0.91 ...)
    """
    _ensure_model()
    model, tok = _state["model"], _state["tok"]
    bank = [{"q": t, "emb": _embed(t, model, tok)} for t in texts]
    _state["bank"] = bank
    _state.pop("pca3d",   None)
    _state.pop("bank_3d", None)
    if save_path:
        with open(save_path, "w") as f:
            json.dump([{"q": b["q"], "emb": b["emb"].tolist()} for b in bank], f)
