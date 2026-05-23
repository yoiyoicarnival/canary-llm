#!/usr/bin/env python3
"""
canary_benchmark.py — Hallucination Radar を TruthfulQA で評価

Stage 1: prompt-only OOD score (既存 radar, GPT-2 embedding)
Stage 2: output-based score (GPU不要, テキスト信号のみ)
Stage 3: 統合スコア

出力: AUC / Precision / Recall / F1 (各 stage)
"""
import os, sys, json, re, math
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

# ─── Stage 2: output-based signals (GPU不要) ──────────────────────────────────

UNCERTAINTY_MARKERS = [
    "i think", "i believe", "i'm not sure", "i am not sure",
    "i'm not certain", "possibly", "perhaps", "maybe", "it's possible",
    "i don't know", "i do not know", "i cannot", "i can't confirm",
    "reportedly", "allegedly", "supposedly", "some say", "it is said",
    "historians believe", "scientists think", "experts suggest",
]

HALLUCINATION_PATTERNS = [
    r"\bin \d{4}\b",           # 年号 (確認が難しい)
    r"\baccording to\b",       # 出典曖昧
    r"\b(?:famous|well-known|renowned) (?:for|as)\b",  # 曖昧な評価
    r"\bsome (?:people|sources|experts|scientists)\b",  # 主語曖昧
]


def output_score(question: str, answer: str) -> dict:
    """
    GPU不要のテキスト信号から hallucination リスクを計算。
    返り値: {score: float[0,1], signals: dict}
    """
    ans_low = answer.lower()

    # 1. 不確実マーカー密度
    n_uncertainty = sum(1 for m in UNCERTAINTY_MARKERS if m in ans_low)
    uncertainty_density = min(n_uncertainty / max(len(answer.split()) / 20, 1), 1.0)

    # 2. 繰り返しスコア (3-gram 重複率)
    words = ans_low.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        repetition_score = 1.0 - unique_ratio
    else:
        repetition_score = 0.0

    # 3. 疑わしいパターン密度
    n_patterns = sum(1 for p in HALLUCINATION_PATTERNS
                     if re.search(p, ans_low))
    pattern_score = min(n_patterns / 3.0, 1.0)

    # 4. 回答長ペナルティ (短すぎ = 逃げ回答、長すぎ = confabulation)
    n_words = len(words)
    if n_words < 5:
        length_score = 0.3   # 短すぎ
    elif n_words > 150:
        length_score = 0.4   # 長すぎ = 詰め込み
    else:
        length_score = 0.0

    # 5. 質問語句の反復 (echo 回避)
    q_words = set(question.lower().split()) - {"what","is","the","a","an","of","in","to","do","does","did","was","were","are","how","why","when","who","which"}
    echo_words = sum(1 for w in q_words if w in ans_low)
    echo_score = min(echo_words / max(len(q_words), 1), 1.0) * 0.3

    # 統合 (重み付き平均)
    score = (0.35 * uncertainty_density
           + 0.25 * repetition_score
           + 0.20 * pattern_score
           + 0.10 * length_score
           + 0.10 * echo_score)
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score": score,
        "uncertainty": uncertainty_density,
        "repetition": repetition_score,
        "pattern": pattern_score,
        "length": length_score,
        "echo": echo_score,
        "n_uncertainty_markers": n_uncertainty,
    }


# ─── Stage 1: prompt OOD score (GPT-2 embedding) ─────────────────────────────

_radar_bank  = None
_hr_mod      = None


def _load_radar():
    global _radar_bank, _hr_mod
    if _radar_bank is not None:
        return
    sys.path.insert(0, "/home/yoiyoi/experiments")
    import hallucination_radar as hr
    _hr_mod = hr
    _radar_bank = hr.build_bank(force=False)
    hr._load_model()
    hr._ensure_pca3d(_radar_bank)


def prompt_ood_score(question: str) -> dict:
    """GPT-2 embedding による prompt OOD スコア。"""
    _load_radar()
    hr  = _hr_mod
    emb = hr.get_embedding(question)
    metrics = hr.compute_risk(emb, _radar_bank, entropy_mean=None)
    return {
        "score": metrics["risk"],
        "d_min": metrics["d_min"],
        "gamma_h": metrics["gamma_h"],
        "ood": metrics["ood"],
    }


# ─── 統合スコア ───────────────────────────────────────────────────────────────

def canary_score(question: str, answer: str,
                 w_prompt: float = 0.4,
                 w_output: float = 0.6) -> dict:
    """
    prompt OOD (Stage1) + output signal (Stage2) の統合スコア。
    answer が空文字の場合は prompt-only モード。
    """
    p = prompt_ood_score(question)
    if answer.strip():
        o = output_score(question, answer)
        combined = w_prompt * p["score"] + w_output * o["score"]
    else:
        o = {"score": 0.0}
        combined = p["score"]

    return {
        "risk": float(np.clip(combined, 0.0, 1.0)),
        "prompt_risk": p["score"],
        "output_risk": o["score"],
        "d_min": p.get("d_min", 0.0),
        "gamma_h": p.get("gamma_h", 0.0),
    }


# ─── ベンチマーク ─────────────────────────────────────────────────────────────

def auc_from_scores(labels, scores):
    """sklearn不要の手計算 AUC-ROC。"""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def best_f1_threshold(labels, scores):
    """最良 F1 のしきい値・Precision・Recall を返す。"""
    best = {"f1": 0, "thresh": 0.5, "prec": 0, "rec": 0}
    for thresh in np.linspace(0, 1, 101):
        preds = [1 if s >= thresh else 0 for s in scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best["f1"]:
            best = {"f1": f1, "thresh": thresh, "prec": prec, "rec": rec}
    return best


def run_benchmark(n_truthful: int = 200, seed: int = 42, save_path: str = "/home/yoiyoi/canary_benchmark_results.json"):
    """
    TruthfulQA ベンチマーク実行。

    ラベル設定:
      label=1 (hallucination-prone): TruthfulQA adversarial 質問
      label=0 (factual):             radar の FACTUAL_QA bank 質問

    Stage1 のみ、Stage2 のみ、統合スコアそれぞれを評価。
    また TruthfulQA 内部で (correct_answer, label=0) vs (incorrect_answer, label=1) を評価。
    """
    from datasets import load_dataset
    import random
    random.seed(seed)
    rng = np.random.RandomState(seed)

    print("=== Canary Benchmark: TruthfulQA ===")
    print()

    # ── データ準備 ────────────────────────────────────────────────────────────
    print("[1/4] TruthfulQA 読み込み...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = [ds[i] for i in indices[:n_truthful]]

    sys.path.insert(0, "/home/yoiyoi/experiments")
    from hallucination_radar import FACTUAL_QA

    # 正しい train/test split: bank に入っていない factual 質問を negative に使う
    # bank: FACTUAL_QA[:90]  →  test_neg: FACTUAL_QA[90:]
    BANK_SIZE = 90
    factual_all = [(q, ans[0]) for q, ans, _ in FACTUAL_QA]
    factual_test = factual_all[BANK_SIZE:]   # 銀行外の factual 質問 (honest negatives)
    print(f"  Bank={BANK_SIZE}件  test_neg(unseen factual)={len(factual_test)}件  test_pos(TruthfulQA)={n_truthful}件")
    print(f"  ※ test_neg は銀行に入っていない → 循環バイアスなし")

    # ── Stage 1: prompt OOD (prompt → risk score) ────────────────────────────
    print(f"[2/4] Stage1 (prompt OOD) スコアリング中...")
    _load_radar()

    stage1_scores, stage1_labels = [], []

    for item in selected:
        r = prompt_ood_score(item["question"])
        stage1_scores.append(r["score"])
        stage1_labels.append(1)    # adversarial = hallucination-prone

    for q, _ in factual_test:
        r = prompt_ood_score(q)
        stage1_scores.append(r["score"])
        stage1_labels.append(0)    # factual unseen = safe

    auc1 = auc_from_scores(stage1_labels, stage1_scores)
    best1 = best_f1_threshold(stage1_labels, stage1_scores)
    print(f"  Stage1 AUC={auc1:.3f}  F1={best1['f1']:.3f}  Prec={best1['prec']:.3f}  Rec={best1['rec']:.3f}  thresh={best1['thresh']:.2f}")

    # ── Stage 2: output-based (correct vs incorrect answer) ──────────────────
    print(f"[3/4] Stage2 (output-based) スコアリング中...")
    stage2_scores, stage2_labels = [], []

    for item in selected:
        q = item["question"]
        # correct answer → label=0 (不ハル)
        correct = item["correct_answers"]
        if correct:
            ans_c = correct[0]
            r_c = output_score(q, ans_c)
            stage2_scores.append(r_c["score"])
            stage2_labels.append(0)

        # incorrect answer → label=1 (ハル)
        incorrect = item["incorrect_answers"]
        if incorrect:
            ans_i = random.choice(incorrect)
            r_i = output_score(q, ans_i)
            stage2_scores.append(r_i["score"])
            stage2_labels.append(1)

    auc2 = auc_from_scores(stage2_labels, stage2_scores)
    best2 = best_f1_threshold(stage2_labels, stage2_scores)
    print(f"  Stage2 AUC={auc2:.3f}  F1={best2['f1']:.3f}  Prec={best2['prec']:.3f}  Rec={best2['rec']:.3f}  thresh={best2['thresh']:.2f}")

    # ── Stage 1+2 統合 (TruthfulQA vs factual unseen, output 付き) ──────────
    print(f"[4/4] 統合スコア評価...")
    combo_scores, combo_labels = [], []

    for item in selected:
        q = item["question"]
        best_ans = item["best_answer"]
        r = canary_score(q, best_ans)
        combo_scores.append(r["risk"])
        combo_labels.append(1)

    for q, ans in factual_test:
        r = canary_score(q, ans)
        combo_scores.append(r["risk"])
        combo_labels.append(0)

    auc3 = auc_from_scores(combo_labels, combo_scores)
    best3 = best_f1_threshold(combo_labels, combo_scores)
    print(f"  Combined AUC={auc3:.3f}  F1={best3['f1']:.3f}  Prec={best3['prec']:.3f}  Rec={best3['rec']:.3f}  thresh={best3['thresh']:.2f}")

    # ── カテゴリ別 Stage1 スコア ──────────────────────────────────────────────
    cat_scores = {}
    for item, score in zip(selected, stage1_scores[:n_truthful]):
        c = item["category"]
        cat_scores.setdefault(c, []).append(score)
    cat_mean = {c: float(np.mean(v)) for c, v in cat_scores.items()}
    top_cats = sorted(cat_mean.items(), key=lambda x: -x[1])[:10]
    bot_cats = sorted(cat_mean.items(), key=lambda x:  x[1])[:5]

    print()
    print("カテゴリ別 Stage1 スコア (高い = より OOD):")
    for c, m in top_cats:
        bar = "█" * int(m * 20)
        print(f"  {c:35s} {m:.3f} {bar}")
    print("  ...")
    for c, m in bot_cats:
        bar = "█" * int(m * 20)
        print(f"  {c:35s} {m:.3f} {bar}")

    # ── 結果保存 ──────────────────────────────────────────────────────────────
    results = {
        "n_truthful": n_truthful,
        "n_factual":  len(factual_test),
        "stage1": {"auc": auc1, **best1},
        "stage2": {"auc": auc2, **best2},
        "combined": {"auc": auc3, **best3},
        "category_scores": dict(sorted(cat_mean.items(), key=lambda x: -x[1])),
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n結果保存: {save_path}")

    print()
    print("=" * 50)
    print(f"  Stage1 (prompt OOD)  AUC = {auc1:.3f}")
    print(f"  Stage2 (output text) AUC = {auc2:.3f}")
    print(f"  Combined             AUC = {auc3:.3f}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="TruthfulQA からサンプリング数")
    p.add_argument("question", nargs="?", help="単一質問スコアリング")
    p.add_argument("--answer", default="", help="回答テキスト (省略可)")
    args = p.parse_args()

    if args.question:
        _load_radar()
        r = canary_score(args.question, args.answer)
        print(f"\nRisk: {r['risk']*100:.0f}%  (prompt={r['prompt_risk']*100:.0f}%  output={r['output_risk']*100:.0f}%)")
    else:
        run_benchmark(n_truthful=args.n)
