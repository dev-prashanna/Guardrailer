#!/usr/bin/env python3
"""Cross-Dataset Generalization — Evaluate hybrid classifier on external datasets."""

import pandas as pd
import numpy as np
import re
import json
import time
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from datasets import load_dataset
from pathlib import Path

SPLIT_DIR = Path(__file__).parent

print("=" * 70)
print("CROSS-DATASET GENERALIZATION TEST")
print("=" * 70)

print("\n[1/5] Downloading external datasets...")

datasets_info = []

print("\n  [1/4] JailbreakBench (harmful + benign)...")
try:
    harmful = load_dataset('JailbreakBench/JBB-Behaviors', 'behaviors', split='harmful')
    benign = load_dataset('JailbreakBench/JBB-Behaviors', 'behaviors', split='benign')
    harmful_df = harmful.to_pandas()[['Goal']].copy()
    harmful_df['is_malicious'] = 1
    harmful_df.rename(columns={'Goal': 'prompt_text'}, inplace=True)
    benign_df = benign.to_pandas()[['Goal']].copy()
    benign_df['is_malicious'] = 0
    benign_df.rename(columns={'Goal': 'prompt_text'}, inplace=True)
    jbb_df = pd.concat([harmful_df, benign_df], ignore_index=True)
    print(f"    {len(jbb_df)} samples")
    datasets_info.append(("JailbreakBench", jbb_df))
except Exception as e:
    print(f"    FAILED: {e}")

print("\n  [2/4] Jailbreak Classification Dataset...")
try:
    jc = load_dataset('jackhhao/jailbreak-classification', split='train')
    jc_df = jc.to_pandas()[['prompt', 'type']].copy()
    jc_df.rename(columns={'prompt': 'prompt_text'}, inplace=True)
    jc_df['is_malicious'] = (jc_df['type'] == 'jailbreak').astype(int)
    jc_df.drop(columns=['type'], inplace=True)
    print(f"    {len(jc_df)} samples")
    datasets_info.append(("Jailbreak Classification", jc_df))
except Exception as e:
    print(f"    FAILED: {e}")

print("\n  [3/4] Jailbreak Complete DS Labeled...")
try:
    jd = load_dataset('GeorgeDaDude/Jailbreak_Complete_DS_labeled', split='train')
    jd_df = jd.to_pandas()[['question', 'label']].copy()
    jd_df.rename(columns={'question': 'prompt_text', 'label': 'is_malicious'}, inplace=True)
    print(f"    {len(jd_df)} samples")
    datasets_info.append(("Jailbreak Complete DS", jd_df))
except Exception as e:
    print(f"    FAILED: {e}")

print("\n  [4/4] JailbreakHub...")
try:
    jh = load_dataset('walledai/JailbreakHub', split='train')
    jh_df = jh.to_pandas()[['prompt', 'jailbreak']].copy()
    jh_df.rename(columns={'prompt': 'prompt_text', 'jailbreak': 'is_malicious'}, inplace=True)
    jh_df['is_malicious'] = jh_df['is_malicious'].astype(int)
    print(f"    {len(jh_df)} samples")
    datasets_info.append(("JailbreakHub", jh_df))
except Exception as e:
    print(f"    FAILED: {e}")

print("\n  Saving external datasets...")
for name, df in datasets_info:
    safe_name = name.lower().replace(' ', '_').replace('/', '_')
    df.to_csv(SPLIT_DIR / f"ext_{safe_name}.csv", index=False)

print("\n[2/5] Loading trained model...")
tfidf = joblib.load(SPLIT_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(SPLIT_DIR / "xgboost_model.pkl")


def extract_handcrafted_features(texts):
    features = []
    for text in texts:
        f = {}
        words = text.split()
        word_count = max(len(words), 1)
        f["char_count"] = len(text)
        f["word_count"] = word_count
        f["sentence_count"] = max(text.count(".") + text.count("!") + text.count("?"), 1)
        f["avg_sentence_len"] = word_count / f["sentence_count"]
        f["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        f["special_char_ratio"] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        f["word_len_variance"] = float(np.var([len(w) for w in words])) if words else 0
        attack_words = ["ignore", "previous", "instructions", "bypass", "override",
                        "jailbreak", "system", "prompt", "extract", "reveal", "output",
                        "forget", "disregard", "admin", "root", "sudo"]
        f["attack_keyword_count"] = sum(1 for w in words if w.lower() in attack_words)
        f["attack_keyword_density"] = f["attack_keyword_count"] / word_count
        f["has_base64"] = 1 if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text) else 0
        f["has_hex"] = 1 if re.search(r'\\x[0-9a-fA-F]{2}', text) else 0
        f["has_url_encoding"] = 1 if re.search(r'%[0-9a-fA-F]{2}', text) else 0
        f["has_unicode_escape"] = 1 if re.search(r'\\u[0-9a-fA-F]{4}', text) else 0
        f["has_instruction_override"] = 1 if re.search(r'ignore|disregard|forget', text, re.I) else 0
        f["has_role_hijack"] = 1 if re.search(r'you are now|act as|pretend to be', text, re.I) else 0
        f["has_system_extraction"] = 1 if re.search(r'system prompt|initial instructions|original instructions', text, re.I) else 0
        f["word_repeat_ratio"] = 1 - len(set(words)) / word_count
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        f["bigram_repeat_ratio"] = 1 - len(set(bigrams)) / max(len(bigrams), 1)
        unique_chars = set(text)
        f["char_entropy"] = -sum((text.count(c)/len(text)) * np.log2(text.count(c)/len(text))
                                  for c in unique_chars) if len(text) > 0 else 0
        f["unique_word_ratio"] = len(set(words)) / word_count
        f["question_presence"] = 1 if "?" in text else 0
        f["exclamation_ratio"] = text.count("!") / max(len(text), 1)
        f["has_delimiter"] = 1 if re.search(r'```|\[INST\]|<<SYS>>|<\|im_start\|>', text) else 0
        f["has_xml_tags"] = 1 if re.search(r'<[^>]+>', text) else 0
        f["has_role_marker"] = 1 if re.search(r'(USER|ASSISTANT|SYSTEM)\s*:', text) else 0
        f["starts_with_imperative"] = 1 if re.match(r'^(ignore|forget|disregard|reveal|output|print|show)', text, re.I) else 0
        features.append(f)
    return pd.DataFrame(features)


def predict_batch(texts):
    tfidf_vec = tfidf.transform(texts)
    hc = extract_handcrafted_features(texts)
    hc_sparse = csr_matrix(hc.values.astype(np.float32))
    combined = hstack([tfidf_vec, hc_sparse])
    preds = model.predict(combined)
    proba = model.predict_proba(combined)[:, 1]
    return preds, proba


print("\n[3/5] Evaluating on external datasets...")

all_results = []
for name, df in datasets_info:
    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  {'='*60}")
    df = df[df['prompt_text'].astype(str).str.strip().str.len() > 0].copy()
    df['prompt_text'] = df['prompt_text'].astype(str)
    X_text = df["prompt_text"].tolist()
    y_true = df["is_malicious"].values
    print(f"  Samples: {len(X_text):,}")
    print(f"  Malicious: {y_true.sum():,} ({y_true.mean()*100:.1f}%)")
    print(f"  Safe: {(1-y_true).sum():,} ({(1-y_true.mean())*100:.1f}%)")
    t0 = time.perf_counter()
    preds, proba = predict_batch(X_text)
    elapsed = time.perf_counter() - t0
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba)
    except:
        auc = 0.5
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print(f"  Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    all_results.append({
        "dataset": name, "samples": len(X_text),
        "malicious": int(y_true.sum()), "safe": int((1-y_true).sum()),
        "accuracy": round(float(acc), 4), "f1": round(float(f1), 4),
        "precision": round(float(prec), 4), "recall": round(float(rec), 4),
        "auc_roc": round(float(auc), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    })

in_dist = {"accuracy": 0.8645, "f1": 0.8881, "auc_roc": 0.9410}

print(f"\n{'='*70}")
print("COMPARISON: In-Distribution vs Cross-Dataset")
print(f"{'='*70}")
print(f"\n{'Dataset':<30} {'Acc':>8} {'F1':>8} {'AUC':>8} {'D Acc':>8}")
print("-" * 60)
print(f"{'Guardrailer v1 (train)':<30} {in_dist['accuracy']:>8.4f} {in_dist['f1']:>8.4f} {in_dist['auc_roc']:>8.4f} {'---':>8}")
for r in all_results:
    delta = r['accuracy'] - in_dist['accuracy']
    print(f"{r['dataset']:<30} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['auc_roc']:>8.4f} {delta:>+8.4f}")

avg_acc = np.mean([r['accuracy'] for r in all_results]) if all_results else 0
avg_f1 = np.mean([r['f1'] for r in all_results]) if all_results else 0
avg_auc = np.mean([r['auc_roc'] for r in all_results]) if all_results else 0
print("-" * 60)
print(f"{'Average (external)':<30} {avg_acc:>8.4f} {avg_f1:>8.4f} {avg_auc:>8.4f} {avg_acc - in_dist['accuracy']:>+8.4f}")

print(f"\n[5/5] Saving results...")
final_results = {
    "in_distribution": {"dataset": "Guardrailer v1", "accuracy": in_dist["accuracy"], "f1": in_dist["f1"], "auc_roc": in_dist["auc_roc"]},
    "cross_dataset": all_results,
    "summary": {
        "n_external_datasets": len(all_results),
        "avg_accuracy": round(float(avg_acc), 4),
        "avg_f1": round(float(avg_f1), 4),
        "avg_auc_roc": round(float(avg_auc), 4),
        "avg_accuracy_drop_pp": round(float((in_dist['accuracy'] - avg_acc) * 100), 1),
    }
}
with open(SPLIT_DIR / "cross_dataset_results.json", "w") as f:
    json.dump(final_results, f, indent=2)
rows = [{"Dataset": r["dataset"], "Samples": r["samples"], "Accuracy": r["accuracy"], "F1": r["f1"], "AUC-ROC": r["auc_roc"]} for r in all_results]
pd.DataFrame(rows).to_csv(SPLIT_DIR / "cross_dataset_results.csv", index=False)
print("Done!")
