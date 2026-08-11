#!/usr/bin/env python3
"""Step 3: Ablation study."""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
from pathlib import Path

SPLIT_DIR = Path(__file__).parent

train = pd.read_parquet(SPLIT_DIR / "split_train.parquet")
test = pd.read_parquet(SPLIT_DIR / "split_test.parquet")

X_train_text = train["prompt_text"].astype(str).tolist()
y_train = train["is_malicious"].values
X_test_text = test["prompt_text"].astype(str).tolist()
y_test = test["is_malicious"].values

print(f"Train: {len(X_train_text):,}  Test: {len(X_test_text):,}")


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


print("Extracting handcrafted features...")
X_train_hc = extract_handcrafted_features(X_train_text)
X_test_hc = extract_handcrafted_features(X_test_text)
print(f"  Features: {X_train_hc.shape[1]}")

print("Computing TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
print(f"  Shape: {X_train_tfidf.shape}")

X_train_hc_sparse = csr_matrix(X_train_hc.values.astype(np.float32))
X_test_hc_sparse = csr_matrix(X_test_hc.values.astype(np.float32))
X_train_combined = hstack([X_train_tfidf, X_train_hc_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_hc_sparse])

ablation_results = []

configs = {
    "TF-IDF only": (X_train_tfidf, X_test_tfidf),
    "Handcrafted only": (X_train_hc_sparse, X_test_hc_sparse),
    "TF-IDF + Handcrafted": (X_train_combined, X_test_combined),
}

for name, (Xtr, Xte) in configs.items():
    print(f"\nTraining: {name} (shape: {Xtr.shape})")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, device="cuda", tree_method="hist")
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    ablation_results.append({"Config": name, "Features": Xtr.shape[1], "Accuracy": acc, "F1": f1, "AUC-ROC": auc})
    print(f"  Acc: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

ablation_df = pd.DataFrame(ablation_results)
print("\n" + "=" * 80)
print("ABLATION RESULTS")
print("=" * 80)
print(ablation_df.to_string(index=False, float_format="{:.4f}".format))
ablation_df.to_csv(SPLIT_DIR / "ablation_results.csv", index=False)
