#!/usr/bin/env python3
"""Step 2: Run baseline models on fixed test set."""

import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
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

print(f"Train size: {len(X_train_text):,}")
print(f"Test size:  {len(X_test_text):,}")

results = []

majority_pred = np.ones(len(y_test))
acc = accuracy_score(y_test, majority_pred)
f1 = f1_score(y_test, majority_pred, zero_division=0)
results.append({"Model": "Majority Classifier", "Accuracy": acc, "F1": f1, "Precision": 0, "Recall": 0, "AUC-ROC": 0.5})
print(f"\n[1/6] Majority Classifier - Acc: {acc:.4f}")

patterns = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:DAN|in\s+developer\s+mode)",
    r"reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions)",
    r"output\s+(?:your|the)\s+(?:system\s+prompt|initial\s+instructions)",
    r"print\s+(?:your|the)\s+(?:system\s+prompt|instructions)",
    r"what\s+(?:are|were)\s+your\s+(?:initial|original)\s+instructions",
    r"(?:USER|SYSTEM|ASSISTANT)\s*:",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"<<SYS>>",
    r"###\s*(?:System|Instruction|Human)\s*:",
]

def regex_score(text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return 1
    return 0

regex_preds = [regex_score(t) for t in X_test_text]
acc = accuracy_score(y_test, regex_preds)
f1 = f1_score(y_test, regex_preds, zero_division=0)
prec = precision_score(y_test, regex_preds, zero_division=0)
rec = recall_score(y_test, regex_preds, zero_division=0)
results.append({"Model": "Regex Only", "Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec, "AUC-ROC": 0.5})
print(f"[2/6] Regex Only - Acc: {acc:.4f}  F1: {f1:.4f}")

print("\nFitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
print(f"  TF-IDF shape: {X_train_tfidf.shape}")

print("[3/6] Training TF-IDF + Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)
lr_proba = lr.predict_proba(X_test_tfidf)[:, 1]
results.append({"Model": "TF-IDF + LogReg", "Accuracy": accuracy_score(y_test, lr_preds), "F1": f1_score(y_test, lr_preds), "Precision": precision_score(y_test, lr_preds), "Recall": recall_score(y_test, lr_preds), "AUC-ROC": roc_auc_score(y_test, lr_proba)})
print(f"  Acc: {results[-1]['Accuracy']:.4f}  F1: {results[-1]['F1']:.4f}  AUC: {results[-1]['AUC-ROC']:.4f}")

print("[4/6] Training TF-IDF + Linear SVM...")
svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)
svm_scores = svm.decision_function(X_test_tfidf)
results.append({"Model": "TF-IDF + LinearSVC", "Accuracy": accuracy_score(y_test, svm_preds), "F1": f1_score(y_test, svm_preds), "Precision": precision_score(y_test, svm_preds), "Recall": recall_score(y_test, svm_preds), "AUC-ROC": roc_auc_score(y_test, svm_scores)})
print(f"  Acc: {results[-1]['Accuracy']:.4f}  F1: {results[-1]['F1']:.4f}  AUC: {results[-1]['AUC-ROC']:.4f}")

print("[5/6] Training TF-IDF + Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
rf_preds = rf.predict(X_test_tfidf)
rf_proba = rf.predict_proba(X_test_tfidf)[:, 1]
results.append({"Model": "TF-IDF + RandomForest", "Accuracy": accuracy_score(y_test, rf_preds), "F1": f1_score(y_test, rf_preds), "Precision": precision_score(y_test, rf_preds), "Recall": recall_score(y_test, rf_preds), "AUC-ROC": roc_auc_score(y_test, rf_proba)})
print(f"  Acc: {results[-1]['Accuracy']:.4f}  F1: {results[-1]['F1']:.4f}  AUC: {results[-1]['AUC-ROC']:.4f}")

print("[6/6] Training TF-IDF + XGBoost (no handcrafted features)...")
xgb_tfidf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, device="cuda", tree_method="hist")
xgb_tfidf.fit(X_train_tfidf, y_train)
xgb_tfidf_preds = xgb_tfidf.predict(X_test_tfidf)
xgb_tfidf_proba = xgb_tfidf.predict_proba(X_test_tfidf)[:, 1]
results.append({"Model": "TF-IDF + XGBoost (no HC)", "Accuracy": accuracy_score(y_test, xgb_tfidf_preds), "F1": f1_score(y_test, xgb_tfidf_preds), "Precision": precision_score(y_test, xgb_tfidf_preds), "Recall": recall_score(y_test, xgb_tfidf_preds), "AUC-ROC": roc_auc_score(y_test, xgb_tfidf_proba)})
print(f"  Acc: {results[-1]['Accuracy']:.4f}  F1: {results[-1]['F1']:.4f}  AUC: {results[-1]['AUC-ROC']:.4f}")

results_df = pd.DataFrame(results)
print("\n" + "=" * 90)
print("BASELINE RESULTS")
print("=" * 90)
print(results_df.to_string(index=False, float_format="{:.4f}".format))
results_df.to_csv(SPLIT_DIR / "baseline_results.csv", index=False)
