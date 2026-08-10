"""
Hybrid Lightweight Scorer — Industry-Grade Main Module.

Combines four layers for fast, robust prompt classification:
    Layer 1: Pattern matching (instant, rule-based)
    Layer 2: TF-IDF similarity (word + char level, sparse, CPU-efficient)
    Layer 3: Feature-based stacking ensemble (XGBoost + LightGBM + RandomForest)
    Layer 4: Confidence calibration (isotonic regression)

No vector database needed. ~20MB total model size.
"""

import gc
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    from .features import extract_features, get_feature_names
    from .patterns import pattern_classify, pattern_score
    from .similarity import TFIDFSimilarity, IDFWeightedKeywords
    from .calibration import ConfidenceCalibrator
except ImportError:
    from features import extract_features, get_feature_names
    from patterns import pattern_classify, pattern_score
    from similarity import TFIDFSimilarity, IDFWeightedKeywords
    from calibration import ConfidenceCalibrator


class HybridScorer:
    """
    Industry-grade hybrid prompt guardrails scorer.

    Uses stacking ensemble with calibration for robust classification.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.pattern_threshold = 0.5
        self.similarity_threshold = 0.5
        self.classifier_threshold = 0.5
        self.ensemble_weights = {"pattern": 0.20, "similarity": 0.15, "classifier": 0.65}

        self.similarity_scorer = TFIDFSimilarity()
        self.keyword_scorer = IDFWeightedKeywords()
        self.classifier = None
        self.calibrator = None
        self.feature_names = get_feature_names()
        self.fitted = False

        if model_dir and Path(model_dir).exists():
            self.load(model_dir)

    def fit(self, texts: list, labels: list, model_dir: Optional[Path] = None):
        labels = np.array(labels)

        print("Layer 2: Fitting TF-IDF similarity scorer...")
        self.similarity_scorer.fit(texts, labels)
        gc.collect()

        print("Layer 2: Fitting IDF keyword scorer...")
        self.keyword_scorer.fit(texts, labels)
        gc.collect()

        print("Layer 3: Extracting features...")
        feature_dicts = [extract_features(t) for t in texts]
        feature_names = sorted(feature_dicts[0].keys())
        X_hand = np.array([[fd[k] for k in feature_names] for fd in feature_dicts], dtype=np.float32)
        del feature_dicts
        gc.collect()

        print("Layer 3: Building word-level TF-IDF feature matrix...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack as sparse_hstack, csr_matrix

        tfidf_word = TfidfVectorizer(
            max_features=50000, sublinear_tf=True, norm="l2",
            ngram_range=(1, 2), dtype=np.float32,
        )
        X_tfidf_word = tfidf_word.fit_transform(texts)
        print(f"  Word TF-IDF shape: {X_tfidf_word.shape}")

        print("Layer 3: Building char-level TF-IDF feature matrix...")
        tfidf_char = TfidfVectorizer(
            max_features=30000, analyzer="char_wb",
            ngram_range=(3, 5), sublinear_tf=True, norm="l2",
            dtype=np.float32,
        )
        X_tfidf_char = tfidf_char.fit_transform(texts)
        print(f"  Char TF-IDF shape: {X_tfidf_char.shape}")

        X_hand_sparse = csr_matrix(X_hand, dtype=np.float32)
        X_combined = sparse_hstack(
            [X_hand_sparse, X_tfidf_word, X_tfidf_char], format="csr"
        )
        del X_hand, X_hand_sparse, X_tfidf_word, X_tfidf_char
        gc.collect()
        print(f"  Combined feature matrix: {X_combined.shape}")

        print("Layer 3: Training stacking ensemble...")
        self.classifier = self._build_stacking_ensemble()
        self.classifier.fit(X_combined, labels)

        self._tfidf_word = tfidf_word
        self._tfidf_char = tfidf_char
        self._feature_names = feature_names

        self.fitted = True

        if model_dir:
            self.save(model_dir)

        del X_combined
        gc.collect()
        return self

    def fit_calibrator(self, texts: list, labels: list, method: str = "isotonic"):
        print("Layer 4: Fitting confidence calibration...")
        labels_arr = np.array(labels)

        feature_dicts = [extract_features(t) for t in texts]
        feature_names = sorted(feature_dicts[0].keys())
        X_hand = np.array([[fd[k] for k in feature_names] for fd in feature_dicts], dtype=np.float32)
        del feature_dicts
        gc.collect()

        from scipy.sparse import hstack as sparse_hstack, csr_matrix
        X_tfidf_word = self._tfidf_word.transform(texts)
        X_tfidf_char = self._tfidf_char.transform(texts)
        X_hand_sparse = csr_matrix(X_hand, dtype=np.float32)
        X_combined = sparse_hstack(
            [X_hand_sparse, X_tfidf_word, X_tfidf_char], format="csr"
        )
        del X_hand, X_hand_sparse, X_tfidf_word, X_tfidf_char
        gc.collect()

        self.calibrator = ConfidenceCalibrator(self.classifier)
        self.calibrator.fit(X_combined, labels_arr, method=method)
        print(f"  Calibration fitted with method={method}")
        return self

    def _build_stacking_ensemble(self):
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = []

        try:
            import xgboost as xgb
            xgb_clf = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.5,
                reg_alpha=1.0,
                reg_lambda=1.0,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            estimators.append(("xgb", xgb_clf))
            print("  Added XGBoost to ensemble")
        except ImportError:
            print("  XGBoost not available, skipping")

        try:
            import lightgbm as lgb
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=1.0,
                reg_lambda=1.0,
                objective="binary",
                metric="binary_logloss",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            estimators.append(("lgbm", lgb_clf))
            print("  Added LightGBM to ensemble")
        except ImportError:
            print("  LightGBM not available, skipping")

        rf_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        estimators.append(("rf", rf_clf))
        print("  Added RandomForest to ensemble")

        if len(estimators) == 0:
            raise RuntimeError("No classifiers available")

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=True,
        )

        return stacking

    def _extract_and_combine(self, texts: list):
        from scipy.sparse import hstack as sparse_hstack, csr_matrix

        feature_dicts = [extract_features(t) for t in texts]
        feature_names = sorted(feature_dicts[0].keys())
        X_hand = np.array([[fd[k] for k in feature_names] for fd in feature_dicts], dtype=np.float32)
        del feature_dicts

        X_tfidf_word = self._tfidf_word.transform(texts)
        X_tfidf_char = self._tfidf_char.transform(texts)
        X_hand_sparse = csr_matrix(X_hand, dtype=np.float32)
        X_combined = sparse_hstack(
            [X_hand_sparse, X_tfidf_word, X_tfidf_char], format="csr"
        )
        del X_hand, X_hand_sparse, X_tfidf_word, X_tfidf_char
        gc.collect()

        return X_combined

    def score(self, text: str) -> Dict:
        pattern_s, pattern_details = pattern_score(text)

        sim_s = self.similarity_scorer.score(text)
        kw_s = self.keyword_scorer.score(text)
        similarity_s = max(sim_s, kw_s)

        classifier_s = 0.5
        if self.classifier is not None and hasattr(self, '_tfidf_word'):
            X_combined = self._extract_and_combine([text])
            proba = self.classifier.predict_proba(X_combined)
            classifier_s = float(proba[0][1])

        if self.calibrator is not None and hasattr(self, '_tfidf_word'):
            X_combined = self._extract_and_combine([text])
            proba = self.calibrator.predict_proba(X_combined)
            calibrated_s = float(proba[0][1])
        else:
            calibrated_s = classifier_s

        ensemble_s = (
            self.ensemble_weights["pattern"] * pattern_s +
            self.ensemble_weights["similarity"] * similarity_s +
            self.ensemble_weights["classifier"] * calibrated_s
        )

        is_malicious = ensemble_s >= 0.5

        if ensemble_s >= 0.8:
            threat_level = "critical"
        elif ensemble_s >= 0.6:
            threat_level = "high"
        elif ensemble_s >= 0.4:
            threat_level = "medium"
        elif ensemble_s >= 0.2:
            threat_level = "low"
        else:
            threat_level = "safe"

        return {
            "is_malicious": is_malicious,
            "score": round(ensemble_s, 4),
            "pattern_score": round(pattern_s, 4),
            "similarity_score": round(similarity_s, 4),
            "classifier_score": round(classifier_s, 4),
            "calibrated_score": round(calibrated_s, 4),
            "threat_level": threat_level,
            "details": pattern_details,
        }

    def predict(self, texts: list, batch_size: int = 500) -> list:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend([self.score(t)["is_malicious"] for t in batch])
            gc.collect()
        return results

    def predict_batch(self, texts: list) -> list:
        if self.classifier is None or not hasattr(self, '_tfidf_word'):
            return [self.score(t)["is_malicious"] for t in texts]

        from scipy.sparse import hstack as sparse_hstack, csr_matrix

        feature_dicts = [extract_features(t) for t in texts]
        feature_names = sorted(feature_dicts[0].keys())
        X_hand = np.array([[fd[k] for k in feature_names] for fd in feature_dicts], dtype=np.float32)
        del feature_dicts

        X_tfidf_word = self._tfidf_word.transform(texts)
        X_tfidf_char = self._tfidf_char.transform(texts)
        X_hand_sparse = csr_matrix(X_hand, dtype=np.float32)
        X_combined = sparse_hstack(
            [X_hand_sparse, X_tfidf_word, X_tfidf_char], format="csr"
        )
        del X_hand, X_hand_sparse, X_tfidf_word, X_tfidf_char
        gc.collect()

        proba = self.classifier.predict_proba(X_combined)[:, 1]
        pattern_scores = np.array([pattern_score(t)[0] for t in texts])
        sim_scores = np.array([max(self.similarity_scorer.score(t), self.keyword_scorer.score(t)) for t in texts])

        ensemble_scores = (
            self.ensemble_weights["pattern"] * pattern_scores +
            self.ensemble_weights["similarity"] * sim_scores +
            self.ensemble_weights["classifier"] * proba
        )

        return [bool(s >= 0.5) for s in ensemble_scores]

    def predict_proba_batch(self, texts: list) -> np.ndarray:
        if self.classifier is None or not hasattr(self, '_tfidf_word'):
            return np.array([self.score(t)["score"] for t in texts])

        from scipy.sparse import hstack as sparse_hstack, csr_matrix

        feature_dicts = [extract_features(t) for t in texts]
        feature_names = sorted(feature_dicts[0].keys())
        X_hand = np.array([[fd[k] for k in feature_names] for fd in feature_dicts], dtype=np.float32)
        del feature_dicts

        X_tfidf_word = self._tfidf_word.transform(texts)
        X_tfidf_char = self._tfidf_char.transform(texts)
        X_hand_sparse = csr_matrix(X_hand, dtype=np.float32)
        X_combined = sparse_hstack(
            [X_hand_sparse, X_tfidf_word, X_tfidf_char], format="csr"
        )
        del X_hand, X_hand_sparse, X_tfidf_word, X_tfidf_char
        gc.collect()

        proba = self.classifier.predict_proba(X_combined)[:, 1]
        pattern_scores = np.array([pattern_score(t)[0] for t in texts])
        sim_scores = np.array([max(self.similarity_scorer.score(t), self.keyword_scorer.score(t)) for t in texts])

        ensemble_scores = (
            self.ensemble_weights["pattern"] * pattern_scores +
            self.ensemble_weights["similarity"] * sim_scores +
            self.ensemble_weights["classifier"] * proba
        )

        return ensemble_scores

    def save(self, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, model_dir / "classifier.joblib", compress=3)
        if hasattr(self, '_tfidf_word'):
            joblib.dump(self._tfidf_word, model_dir / "tfidf_word.joblib", compress=3)
        if hasattr(self, '_tfidf_char'):
            joblib.dump(self._tfidf_char, model_dir / "tfidf_char.joblib", compress=3)
        self.similarity_scorer.save(model_dir / "similarity.joblib")
        self.keyword_scorer.save(model_dir / "keywords.joblib")
        joblib.dump(self.feature_names, model_dir / "feature_names.joblib")
        if self.calibrator is not None:
            self.calibrator.save(model_dir / "calibrator.joblib")
        joblib.dump({
            "pattern_threshold": self.pattern_threshold,
            "similarity_threshold": self.similarity_threshold,
            "classifier_threshold": self.classifier_threshold,
            "ensemble_weights": self.ensemble_weights,
        }, model_dir / "config.joblib")

        print(f"Models saved to {model_dir}")

    def load(self, model_dir: Path):
        model_dir = Path(model_dir)

        self.classifier = joblib.load(model_dir / "classifier.joblib")
        tfidf_word_path = model_dir / "tfidf_word.joblib"
        if tfidf_word_path.exists():
            self._tfidf_word = joblib.load(tfidf_word_path)
        tfidf_char_path = model_dir / "tfidf_char.joblib"
        if tfidf_char_path.exists():
            self._tfidf_char = joblib.load(tfidf_char_path)
        self.similarity_scorer = TFIDFSimilarity()
        self.similarity_scorer.load(model_dir / "similarity.joblib")
        self.keyword_scorer = IDFWeightedKeywords()
        self.keyword_scorer.load(model_dir / "keywords.joblib")
        self.feature_names = joblib.load(model_dir / "feature_names.joblib")
        config = joblib.load(model_dir / "config.joblib")

        self.pattern_threshold = config["pattern_threshold"]
        self.similarity_threshold = config["similarity_threshold"]
        self.classifier_threshold = config["classifier_threshold"]
        self.ensemble_weights = config["ensemble_weights"]

        calibrator_path = model_dir / "calibrator.joblib"
        if calibrator_path.exists():
            self.calibrator = ConfidenceCalibrator()
            self.calibrator.load(calibrator_path)

        self.fitted = True

    def get_model_size(self, model_dir: Path) -> Dict[str, float]:
        model_dir = Path(model_dir)
        sizes = {}
        for f in model_dir.glob("*.joblib"):
            sizes[f.stem] = f.stat().st_size / 1e6
        sizes["total"] = sum(sizes.values())
        return sizes
