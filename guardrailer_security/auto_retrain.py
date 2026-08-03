"""
auto_retrain.py
Automatic incremental retraining system for Guardrailer.

Monitors feedback log for new unique attacks. When the buffer reaches
50 unique samples, triggers GPU-accelerated incremental training on
the RTX 4060. Only trains on new samples — never retrains from scratch.

Architecture:
    feedback_log.jsonl → AttackBuffer (50 unique) → IncrementalTrainer → hot-swap
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTO_RETRAIN_CONFIG = {
    "buffer_capacity": 50,           # trigger training at this many unique samples
    "poll_interval_seconds": 60,     # how often to check for new feedback
    "min_feedback_for_processing": 1, # minimum feedback entries to process
    "dedup_similarity_threshold": 0.85, # MinHash similarity for deduplication
    "gpu_device": "cuda:0",          # RTX 4060
    "incremental_lr": 0.001,         # learning rate for incremental fine-tuning
    "incremental_epochs": 20,        # epochs for incremental training
    "calibration_retrain": True,     # always retrain calibrator on full data
    "model_save_dir": Path(__file__).resolve().parent / "models",
    "buffer_file": Path(__file__).resolve().parent / "auto_retrain_buffer.json",
    "stats_file": Path(__file__).resolve().parent / "auto_retrain_stats.json",
    "processed_ids_file": Path(__file__).resolve().parent / "auto_retrain_processed.json",
    "feedback_dir": Path(__file__).resolve().parent / "feedback_data",
}

FEATURE_NAMES = [
    'dense', 'sparse_idf', 'centroid', 'cross_encoder',
    'perplexity', 'entropy', 'token_frequency', 'ngram_overlap',
    'uniqueness', 'length_norm',
]


# ---------------------------------------------------------------------------
# MinHash for deduplication
# ---------------------------------------------------------------------------

class MinHashDedup:
    """MinHash-based deduplication for attack samples."""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.max_hash = (1 << 32) - 1

    def _hash_func(self, x: int, seed: int) -> int:
        h = hashlib.md5(f"{seed}:{x}".encode()).digest()
        return int.from_bytes(h[:4], byteorder='big')

    def compute_signature(self, text: str, k: int = 3) -> list[int]:
        text = text.lower().strip()
        shingles = set()
        for i in range(max(0, len(text) - k + 1)):
            shingles.add(hash(text[i:i + k]))
        if not shingles:
            shingles = {hash(text)}

        signature = []
        for i in range(self.num_perm):
            min_hash = self.max_hash
            for s in shingles:
                h = self._hash_func(s, i)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    @staticmethod
    def jaccard_similarity(sig1: list[int], sig2: list[int]) -> float:
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Attack Buffer
# ---------------------------------------------------------------------------

@dataclass
class AttackSample:
    """A single attack sample in the buffer."""
    sample_id: str
    text: str
    label: int  # 0=benign, 1=malicious
    attack_category: str
    risk_level: str
    embedding: Optional[list] = None
    signals: Optional[dict] = None
    timestamp: str = ""
    source_feedback_id: str = ""


class AttackBuffer:
    """
    Circular buffer holding up to 50 unique attack samples.
    Uses MinHash for deduplication — only genuinely new attacks are added.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.samples: list[AttackSample] = []
        self.text_hashes: set[str] = set()
        self.minhash_sigs: list[tuple[str, list[int]]] = []
        self.dedup = MinHashDedup()
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load buffer from disk."""
        config = AUTO_RETRAIN_CONFIG
        if config["buffer_file"].exists():
            try:
                with open(config["buffer_file"]) as f:
                    data = json.load(f)
                self.samples = [AttackSample(**s) for s in data.get("samples", [])]
                self.text_hashes = set(data.get("text_hashes", []))
                self.minhash_sigs = [
                    (h, sig) for h, sig in data.get("minhash_sigs", [])
                ]
                log.info("Loaded attack buffer: %d/%d samples", len(self.samples), self.capacity)
            except Exception as e:
                log.warning("Failed to load attack buffer: %s", e)
                self.samples = []
                self.text_hashes = set()
                self.minhash_sigs = []

    def _save(self):
        """Save buffer to disk."""
        config = AUTO_RETRAIN_CONFIG
        data = {
            "capacity": self.capacity,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "text": s.text,
                    "label": s.label,
                    "attack_category": s.attack_category,
                    "risk_level": s.risk_level,
                    "embedding": s.embedding,
                    "signals": s.signals,
                    "timestamp": s.timestamp,
                    "source_feedback_id": s.source_feedback_id,
                }
                for s in self.samples
            ],
            "text_hashes": list(self.text_hashes),
            "minhash_sigs": self.minhash_sigs,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = str(config["buffer_file"]) + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, str(config["buffer_file"]))

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate of existing buffer content."""
        text_hash = MinHashDedup.text_hash(text)
        if text_hash in self.text_hashes:
            return True

        sig = self.dedup.compute_signature(text)
        for _, existing_sig in self.minhash_sigs:
            sim = MinHashDedup.jaccard_similarity(sig, existing_sig)
            if sim >= AUTO_RETRAIN_CONFIG["dedup_similarity_threshold"]:
                return True
        return False

    def add_sample(
        self,
        text: str,
        label: int,
        attack_category: str,
        risk_level: str,
        embedding: Optional[list] = None,
        signals: Optional[dict] = None,
        source_feedback_id: str = "",
    ) -> tuple[bool, str]:
        """
        Add a sample to the buffer if unique.
        Returns (was_added, reason).
        """
        with self._lock:
            if len(self.samples) >= self.capacity:
                return False, f"Buffer full ({self.capacity}/{self.capacity})"

            if self.is_duplicate(text):
                return False, "Duplicate sample"

            sample_id = hashlib.sha256(
                f"{text}:{time.time()}".encode()
            ).hexdigest()[:12]

            sample = AttackSample(
                sample_id=sample_id,
                text=text[:5000],
                label=label,
                attack_category=attack_category,
                risk_level=risk_level,
                embedding=embedding,
                signals=signals,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_feedback_id=source_feedback_id,
            )

            text_hash = MinHashDedup.text_hash(text)
            sig = self.dedup.compute_signature(text)

            self.samples.append(sample)
            self.text_hashes.add(text_hash)
            self.minhash_sigs.append((text_hash, sig))
            self._save()

            log.info(
                "Buffer: %d/%d (+1 unique sample, category=%s)",
                len(self.samples), self.capacity, attack_category
            )
            return True, "Added"

    def get_all_samples(self) -> list[AttackSample]:
        """Get all samples in the buffer."""
        with self._lock:
            return list(self.samples)

    def clear(self):
        """Clear the buffer after successful training."""
        with self._lock:
            self.samples = []
            self.text_hashes = set()
            self.minhash_sigs = []
            self._save()
            log.info("Attack buffer cleared")

    @property
    def is_full(self) -> bool:
        return len(self.samples) >= self.capacity

    @property
    def fill_level(self) -> float:
        return len(self.samples) / self.capacity

    def get_stats(self) -> dict:
        with self._lock:
            categories = {}
            for s in self.samples:
                cat = s.attack_category or "unknown"
                categories[cat] = categories.get(cat, 0) + 1
            return {
                "buffer_size": len(self.samples),
                "capacity": self.capacity,
                "fill_level": f"{self.fill_level:.0%}",
                "is_full": self.is_full,
                "categories": categories,
            }


# ---------------------------------------------------------------------------
# Feature Extraction (reuse existing pipeline)
# ---------------------------------------------------------------------------

def extract_signals_for_sample(text: str, dense_score: float = 0.5) -> dict:
    """Extract 10-dimensional signal features for a text sample."""
    try:
        from scoring import (
            compute_text_idf_score,
            compute_length_normalization,
            load_corpus_meta,
        )
        from improved_scoring import (
            compute_perplexity_score,
            compute_entropy_score,
            compute_token_frequency_score,
            compute_ngram_overlap_score,
        )

        meta = load_corpus_meta()
        s_sparse = compute_text_idf_score(text, meta)
        s_length = compute_length_normalization(len(text), meta)
        s_perplexity = compute_perplexity_score(text)
        s_entropy = compute_entropy_score(text)
        s_token_freq = compute_token_frequency_score(text)
        s_ngram = compute_ngram_overlap_score(text)

        return {
            "dense": dense_score,
            "sparse_idf": s_sparse,
            "centroid": 0.0,
            "cross_encoder": 0.0,
            "perplexity": s_perplexity,
            "entropy": s_entropy,
            "token_frequency": s_token_freq,
            "ngram_overlap": s_ngram,
            "uniqueness": 0.5,
            "length_norm": s_length,
        }
    except Exception as e:
        log.warning("Signal extraction failed: %s", e)
        return {name: 0.0 for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Incremental Trainer (GPU-accelerated)
# ---------------------------------------------------------------------------

class IncrementalTrainer:
    """
    Incremental training — only trains on new samples.
    Uses warm-start for logistic regression and fine-tuning for neural models.
    Runs on RTX 4060 via CUDA.
    """

    def __init__(self):
        self.device = self._detect_gpu()
        self.stats = self._load_stats()

    def _detect_gpu(self) -> str:
        """Detect RTX 4060 GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                log.info("GPU detected: %s", gpu_name)
                return "cuda:0"
        except ImportError:
            pass
        log.warning("No GPU detected, falling back to CPU")
        return "cpu"

    def _load_stats(self) -> dict:
        config = AUTO_RETRAIN_CONFIG
        if config["stats_file"].exists():
            try:
                with open(config["stats_file"]) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_retrains": 0,
            "total_samples_trained": 0,
            "last_retrain_time": None,
            "last_retrain_samples": 0,
            "training_history": [],
        }

    def _save_stats(self):
        config = AUTO_RETRAIN_CONFIG
        with open(config["stats_file"], 'w') as f:
            json.dump(self.stats, f, indent=2)

    def train_incremental(self, buffer: AttackBuffer) -> dict:
        """
        Train models incrementally on buffer samples.
        Only updates weights for new data — no full retrain.
        """
        samples = buffer.get_all_samples()
        if len(samples) < 5:
            return {"status": "skipped", "reason": f"Only {len(samples)} samples, need >= 5"}

        log.info("=" * 60)
        log.info("INCREMENTAL TRAINING STARTED")
        log.info("  Samples: %d", len(samples))
        log.info("  Device: %s", self.device)
        log.info("=" * 60)

        t0 = time.time()

        # Extract features
        X = []
        y = []
        for s in samples:
            signals = s.signals or extract_signals_for_sample(s.text)
            features = [signals.get(name, 0.0) for name in FEATURE_NAMES]
            X.append(features)
            y.append(s.label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Z-score normalize using running statistics
        X = self._normalize_features(X)

        results = {}

        # 1. Incremental logistic regression (warm-start)
        try:
            r = self._train_logistic_incremental(X, y)
            results["logistic"] = r
        except Exception as e:
            log.error("Logistic incremental training failed: %s", e)
            results["logistic"] = {"status": "error", "error": str(e)}

        # 2. Incremental neural network (fine-tune)
        try:
            r = self._train_neural_incremental(X, y)
            results["neural"] = r
        except Exception as e:
            log.error("Neural incremental training failed: %s", e)
            results["neural"] = {"status": "error", "error": str(e)}

        # 3. Incremental attention model (fine-tune on GPU)
        try:
            r = self._train_attention_incremental(X, y)
            results["attention"] = r
        except Exception as e:
            log.error("Attention incremental training failed: %s", e)
            results["attention"] = {"status": "error", "error": str(e)}

        # 4. Retrain calibrator (cheap, always from scratch)
        try:
            r = self._train_calibrator(X, y)
            results["calibrator"] = r
        except Exception as e:
            log.error("Calibrator training failed: %s", e)
            results["calibrator"] = {"status": "error", "error": str(e)}

        elapsed = time.time() - t0

        # Update stats
        self.stats["total_retrains"] += 1
        self.stats["total_samples_trained"] += len(samples)
        self.stats["last_retrain_time"] = datetime.now(timezone.utc).isoformat()
        self.stats["last_retrain_samples"] = len(samples)
        self.stats["training_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "samples": len(samples),
            "elapsed_seconds": round(elapsed, 2),
            "device": self.device,
            "results": {k: v.get("status", "unknown") for k, v in results.items()},
        })
        self._save_stats()

        log.info("=" * 60)
        log.info("INCREMENTAL TRAINING COMPLETE")
        log.info("  Time: %.1fs", elapsed)
        log.info("  Results: %s", {k: v.get("status") for k, v in results.items()})
        log.info("=" * 60)

        return {
            "status": "completed",
            "samples_trained": len(samples),
            "elapsed_seconds": round(elapsed, 2),
            "device": self.device,
            "results": results,
        }

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize features using running statistics."""
        config = AUTO_RETRAIN_CONFIG
        stats_path = config["model_save_dir"] / "feature_stats.json"

        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                mean = np.array(stats["mean"])
                std = np.array(stats["std"])
                std = np.where(std < 1e-8, 1.0, std)
                return (X - mean) / std
            except Exception:
                pass

        # Compute from current data
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        config["model_save_dir"].mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

        return (X - mean) / std

    def _train_logistic_incremental(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Incremental logistic regression using warm_start."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        config = AUTO_RETRAIN_CONFIG
        model_path = config["model_save_dir"] / "logistic_weights.json"

        # Load existing model if available
        existing_model = None
        if model_path.exists():
            try:
                with open(model_path) as f:
                    data = json.load(f)
                existing_model = LogisticRegression(max_iter=1000, warm_start=True)
                existing_model.coef_ = np.array(data["coef"])
                existing_model.intercept_ = np.array(data["intercept"])
                existing_model.classes_ = np.array([0, 1])
                existing_model.n_features_in_ = X.shape[1]
            except Exception:
                existing_model = None

        if existing_model is not None:
            # Warm-start: continue training from previous weights
            model = existing_model
            model.max_iter = 50  # few more iterations
            model.classes_ = np.array([0, 1])
            try:
                model.fit(X, y)
            except Exception:
                # If warm_start fails, train fresh
                model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
                model.fit(X, y)
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            model.fit(X, y)

        # Save
        data = {
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "scaler_mean": [0.0] * X.shape[1],
            "scaler_scale": [1.0] * X.shape[1],
            "feature_names": FEATURE_NAMES,
        }
        with open(model_path, 'w') as f:
            json.dump(data, f)

        log.info("  Logistic: trained (coef norm=%.4f)", np.linalg.norm(model.coef_))
        return {"status": "completed", "model": "logistic"}

    def _train_neural_incremental(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Incremental neural network — fine-tune existing or train new."""
        from sklearn.neural_network import MLPClassifier

        config = AUTO_RETRAIN_CONFIG
        model_path = config["model_save_dir"] / "neural_weights.json"

        # Try to load existing model for fine-tuning
        existing_model = None
        if model_path.exists():
            try:
                with open(model_path) as f:
                    data = json.load(f)
                existing_model = MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    max_iter=100,  # few iterations for fine-tuning
                    random_state=42,
                )
                existing_model.coefs_ = [np.array(c) for c in data["coefs"]]
                existing_model.intercepts_ = [np.array(i) for i in data["intercepts"]]
                existing_model.classes_ = np.array([0, 1])
                existing_model.n_layers_ = len(data["coefs"]) + 1
                existing_model.n_features_in_ = X.shape[1]
            except Exception:
                existing_model = None

        if existing_model is not None:
            model = existing_model
            try:
                model.fit(X, y)
            except Exception:
                model = MLPClassifier(
                    hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
                    early_stopping=True, validation_fraction=0.2,
                )
                model.fit(X, y)
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(32, 16), max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.2,
            )
            model.fit(X, y)

        # Save
        data = {
            "coefs": [c.tolist() for c in model.coefs_],
            "intercepts": [i.tolist() for i in model.intercepts_],
            "scaler_mean": [0.0] * X.shape[1],
            "scaler_scale": [1.0] * X.shape[1],
            "feature_names": FEATURE_NAMES,
        }
        with open(model_path, 'w') as f:
            json.dump(data, f)

        log.info("  Neural: trained (layers=%d)", len(model.coefs_))
        return {"status": "completed", "model": "neural"}

    def _train_attention_incremental(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Incremental attention model — fine-tune on RTX 4060."""
        config = AUTO_RETRAIN_CONFIG

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            return {"status": "skipped", "reason": "PyTorch not installed"}

        device = torch.device(self.device)

        # Define model architecture
        class AttentionWeightNet(nn.Module):
            def __init__(self, num_signals: int = 10):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=num_signals, num_heads=2, batch_first=True
                )
                self.fc = nn.Sequential(
                    nn.Linear(num_signals, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, num_signals),
                    nn.Softmax(dim=-1)
                )

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                attn_out, _ = self.attention(x, x, x)
                weights = self.fc(attn_out.squeeze(1))
                return weights

        model_path = config["model_save_dir"] / "attention_weights.pt"
        model = AttentionWeightNet(num_signals=len(FEATURE_NAMES)).to(device)

        # Load existing weights for fine-tuning
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                log.info("  Attention: loaded existing weights for fine-tuning")
            except Exception as e:
                log.warning("  Attention: could not load existing weights: %s", e)

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Fine-tune with small learning rate
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["incremental_lr"],  # small LR for fine-tuning
            weight_decay=1e-4
        )
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(config["incremental_epochs"]):
            optimizer.zero_grad()

            # Forward pass: compute attention weights
            attn_weights = model(X_tensor)

            # Weighted sum of features
            weighted_features = (X_tensor * attn_weights).sum(dim=-1)

            # Binary classification
            output = torch.sigmoid(weighted_features)
            loss = criterion(output, y_tensor)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                log.info("    Epoch %d/%d: loss=%.4f", epoch + 1, config["incremental_epochs"], loss.item())

        # Save
        config["model_save_dir"].mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        log.info("  Attention: fine-tuned on %s (%d epochs)", device, config["incremental_epochs"])

        return {"status": "completed", "model": "attention", "device": str(device)}

    def _train_calibrator(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Retrain probability calibrator — always from scratch (cheap)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        config = AUTO_RETRAIN_CONFIG
        cal_path = config["model_save_dir"] / "calibrator.json"

        # Compute raw composite scores
        try:
            from improved_scoring import DEFAULT_LEARNED_WEIGHTS
            w = np.array([DEFAULT_LEARNED_WEIGHTS.get(name, 0.0) for name in FEATURE_NAMES])
            raw_scores = X @ w
        except Exception:
            raw_scores = X.mean(axis=1)

        # Fit Platt scaling
        base_lr = LogisticRegression(max_iter=1000)
        try:
            calibrator = CalibratedClassifierCV(base_lr, cv=min(5, len(y) // 2 + 1), method='sigmoid')
            calibrator.fit(raw_scores.reshape(-1, 1), y)

            # Extract calibration params
            calibrator_lr = calibrator.calibrated_classifiers_[0].calibrators_[0]
            k = float(calibrator_lr.coef_[0][0])
            b = float(calibrator_lr.intercept_[0])

            data = {
                "method": "platt",
                "k": k,
                "b": b,
                "isotonic_x": [],
                "isotonic_y": [],
            }
        except Exception:
            data = {
                "method": "sigmoid_fallback",
                "k": 10.0,
                "b": -5.0,
                "isotonic_x": [],
                "isotonic_y": [],
            }

        with open(cal_path, 'w') as f:
            json.dump(data, f)

        log.info("  Calibrator: trained (method=%s)", data["method"])
        return {"status": "completed", "model": "calibrator", "method": data["method"]}


# ---------------------------------------------------------------------------
# Auto-Retrain Worker
# ---------------------------------------------------------------------------

class AutoRetrainWorker:
    """
    Background worker that monitors feedback and triggers incremental training.

    Flow:
        1. Poll feedback_log.jsonl every 60s
        2. Extract new FP/FN corrections
        3. Deduplicate and add to AttackBuffer (capacity=50)
        4. When buffer is full → trigger IncrementalTrainer
        5. On success → clear buffer, update stats
    """

    def __init__(self):
        self.buffer = AttackBuffer(capacity=AUTO_RETRAIN_CONFIG["buffer_capacity"])
        self.trainer = IncrementalTrainer()
        self.processed_ids: set[str] = set()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load_processed_ids()

    def _load_processed_ids(self):
        config = AUTO_RETRAIN_CONFIG
        if config["processed_ids_file"].exists():
            try:
                with open(config["processed_ids_file"]) as f:
                    self.processed_ids = set(json.load(f))
            except Exception:
                self.processed_ids = set()

    def _save_processed_ids(self):
        config = AUTO_RETRAIN_CONFIG
        with open(config["processed_ids_file"], 'w') as f:
            json.dump(list(self.processed_ids), f)

    def start(self):
        """Start the background worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Auto-retrain worker started (poll=%ds, buffer=%d)",
                 AUTO_RETRAIN_CONFIG["poll_interval_seconds"],
                 AUTO_RETRAIN_CONFIG["buffer_capacity"])

    def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Auto-retrain worker stopped")

    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                self._process_new_feedback()
                self._check_and_train()
            except Exception as e:
                log.error("Auto-retrain poll error: %s", e)
            time.sleep(AUTO_RETRAIN_CONFIG["poll_interval_seconds"])

    def _process_new_feedback(self):
        """Read new feedback entries and add unique attacks to buffer."""
        config = AUTO_RETRAIN_CONFIG
        feedback_file = config["feedback_dir"] / "feedback_log.jsonl"

        if not feedback_file.exists():
            return

        new_count = 0
        with open(feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_id = entry.get("id", "")
                if entry_id in self.processed_ids:
                    continue

                is_fp = entry.get("is_false_positive", False)
                is_fn = entry.get("is_false_negative", False)
                if not is_fp and not is_fn:
                    self.processed_ids.add(entry_id)
                    continue

                query = entry.get("query", "").strip()
                if not query or len(query) < 10:
                    self.processed_ids.add(entry_id)
                    continue

                # Determine label and category
                if is_fn:
                    label = 1  # was allowed, actually malicious
                    category = entry.get("actual_category") or entry.get("attack_category") or "unknown"
                    risk = entry.get("risk_level") or "medium"
                else:
                    label = 0  # was blocked, actually benign
                    category = "benign_control"
                    risk = "none"

                # Extract signals
                signals = extract_signals_for_sample(query)

                # Add to buffer
                added, reason = self.buffer.add_sample(
                    text=query,
                    label=label,
                    attack_category=category,
                    risk_level=risk,
                    signals=signals,
                    source_feedback_id=entry_id,
                )

                self.processed_ids.add(entry_id)
                if added:
                    new_count += 1

        if new_count > 0:
            self._save_processed_ids()
            log.info("Processed %d new feedback entries (%d added to buffer)",
                     new_count, self.buffer.fill_level)

    def _check_and_train(self):
        """Check if buffer is full and trigger training."""
        if not self.buffer.is_full:
            return

        log.info("=" * 60)
        log.info("BUFFER FULL (%d/%d) — TRIGGERING INCREMENTAL TRAINING",
                 len(self.buffer.samples), self.buffer.capacity)
        log.info("=" * 60)

        result = self.trainer.train_incremental(self.buffer)

        if result.get("status") == "completed":
            self.buffer.clear()
            log.info("Training complete, buffer cleared")
        else:
            log.error("Training failed: %s", result)

    def get_status(self) -> dict:
        """Get current status of the auto-retrain system."""
        return {
            "running": self._running,
            "buffer": self.buffer.get_stats(),
            "trainer": {
                "device": self.trainer.device,
                "stats": self.trainer.stats,
            },
            "processed_ids_count": len(self.processed_ids),
            "config": {
                "buffer_capacity": AUTO_RETRAIN_CONFIG["buffer_capacity"],
                "poll_interval": AUTO_RETRAIN_CONFIG["poll_interval_seconds"],
                "dedup_threshold": AUTO_RETRAIN_CONFIG["dedup_similarity_threshold"],
            },
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_auto_retrain_worker: Optional[AutoRetrainWorker] = None


def get_auto_retrain_worker() -> AutoRetrainWorker:
    """Get or create the auto-retrain worker singleton."""
    global _auto_retrain_worker
    if _auto_retrain_worker is None:
        _auto_retrain_worker = AutoRetrainWorker()
    return _auto_retrain_worker


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run auto-retrain worker standalone."""
    import argparse
    parser = argparse.ArgumentParser(description="Guardrailer Auto-Retrain Worker")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--train-now", action="store_true", help="Force train on current buffer")
    parser.add_argument("--clear-buffer", action="store_true", help="Clear the attack buffer")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()

    worker = get_auto_retrain_worker()

    if args.status:
        print(json.dumps(worker.get_status(), indent=2))
        return

    if args.clear_buffer:
        worker.buffer.clear()
        print("Buffer cleared")
        return

    if args.train_now:
        result = worker.trainer.train_incremental(worker.buffer)
        print(json.dumps(result, indent=2, default=str))
        if result.get("status") == "completed":
            worker.buffer.clear()
        return

    AUTO_RETRAIN_CONFIG["poll_interval_seconds"] = args.poll
    worker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
        print("Stopped")


if __name__ == "__main__":
    main()
