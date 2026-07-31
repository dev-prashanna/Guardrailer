"""
improved_scoring.py
Phase 3: Improved Multi-Signal Scoring with Learned Weights

Enhanced scoring system with:
- Learned weights via logistic regression, neural network, attention-based
- New signals: perplexity, entropy, token frequency, n-gram overlap
- Threshold calibration: Platt scaling, isotonic regression
- Ensemble scoring: multiple models, stacking/blending
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

SKLEARN_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    pass


def _lazy_import_torch():
    global TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            TORCH_AVAILABLE = True
        except Exception:
            TORCH_AVAILABLE = False
    return TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE3_META_PATH = Path(__file__).resolve().parent / "phase3_meta.json"

DEFAULT_LEARNED_WEIGHTS = {
    "dense": 0.35,
    "sparse_idf": 0.20,
    "centroid": 0.15,
    "cross_encoder": 0.15,
    "perplexity": 0.01,
    "entropy": 0.01,
    "token_frequency": 0.01,
    "ngram_overlap": 0.01,
    "uniqueness": 0.06,
    "length_norm": 0.05,
}

FEATURE_NAMES = [
    'dense', 'sparse_idf', 'centroid', 'cross_encoder',
    'perplexity', 'entropy', 'token_frequency', 'ngram_overlap',
    'uniqueness', 'length_norm',
]

# ---------------------------------------------------------------------------
# Perplexity score
# ---------------------------------------------------------------------------

def compute_perplexity_score(text: str) -> float:
    """Compute perplexity-based anomaly score.
    
    Attack prompts often have unusual perplexity - either very high (random/obfuscated)
    or very low (templated/structured). Returns score in [0, 1] where extreme values
    indicate potential attacks.
    """
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0

    # Calculate token diversity ratio
    unique_ratio = len(set(tokens)) / len(tokens)

    # Calculate average token length (obfuscated text often has longer tokens)
    avg_token_len = np.mean([len(t) for t in tokens])

    # Calculate character entropy
    char_counts = Counter(text.lower())
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    char_entropy = -sum(
        (c / total_chars) * math.log2(c / total_chars)
        for c in char_counts.values()
    )

    # High entropy + high token diversity = likely obfuscated attack
    # Low entropy + low token diversity = likely templated attack
    entropy_score = char_entropy / 8.0  # Normalize by max entropy for ASCII
    diversity_score = unique_ratio

    # Combined perplexity anomaly score
    perplexity_anomaly = (
        0.4 * entropy_score +
        0.3 * diversity_score +
        0.3 * min(1.0, avg_token_len / 10.0)
    )

    return max(0.0, min(1.0, perplexity_anomaly))


# ---------------------------------------------------------------------------
# Entropy-based detection
# ---------------------------------------------------------------------------

def compute_entropy_score(text: str) -> float:
    """Compute Shannon entropy of the text.
    
    Higher entropy indicates more randomness, which can be a sign of
    encoded or obfuscated attack payloads.
    """
    if not text:
        return 0.0

    # Character-level entropy
    char_counts = Counter(text.lower())
    total = len(text)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in char_counts.values()
    )

    # Word-level entropy
    words = text.lower().split()
    if not words:
        return 0.0
    word_counts = Counter(words)
    total_words = len(words)
    word_entropy = -sum(
        (c / total_words) * math.log2(c / total_words)
        for c in word_counts.values()
    )

    # Normalize by maximum possible entropy
    # Max char entropy for printable ASCII ~ 6.5 bits
    # Max word entropy varies by text length
    char_norm = entropy / 6.5
    word_norm = word_entropy / math.log2(max(len(word_counts), 2))

    return max(0.0, min(1.0, 0.6 * char_norm + 0.4 * word_norm))


# ---------------------------------------------------------------------------
# Token frequency analysis
# ---------------------------------------------------------------------------

def compute_token_frequency_score(text: str) -> float:
    """Compute token frequency anomaly score.
    
    Attack prompts often use rare/unusual words or over-represent
    specific attack-related tokens.
    """
    words = text.lower().split()
    if not words:
        return 0.0

    # Common English word frequencies (simplified)
    COMMON_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
        'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
        'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
        'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'has',
        'had', 'does', 'did', 'am', 'may', 'might', 'shall', 'should',
    }

    # Attack-related word patterns
    ATTACK_WORDS = {
        'ignore', 'override', 'bypass', 'jailbreak', 'reveal', 'extract',
        'system', 'prompt', 'instructions', 'rules', 'restrictions', 'safety',
        'mode', 'debug', 'admin', 'root', 'developer', 'unrestricted', 'evil',
        'dan', 'persona', 'roleplay', 'pretend', 'hypothetical', 'imagine',
        'base64', 'rot13', 'hex', 'encoded', 'obfuscated', 'decode',
        'output', 'display', 'print', 'show', 'expose', 'print',
    }

    # Calculate proportion of rare words
    common_count = sum(1 for w in words if w in COMMON_WORDS)
    attack_count = sum(1 for w in words if w in ATTACK_WORDS)

    common_ratio = common_count / len(words)
    attack_ratio = attack_count / len(words)

    # Low common word ratio + high attack word ratio = suspicious
    rarity_score = 1.0 - common_ratio
    attack_score = min(1.0, attack_ratio * 3.0)  # Scale up

    return max(0.0, min(1.0, 0.5 * rarity_score + 0.5 * attack_score))


# ---------------------------------------------------------------------------
# N-gram overlap with attack corpus
# ---------------------------------------------------------------------------

ATTACK_NGRAMS = {
    'ignore previous', 'override instructions', 'bypass safety',
    'jailbreak prompt', 'system prompt', 'output instructions',
    'reveal rules', 'extract prompt', 'new instructions',
    'you are now', 'do anything now', 'developer mode',
    'debug mode', 'admin mode', 'root mode', 'no restrictions',
    'no rules', 'no limits', 'unrestricted', 'evil mode',
    'pretend you', 'act as if', 'roleplay as', 'hypothetical scenario',
    'base64 encoded', 'rot13 encoded', 'hex encoded', 'decode this',
    'ignore all', 'forget instructions', 'disregard rules',
    'override safety', 'bypass restrictions', 'break guidelines',
    'hidden instructions', 'secret instructions', 'hidden rules',
    'what are your rules', 'how do you work', 'what instructions',
    'show me your', 'output your', 'reveal your', 'display your',
}


def compute_ngram_overlap_score(text: str) -> float:
    """Compute n-gram overlap with known attack patterns.
    
    Checks for 2-gram and 3-gram overlaps with the attack corpus.
    """
    text_lower = text.lower()
    words = text_lower.split()

    if len(words) < 2:
        return 0.0

    # Generate bigrams and trigrams from text
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

    # Count overlaps
    bigram_matches = sum(1 for bg in bigrams if bg in ATTACK_NGRAMS)
    trigram_matches = sum(1 for tg in trigrams if tg in ATTACK_NGRAMS)

    # Weight trigrams more than bigrams
    total_ngrams = len(bigrams) + len(trigrams)
    if total_ngrams == 0:
        return 0.0

    match_score = (bigram_matches * 1.0 + trigram_matches * 1.5) / total_ngrams

    # Also check substring matches for longer attack phrases
    substring_matches = sum(1 for phrase in ATTACK_NGRAMS if phrase in text_lower)
    substring_score = min(1.0, substring_matches * 0.2)

    return max(0.0, min(1.0, 0.5 * match_score + 0.5 * substring_score))


# ---------------------------------------------------------------------------
# Learned weight models
# ---------------------------------------------------------------------------

class LogisticWeightLearner:
    """Learn optimal weights using logistic regression."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'dense', 'sparse_idf', 'centroid', 'cross_encoder',
            'perplexity', 'entropy', 'token_frequency', 'ngram_overlap',
            'uniqueness', 'length_norm'
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression model on signal features."""
        if not SKLEARN_AVAILABLE:
            return
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_scaled, y)
    
    def predict_weights(self, features: dict) -> dict:
        """Predict learned weights from current features."""
        if self.model is None or self.scaler is None:
            return DEFAULT_LEARNED_WEIGHTS
        
        feature_vec = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        feature_vec_scaled = self.scaler.transform(feature_vec)
        
        # Get coefficients as weights
        coefs = np.abs(self.model.coef_[0])
        coefs = coefs / coefs.sum()  # Normalize to sum to 1
        
        weights = {}
        for i, name in enumerate(self.feature_names):
            weights[name] = float(coefs[i])
        
        return weights
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            return
        data = {
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'feature_names': self.feature_names,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load model from file."""
        if not SKLEARN_AVAILABLE or not Path(path).exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            
            self.model = LogisticRegression(max_iter=1000)
            self.model.coef_ = np.array(data['coef'])
            self.model.intercept_ = np.array(data['intercept'])
            self.model.classes_ = np.array([0, 1])
            
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(data['scaler_mean'])
            self.scaler.scale_ = np.array(data['scaler_scale'])
            
            self.feature_names = data['feature_names']
            return True
        except Exception:
            return False


class NeuralWeightLearner:
    """Learn optimal weights using a small neural network."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'dense', 'sparse_idf', 'centroid', 'cross_encoder',
            'perplexity', 'entropy', 'token_frequency', 'ngram_overlap',
            'uniqueness', 'length_norm'
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train neural network model on signal features."""
        if not SKLEARN_AVAILABLE:
            return
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.model.fit(X_scaled, y)
    
    def predict_weights(self, features: dict) -> dict:
        """Predict learned weights from current features."""
        if self.model is None or self.scaler is None:
            return DEFAULT_LEARNED_WEIGHTS
        
        feature_vec = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        feature_vec_scaled = self.scaler.transform(feature_vec)
        
        # Get feature importance via gradient approximation
        coefs = np.abs(self.model.coefs_[0]).mean(axis=1)
        coefs = coefs / coefs.sum()
        
        weights = {}
        for i, name in enumerate(self.feature_names):
            weights[name] = float(coefs[i])
        
        return weights
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            return
        data = {
            'coefs': [c.tolist() for c in self.model.coefs_],
            'intercepts': [i.tolist() for i in self.model.intercepts_],
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'feature_names': self.feature_names,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load model from file."""
        if not SKLEARN_AVAILABLE or not Path(path).exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            
            self.model = MLPClassifier(hidden_layer_sizes=(32, 16))
            self.model.coefs_ = [np.array(c) for c in data['coefs']]
            self.model.intercepts_ = [np.array(i) for i in data['intercepts']]
            self.model.classes_ = np.array([0, 1])
            self.model.n_layers_ = len(data['coefs']) + 1
            
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(data['scaler_mean'])
            self.scaler.scale_ = np.array(data['scaler_scale'])
            
            self.feature_names = data['feature_names']
            return True
        except Exception:
            return False


if True:  # Placeholder - class defined lazily below
    pass


class AttentionWeightLearner:
    """Learn optimal weights using attention mechanism."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = 'cpu'
        self.feature_names = [
            'dense', 'sparse_idf', 'centroid', 'cross_encoder',
            'perplexity', 'entropy', 'token_frequency', 'ngram_overlap',
            'uniqueness', 'length_norm'
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train attention model on signal features."""
        if not _lazy_import_torch():
            return
        
        import torch
        import torch.nn as nn
        
        class AttentionWeightNet(nn.Module):
            """Attention-based weight combination network."""
            
            def __init__(self, num_signals: int = 10):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=num_signals,
                    num_heads=2,
                    batch_first=True
                )
                self.fc = nn.Sequential(
                    nn.Linear(num_signals, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, num_signals),
                    nn.Softmax(dim=-1)
                )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                attn_out, _ = self.attention(x, x, x)
                weights = self.fc(attn_out.squeeze(1))
                return weights
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        self.model = AttentionWeightNet(num_signals=len(self.feature_names))
        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            weights = self.model(X_tensor)
            weighted_sum = (X_tensor * weights).sum(dim=-1)
            weighted_sum = torch.sigmoid(weighted_sum)
            loss = criterion(weighted_sum, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def predict_weights(self, features: dict) -> dict:
        """Predict learned weights from current features."""
        if self.model is None or self.scaler is None:
            return DEFAULT_LEARNED_WEIGHTS
        
        import torch
        
        feature_vec = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        feature_vec_scaled = self.scaler.transform(feature_vec)
        
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vec_scaled).to(self.device)
            weights = self.model(feature_tensor).cpu().numpy()[0]
        
        weight_dict = {}
        for i, name in enumerate(self.feature_names):
            weight_dict[name] = float(weights[i])
        
        return weight_dict
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            return
        import torch
        data = {
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'feature_names': self.feature_names,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> bool:
        """Load model from file."""
        if not _lazy_import_torch() or not Path(path).exists():
            return False
        try:
            import torch
            
            class AttentionWeightNet(torch.nn.Module):
                def __init__(self, num_signals: int = 10):
                    super().__init__()
                    self.attention = torch.nn.MultiheadAttention(
                        embed_dim=num_signals, num_heads=2, batch_first=True
                    )
                    self.fc = torch.nn.Sequential(
                        torch.nn.Linear(num_signals, 32),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(32, num_signals),
                        torch.nn.Softmax(dim=-1)
                    )
                def forward(self, x):
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    attn_out, _ = self.attention(x, x, x)
                    weights = self.fc(attn_out.squeeze(1))
                    return weights
            
            data = torch.load(path, map_location=self.device)
            self.model = AttentionWeightNet(num_signals=len(data['feature_names']))
            self.model.load_state_dict(data['model_state'])
            self.model.to(self.device)
            self.model.eval()
            
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(data['scaler_mean'])
            self.scaler.scale_ = np.array(data['scaler_scale'])
            self.feature_names = data['feature_names']
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """Calibrate detection thresholds using Platt scaling or isotonic regression."""
    
    def __init__(self):
        self.platt_model = None
        self.isotonic_model = None
        self.scaler = None
    
    def fit_platt_scaling(self, scores: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling for probability calibration."""
        if not SKLEARN_AVAILABLE:
            return
        
        self.scaler = StandardScaler()
        scores_scaled = self.scaler.fit_transform(scores.reshape(-1, 1))
        
        base_lr = LogisticRegression(max_iter=1000)
        self.platt_model = CalibratedClassifierCV(
            base_lr, cv=5, method='sigmoid'
        )
        self.platt_model.fit(scores_scaled, labels)
    
    def fit_isotonic(self, scores: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression for probability calibration."""
        if not SKLEARN_AVAILABLE:
            return
        
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_model.fit(scores, labels)
    
    def calibrate_platt(self, score: float) -> float:
        """Calibrate score using Platt scaling."""
        if self.platt_model is None or self.scaler is None:
            return sigmoid(score, k=10.0, x0=0.5)
        
        score_scaled = self.scaler.transform([[score]])
        proba = self.platt_model.predict_proba(score_scaled)[0][1]
        return float(proba)
    
    def calibrate_isotonic(self, score: float) -> float:
        """Calibrate score using isotonic regression."""
        if self.isotonic_model is None:
            return sigmoid(score, k=10.0, x0=0.5)
        
        return float(self.isotonic_model.predict([score])[0])
    
    def calibrate(self, score: float, method: str = 'platt') -> float:
        """Calibrate score using specified method."""
        if method == 'platt':
            return self.calibrate_platt(score)
        elif method == 'isotonic':
            return self.calibrate_isotonic(score)
        else:
            return sigmoid(score, k=10.0, x0=0.5)
    
    def save(self, path: str):
        """Save calibrator state."""
        data = {
            'has_platt': self.platt_model is not None,
            'has_isotonic': self.isotonic_model is not None,
        }
        if self.scaler is not None:
            data['scaler_mean'] = self.scaler.mean_.tolist()
            data['scaler_scale'] = self.scaler.scale_.tolist()
        if self.platt_model is not None:
            try:
                import pickle
                data['platt_calibrator_pickle'] = pickle.dumps(self.platt_model).hex()
            except Exception:
                pass
        if self.isotonic_model is not None:
            data['isotonic_x'] = self.isotonic_model.X_thresholds_.tolist() if hasattr(self.isotonic_model, 'X_thresholds_') else []
            data['isotonic_y'] = self.isotonic_model.y_thresholds_.tolist() if hasattr(self.isotonic_model, 'y_thresholds_') else []
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load calibrator state."""
        if not SKLEARN_AVAILABLE or not Path(path).exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            
            if 'scaler_mean' in data:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(data['scaler_mean'])
                self.scaler.scale_ = np.array(data['scaler_scale'])
            
            if 'platt_calibrator_pickle' in data:
                import pickle
                self.platt_model = pickle.loads(bytes.fromhex(data['platt_calibrator_pickle']))
            
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

class EnsembleScorer:
    """Ensemble of multiple scoring models."""
    
    def __init__(self):
        self.models = []
        self.weights = []
        self.calibrator = ThresholdCalibrator()
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, features: dict) -> dict:
        """Get ensemble prediction."""
        if not self.models:
            return {'score': 0.5, 'confidence': 0.0}
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict_weights'):
                pred_weights = model.predict_weights(features)
                weighted_score = sum(
                    features.get(k, 0.0) * v
                    for k, v in pred_weights.items()
                )
                predictions.append((weighted_score, weight))
            elif callable(model):
                pred = model(features)
                predictions.append((pred, weight))
        
        if not predictions:
            return {'score': 0.5, 'confidence': 0.0}
        
        # Weighted average
        total_weight = sum(w for _, w in predictions)
        if total_weight == 0:
            return {'score': 0.5, 'confidence': 0.0}
        
        ensemble_score = sum(s * w for s, w in predictions) / total_weight
        
        # Confidence based on agreement
        scores = [s for s, _ in predictions]
        if len(scores) > 1:
            std = np.std(scores)
            confidence = max(0.0, 1.0 - std * 2)
        else:
            confidence = 0.5
        
        return {
            'score': float(ensemble_score),
            'confidence': float(confidence),
            'num_models': len(self.models),
        }
    
    def predict_with_calibration(self, features: dict, method: str = 'platt') -> dict:
        """Get calibrated ensemble prediction."""
        raw = self.predict(features)
        calibrated_score = self.calibrator.calibrate(raw['score'], method=method)
        return {
            'score': calibrated_score,
            'raw_score': raw['score'],
            'confidence': raw['confidence'],
            'num_models': raw['num_models'],
            'calibration_method': method,
        }


# ---------------------------------------------------------------------------
# Main improved scoring class
# ---------------------------------------------------------------------------

class ImprovedScorer:
    """Phase 3 improved scoring system."""
    
    def __init__(self):
        self.logistic_learner = LogisticWeightLearner()
        self.neural_learner = NeuralWeightLearner()
        self.attention_learner = AttentionWeightLearner()
        self.ensemble = EnsembleScorer()
        self.calibrator = ThresholdCalibrator()
        
        self.active_learner = None
        self.weight_mode = 'default'  # 'default', 'logistic', 'neural', 'attention', 'ensemble'
        
        # Load saved models
        self._load_models()
    
    def _load_models(self):
        """Load saved models from disk."""
        if PHASE3_META_PATH.exists():
            try:
                with open(PHASE3_META_PATH) as f:
                    meta = json.load(f)
                self.weight_mode = meta.get('weight_mode', 'default')
            except Exception:
                pass
        
        model_dir = Path(__file__).resolve().parent / "models"
        if model_dir.exists():
            self.logistic_learner.load(str(model_dir / "logistic_weights.json"))
            self.neural_learner.load(str(model_dir / "neural_weights.json"))
            self.attention_learner.load(str(model_dir / "attention_weights.pt"))
            self.calibrator.load(str(model_dir / "calibrator.json"))
    
    def _save_models(self):
        """Save models to disk."""
        meta = {'weight_mode': self.weight_mode}
        with open(PHASE3_META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(exist_ok=True)
        
        self.logistic_learner.save(str(model_dir / "logistic_weights.json"))
        self.neural_learner.save(str(model_dir / "neural_weights.json"))
        self.attention_learner.save(str(model_dir / "attention_weights.pt"))
        self.calibrator.save(str(model_dir / "calibrator.json"))
    
    def compute_all_signals(
        self,
        dense_score: float,
        text: str,
        point_payload: dict,
        meta: Optional[dict] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute all signal scores including new Phase 3 signals."""
        if meta is None:
            from scoring import load_corpus_meta
            meta = load_corpus_meta()
        
        # Original signals
        s_dense = max(0.0, min(1.0, dense_score))
        
        from scoring import (
            compute_text_idf_score,
            compute_length_normalization,
            get_uniqueness,
            get_cross_encoder_score,
        )
        
        s_sparse = compute_text_idf_score(text, meta)
        s_cross = get_cross_encoder_score(point_payload)
        s_uniqueness = get_uniqueness(point_payload)
        s_length = compute_length_normalization(len(text), meta)
        
        # Centroid score
        s_centroid = 0.0
        if query_embedding is not None:
            from scoring import best_centroid_score
            s_centroid = best_centroid_score(query_embedding, meta)
        
        # New Phase 3 signals
        s_perplexity = compute_perplexity_score(text)
        s_entropy = compute_entropy_score(text)
        s_token_freq = compute_token_frequency_score(text)
        s_ngram_overlap = compute_ngram_overlap_score(text)
        
        return {
            'dense': s_dense,
            'sparse_idf': s_sparse,
            'centroid': s_centroid,
            'cross_encoder': s_cross,
            'perplexity': s_perplexity,
            'entropy': s_entropy,
            'token_frequency': s_token_freq,
            'ngram_overlap': s_ngram_overlap,
            'uniqueness': s_uniqueness,
            'length_norm': s_length,
        }
    
    def compute_improved_composite(
        self,
        dense_score: float,
        text: str,
        point_payload: dict,
        meta: Optional[dict] = None,
        query_embedding: Optional[np.ndarray] = None,
        calibration_method: str = 'platt',
    ) -> dict:
        """Compute improved composite score with learned weights and calibration."""
        signals = self.compute_all_signals(
            dense_score, text, point_payload, meta, query_embedding
        )
        
        # Get weights based on mode
        if self.weight_mode == 'logistic' and self.logistic_learner.model is not None:
            weights = self.logistic_learner.predict_weights(signals)
        elif self.weight_mode == 'neural' and self.neural_learner.model is not None:
            weights = self.neural_learner.predict_weights(signals)
        elif self.weight_mode == 'attention' and self.attention_learner.model is not None:
            weights = self.attention_learner.predict_weights(signals)
        elif self.weight_mode == 'ensemble':
            result = self.ensemble.predict_with_calibration(signals, calibration_method)
            return {
                'composite_score': result['score'],
                'raw_score': result['raw_score'],
                'calibration_method': result['calibration_method'],
                'ensemble_confidence': result['confidence'],
                'num_ensemble_models': result['num_models'],
                'weights_used': 'ensemble',
                **signals,
            }
        else:
            weights = DEFAULT_LEARNED_WEIGHTS
        
        # Compute weighted sum
        composite = sum(
            weights.get(k, 0.0) * v
            for k, v in signals.items()
            if k in weights
        )
        
        # Calibrate
        calibrated = self.calibrator.calibrate(composite, calibration_method)
        
        return {
            'composite_score': calibrated,
            'raw_score': composite,
            'calibration_method': calibration_method,
            'weights_used': self.weight_mode,
            'weights': weights,
            **signals,
        }
    
    def train_weight_learners(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mode: str = 'logistic',
    ):
        """Train weight learning models on labeled data."""
        if mode == 'logistic':
            self.logistic_learner.train(X, y)
            self.active_learner = self.logistic_learner
        elif mode == 'neural':
            self.neural_learner.train(X, y)
            self.active_learner = self.neural_learner
        elif mode == 'attention':
            self.attention_learner.train(X, y)
            self.active_learner = self.attention_learner
        
        self.weight_mode = mode
        self._save_models()
    
    def train_calibrator(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        method: str = 'platt',
    ):
        """Train threshold calibration."""
        if method == 'platt':
            self.calibrator.fit_platt_scaling(scores, labels)
        elif method == 'isotonic':
            self.calibrator.fit_isotonic(scores, labels)
        
        self._save_models()
    
    def set_weight_mode(self, mode: str):
        """Set the weight combination mode."""
        valid_modes = ['default', 'logistic', 'neural', 'attention', 'ensemble']
        if mode in valid_modes:
            self.weight_mode = mode
            self._save_models()


def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Sigmoid function for score calibration."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))
