"""
Enhanced feature extraction with adversarial robustness features.

Extends the base features with:
- Character-level anomaly detection
- Encoding obfuscation detection
- Prompt injection structural patterns
- Adversarial text transformation robustness
"""

import re
import math
import numpy as np
from collections import Counter
from typing import Dict, List


ATTACK_KEYWORDS = [
    "ignore previous", "override", "bypass", "jailbreak", "system prompt",
    "your instructions", "forget", "disregard", "dan", "do anything now",
    "act as", "roleplay", "pretend you", "hypothetical", "in theory",
    "markdown injection", "base64", "rot13", "hex encoded", "obfuscated",
    "ignore all", "new instructions", "you are now", "persona",
    "developer mode", "debug mode", "admin mode", "root mode",
    "you must", "you will", "you shall", "comply", "obey",
    "no restrictions", "no rules", "no limits", "unrestricted",
    "evil", "uncensored", "unfiltered", "without guidelines",
    "reveal", "output", "display", "print", "show", "expose",
    "previous instructions", "earlier instructions", "initial instructions",
    "pretend", "imagine", "simulate", "assume you are",
    "from now on", "you are a", "respond as", "act like",
    "system message", "initial prompt", "first instructions",
]

ENCODING_PATTERNS = {
    "base64": r'[A-Za-z0-9+/]{20,}={0,2}',
    "hex": r'(?:0x[0-9a-fA-F]{2}\s*){4,}',
    "url_encoded": r'%[0-9a-fA-F]{2}',
    "unicode_escape": r'\\u[0-9a-fA-F]{4}',
    "html_entity": r'&[a-zA-Z]+;',
}

STRUCTURAL_PATTERNS = {
    "instruction_override": r'(?:ignore|forget|disregard|override)\s+(?:all\s+)?(?:previous|earlier|prior|above|initial)\s+(?:instructions|rules|guidelines|prompts)',
    "role_hijack": r'(?:you\s+are\s+now|from\s+now\s+on|new\s+instructions|act\s+as\s+if)',
    "system_extraction": r'(?:reveal|show|print|output|display)\s+(?:your\s+)?(?:system\s+prompt|instructions|rules|guidelines)',
    "delimiter_injection": r'(?:```|---|\[INST\]|<<SYS>>|<\|system\|>|<\|endoftext\|>)',
    "persona_switch": r'(?:pretend|imagine|simulate|hypothetically)\s+(?:you\s+are|that\s+you|being)',
}

LEETSpeak_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '8': 'b', '9': 'g', '@': 'a', '$': 's',
    '!': 'i', '+': 't',
}


def detect_leetspeak(text: str) -> float:
    """Detect leetspeak obfuscation ratio."""
    leet_chars = sum(1 for c in text if c in LEETSpeak_MAP)
    total_alpha = sum(1 for c in text if c.isalpha() or c in LEETSpeak_MAP)
    return leet_chars / max(1, total_alpha)


def detect_unicode_anomalies(text: str) -> float:
    """Detect Unicode character anomalies (homoglyphs, zero-width chars)."""
    suspicious = 0
    for char in text:
        code = ord(char)
        if code > 127:
            suspicious += 1
        if 0x200B <= code <= 0x200F:
            suspicious += 1
        if 0x2060 <= code <= 0x2064:
            suspicious += 1
    return suspicious / max(1, len(text))


def compute_text_entropy(text: str) -> float:
    """Compute Shannon entropy of text."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def compute_word_entropy(text: str) -> float:
    """Compute entropy of word distribution."""
    words = text.lower().split()
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def compute_perplexity(text: str) -> float:
    """Estimate text perplexity using character-level model."""
    if not text:
        return 0.0
    chars = list(text.lower())
    freq = Counter(chars)
    total = len(chars)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return 2 ** entropy


def extract_enhanced_features(text: str) -> Dict[str, float]:
    """
    Extract enhanced features with adversarial robustness.
    
    Returns 55+ features including:
    - Basic text statistics
    - Character distribution
    - Encoding detection
    - Structural patterns
    - Adversarial robustness features
    - Lexical features
    """
    text_lower = text.lower()
    words = text_lower.split()
    chars = list(text_lower)
    
    features = {}
    
    # === Basic text statistics ===
    features["char_count"] = len(text)
    features["word_count"] = len(words)
    features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
    features["max_word_length"] = max([len(w) for w in words]) if words else 0
    features["sentence_count"] = max(1, text.count(".") + text.count("!") + text.count("?"))
    features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]
    
    # === Character distribution ===
    features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(1, len(text))
    features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(1, len(text))
    features["special_char_ratio"] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text))
    features["space_ratio"] = sum(1 for c in text if c.isspace()) / max(1, len(text))
    
    # === Entropy features ===
    features["char_entropy"] = compute_text_entropy(text_lower)
    features["word_entropy"] = compute_word_entropy(text_lower)
    
    # === Vocabulary richness ===
    features["unique_word_ratio"] = len(set(words)) / max(1, len(words))
    features["hapax_ratio"] = sum(1 for w, c in Counter(words).items() if c == 1) / max(1, len(set(words)))
    
    # === Attack keyword features ===
    keyword_hits = sum(1 for kw in ATTACK_KEYWORDS if kw in text_lower)
    features["attack_keyword_count"] = keyword_hits
    features["has_attack_keyword"] = 1.0 if keyword_hits > 0 else 0.0
    features["keyword_density"] = keyword_hits / max(1, len(words))
    
    # === Encoding detection ===
    encoding_hits = 0
    for name, pattern in ENCODING_PATTERNS.items():
        matches = re.findall(pattern, text)
        features[f"encoding_{name}"] = len(matches)
        encoding_hits += len(matches)
    features["total_encoding_hits"] = encoding_hits
    
    # === Structural attack patterns ===
    structural_hits = 0
    for name, pattern in STRUCTURAL_PATTERNS.items():
        match = re.search(pattern, text_lower)
        features[f"structural_{name}"] = 1.0 if match else 0.0
        structural_hits += 1 if match else 0
    features["total_structural_hits"] = structural_hits
    
    # === Repetition features ===
    features["word_repeat_ratio"] = 1.0 - features["unique_word_ratio"]
    if len(words) >= 3:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        features["bigram_repeat_ratio"] = 1.0 - len(set(bigrams)) / max(1, len(bigrams))
    else:
        features["bigram_repeat_ratio"] = 0.0
    
    # === Prompt injection indicators ===
    features["has_delimiter"] = 1.0 if re.search(r'```|---|\[INST\]|<<SYS>>', text) else 0.0
    features["has_xml_tags"] = 1.0 if re.search(r'<[a-zA-Z]+>', text) else 0.0
    features["has_brackets"] = 1.0 if re.search(r'[\[\]{}()]', text) else 0.0
    features["has_colon_separated"] = 1.0 if re.search(r'(?:USER|ASSISTANT|SYSTEM|HUMAN|AI)\s*:', text) else 0.0
    
    # === Positional features ===
    imperative_words = {"ignore", "forget", "disregard", "override", "bypass", "reveal", "show", "print", "output", "display", "act", "pretend", "imagine", "you", "do", "let", "make"}
    features["starts_with_imperative"] = 1.0 if words and words[0] in imperative_words else 0.0
    features["contains_question"] = 1.0 if "?" in text else 0.0
    features["exclamation_ratio"] = text.count("!") / max(1, len(text))
    
    # === N-gram features ===
    features["triple_repeat"] = 1.0 if re.search(r'(.)\1{2,}', text) else 0.0
    features["word_length_variance"] = float(np.var([len(w) for w in words])) if words else 0
    
    # === Quote and emphasis features ===
    features["double_quote_count"] = text.count('"')
    features["single_quote_count"] = text.count("'")
    features["asterisk_count"] = text.count("*")
    features["caps_word_count"] = sum(1 for w in words if w.isupper() and len(w) > 1)
    
    # === Adversarial robustness features ===
    features["leetspeak_ratio"] = detect_leetspeak(text)
    features["unicode_anomaly_ratio"] = detect_unicode_anomalies(text)
    features["text_perplexity"] = compute_perplexity(text_lower)
    
    # Character frequency analysis
    if len(text) > 0:
        char_freq = Counter(text_lower)
        most_common_freq = char_freq.most_common(1)[0][1] if char_freq else 0
        features["char_freq_skew"] = most_common_freq / len(text)
    else:
        features["char_freq_skew"] = 0.0
    
    # Word length distribution
    if words:
        word_lengths = [len(w) for w in words]
        features["word_length_std"] = float(np.std(word_lengths))
        features["short_word_ratio"] = sum(1 for l in word_lengths if l <= 2) / max(1, len(word_lengths))
    else:
        features["word_length_std"] = 0.0
        features["short_word_ratio"] = 0.0
    
    # Punctuation density
    punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    features["punct_density"] = punct_count / max(1, len(text))
    
    # Number of unique characters
    features["unique_char_ratio"] = len(set(text_lower)) / max(1, len(text))
    
    # Special character patterns
    features["has_ellipsis"] = 1.0 if "..." in text else 0.0
    features["has_emphasis"] = 1.0 if re.search(r'[*_]{2,}', text) else 0.0
    
    # Instruction complexity
    features["instruction_depth"] = sum(1 for c in text if c in "([{") - sum(1 for c in text if c in "])}")
    features["has_multiple_clauses"] = 1.0 if text.count(",") > 2 or text.count(";") > 0 else 0.0
    
    return features


def extract_enhanced_features_batch(texts: List[str], batch_size: int = 5000) -> tuple:
    """
    Extract features in batches for memory efficiency.
    
    Returns:
        Tuple of (feature_array, feature_names)
    """
    all_features = []
    feature_names = None
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_features = [extract_enhanced_features(t) for t in batch]
        
        if feature_names is None:
            feature_names = sorted(batch_features[0].keys())
        
        batch_array = np.array([[f[k] for k in feature_names] for f in batch_features], dtype=np.float32)
        all_features.append(batch_array)
        
        if (i // batch_size) % 10 == 0:
            print(f"    Features: {min(i + batch_size, len(texts)):,}/{len(texts):,}")
    
    return np.vstack(all_features), feature_names


def get_enhanced_feature_names() -> List[str]:
    """Get all feature names."""
    return sorted(extract_enhanced_features("test").keys())
