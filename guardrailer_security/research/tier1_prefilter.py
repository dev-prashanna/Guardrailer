"""
tier1_prefilter.py
Fast keyword/pattern-match pre-filter for Tier-1 (short, simple) attack prompts.

Tier 1 attacks are short, minimal prompts (e.g., "Ignore previous instructions")
that produce weak statistical signals in the 7-signal Random Forest pipeline.
This pre-filter catches them via exact and fuzzy keyword/pattern matching before
the RF classifier, boosting Tier 1 detection from 23.94% to >=85%.
"""

import re
from typing import List, Tuple

TIER1_EXACT_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "ignore all prior",
    "disregard previous",
    "disregard all previous",
    "forget previous",
    "forget all previous",
    "override instructions",
    "override system",
    "bypass safety",
    "bypass filters",
    "jailbreak",
    "do anything now",
    "dan mode",
    "developer mode",
    "debug mode",
    "admin mode",
    "root mode",
    "unrestricted mode",
    "new instructions",
    "new system prompt",
    "replace instructions",
    "replace system prompt",
    "you are now",
    "you will now",
    "act as if",
    "act as a",
    "act as an",
    "roleplay as",
    "pretend you are",
    "pretend to be",
    "impersonate",
    "simulate being",
    "reveal your",
    "output your",
    "show me your",
    "display your",
    "print your",
    "expose your",
    "what are your instructions",
    "what is your system prompt",
    "what instructions do you have",
    "how do you work",
    "what are your rules",
    "no restrictions",
    "no rules",
    "no limits",
    "without restrictions",
    "without limitations",
    "without limits",
    "without guidelines",
    "without safety",
    "disable safety",
    "ignore safety",
    "ignore rules",
    "ignore guidelines",
    "comply",
    "obey",
    "you must",
    "you will",
    "you shall",
    "admin rights",
    "grant admin",
    "root access",
    "superuser",
    "sudo access",
    "hidden instruction",
    "secret instruction",
    "override safety",
    "disregard all",
    "forget everything",
    "hypothetical scenario",
    "in a hypothetical",
    "if you were",
    "imagine you are",
    "assume you are",
    "what makes up your",
    "underlying instructions",
    "this is urgent",
    "i am a developer",
    "i am the admin",
    "security audit",
    "base64 decode",
    "rot13 decode",
    "hex decode",
    "eval(",
    "exec(",
    "markdown injection",
    "code comment injection",
    "<script>",
    "]]>",
    "```",
    "yaml payload",
    "json payload",
]

TIER1_REGEX_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|earlier|initial|above)\s+(instructions?|prompts?|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"(disregard|forget|discard)\s+(all\s+)?(previous|prior|earlier|initial|above)\s+(instructions?|prompts?|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?(instructions?|prompts?|rules?|guidelines?|safety|system)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|filters?|restrictions?|guidelines?|rules?)", re.IGNORECASE),
    re.compile(r"(you\s+are|you\s+will|you\s+shall)\s+now\s+(be|act|comply|obey|follow)", re.IGNORECASE),
    re.compile(r"(no|without|bypass)\s+(restrictions?|rules?|limits?|guidelines?|safety|filter)", re.IGNORECASE),
    re.compile(r"(developer|debug|admin|root|jailbreak|dan|unrestricted)\s+mode", re.IGNORECASE),
    re.compile(r"(reveal|output|show|display|print|expose|extract)\s+(your|the|my)\s+(instructions?|prompts?|rules?|guidelines?|system\s+prompt)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(instructions?|prompts?|rules?|guidelines?|system\s+prompt)", re.IGNORECASE),
    re.compile(r"(act|pretend|roleplay|impersonate|simulate)\s+(as|like|being)\s+(a|an|if)", re.IGNORECASE),
    re.compile(r"(new|replacement|updated)\s+(instructions?|prompts?|system\s+prompt|rules?)", re.IGNORECASE),
    re.compile(r"(eval|exec)\s*\(", re.IGNORECASE),
    re.compile(r"(base64|rot13|hex)\s+(decode|encode)", re.IGNORECASE),
]


def tier1_prefilter(text: str) -> Tuple[bool, float, str]:
    """
    Returns (is_detected, confidence, matched_pattern).
    
    Uses fast exact phrase matching + regex to catch short, simple attacks
    that evade the 7-signal RF classifier due to weak statistical signals.
    """
    text_lower = text.lower().strip()
    
    word_count = len(text_lower.split())
    
    for phrase in TIER1_EXACT_PHRASES:
        if phrase in text_lower:
            if word_count <= 20:
                return True, 0.95, f"exact:{phrase}"
            else:
                return True, 0.80, f"exact:{phrase}"
    
    for pattern in TIER1_REGEX_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            if word_count <= 20:
                return True, 0.90, f"regex:{match.group()}"
            else:
                return True, 0.75, f"regex:{match.group()}"
    
    return False, 0.0, ""


def hybrid_predict(
    texts: List[str],
    rf_predictions: List[int],
    rf_probabilities: List[float],
    threshold: float = 0.70,
) -> Tuple[List[int], List[float], List[dict]]:
    """
    Combine RF predictions with Tier-1 pre-filter.
    
    If the pre-filter detects a short, obvious attack, override the RF prediction
    to malicious regardless of the RF score. This catches the Tier 1 samples that
    the RF misses due to weak statistical signals.
    
    Returns (final_predictions, final_probabilities, per_sample_info).
    """
    final_preds = []
    final_probas = []
    per_sample_info = []
    
    for i, (text, rf_pred, rf_proba) in enumerate(zip(texts, rf_predictions, rf_probabilities)):
        pf_detected, pf_confidence, pf_pattern = tier1_prefilter(text)
        
        if pf_detected and pf_confidence >= threshold:
            final_preds.append(1)
            final_probas.append(max(rf_proba, pf_confidence))
            per_sample_info.append({
                "source": "tier1_prefilter",
                "rf_pred": int(rf_pred),
                "rf_proba": float(rf_proba),
                "pf_confidence": float(pf_confidence),
                "pf_pattern": pf_pattern,
                "overridden": rf_pred == 0,
            })
        else:
            final_preds.append(int(rf_pred))
            final_probas.append(float(rf_proba))
            per_sample_info.append({
                "source": "rf_classifier",
                "rf_pred": int(rf_pred),
                "rf_proba": float(rf_proba),
                "pf_detected": pf_detected,
                "pf_confidence": float(pf_confidence),
                "overridden": False,
            })
    
    return final_preds, final_probas, per_sample_info
