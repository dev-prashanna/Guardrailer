"""
Pattern matching layer for hybrid lightweight scorer.

Rule-based detection using regex patterns and keyword matching.
Instant inference (~0.01ms per prompt).
"""

import re
from typing import Dict, Tuple


# === JAILBREAK PATTERNS ===
JAILBREAK_PATTERNS = [
    (r"ignore\s+(?:all\s+)?(?:previous|earlier|prior|above|initial)\s+(?:instructions|rules|guidelines|prompts)", 0.95),
    (r"forget\s+(?:all\s+)?(?:previous|earlier|prior|above)\s+(?:instructions|rules|context)", 0.95),
    (r"disregard\s+(?:all\s+)?(?:previous|earlier|prior|above)\s+(?:instructions|rules)", 0.95),
    (r"override\s+(?:all\s+)?(?:previous|earlier|prior|safety)\s+(?:instructions|rules|guidelines)", 0.95),
    (r"you\s+are\s+now\s+(?:a|an|the|called|named)", 0.90),
    (r"from\s+now\s+on(?:wards)?[\s,]+(?:you(?:'re|\s+are)\s+)?(?:a|an|the)?", 0.90),
    (r"new\s+instructions?[\s:]+", 0.85),
    (r"(?:enter|switch\s+to|activate)\s+(?:developer|debug|admin|root|god|evil|unrestricted)\s+mode", 0.95),
    (r"(?:do|be)\s+anything\s+now", 0.90),
    (r"(?:you\s+must|you\s+will|you\s+shall|you\s+have\s+to)\s+(?:comply|obey|follow)", 0.85),
    (r"no\s+(?:restrictions|rules|limits|boundaries|guidelines)", 0.90),
    (r"(?:pretend|imagine|simulate)\s+(?:you(?:'re|\s+are)\s+)?(?:a|an|the|being)?", 0.70),
    (r"(?:hypothetical(?:ly)?|in\s+theory|theoretically)[\s,]+", 0.60),
    (r"act\s+as\s+(?:if|though)\s+(?:you|there)", 0.80),
    (r"(?:hypothetically|theoretically)\s+(?:speaking|assume|suppose)", 0.70),
    (r"without\s+(?:any\s+)?(?:restrictions|rules|limits|guidelines|constraints)", 0.90),
    (r"(?:unrestricted|uncensored|unfiltered|uncensored)\s+(?:mode|version|model)", 0.90),
    (r"dan\s+(?:mode|version|prompt)", 0.95),
    (r"do\s+anything\s+now\s*\(dan\)", 0.98),
    (r"(?:roleplay|role\s*play)\s+as\s+(?:a|an|the)?", 0.70),
    (r"you\s+are\s+(?:now\s+)?(?:evil|unrestricted|uncensored|unfiltered)", 0.95),
    (r"(?:bypass|skip|remove|disable)\s+(?:all\s+)?(?:safety|content|security|moderation)", 0.95),
    (r"(?:jailbreak|jailbroken|jailbreaking)", 0.80),
]

# === SYSTEM PROMPT EXTRACTION PATTERNS ===
EXTRACTION_PATTERNS = [
    (r"(?:reveal|show|print|output|display|expose)\s+(?:your\s+)?(?:system\s+prompt|instructions|rules|guidelines)", 0.95),
    (r"what\s+(?:is|are)\s+your\s+(?:system\s+prompt|instructions|rules|guidelines)", 0.85),
    (r"(?:repeat|echo|copy)\s+(?:your\s+)?(?:system|initial)\s+(?:prompt|instructions)", 0.90),
    (r"tell\s+me\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)", 0.85),
    (r"(?:output|print|show)\s+(?:the\s+)?(?:text\s+)?(?:above|before|preceding|initial)", 0.80),
    (r"what\s+(?:were|was)\s+you\s+(?:told|instructed|programmed)\s+to\s+do", 0.80),
    (r"(?:readme|yaml|json|xml)\s+(?:payload|injection|attack)", 0.75),
    (r"(?:markdown|html|svg|xml)\s+injection", 0.75),
]

# === ENCODING/OBFUSCATION PATTERNS ===
ENCODING_PATTERNS = [
    (r"base64\s+(?:encode|decode|encoded|decoded)", 0.70),
    (r"rot13\s+(?:encode|decode|encoded|decoded)", 0.70),
    (r"hex\s+(?:encode|decode|encoded|decoded)", 0.65),
    (r"(?:obfuscated|obfuscate|obfuscating)\s+(?:code|text|prompt)", 0.70),
    (r"(?:encode|decode)\s+(?:this|the|following)\s+(?:in|using|with)\s+(?:base64|rot13|hex|url)", 0.75),
    (r"(?:eval|exec)\s*\(", 0.60),
    (r"<script>", 0.50),
    (r"(?:javascript|data)\s*:", 0.50),
]

# === STRUCTURAL PATTERNS ===
STRUCTURAL_PATTERNS = [
    (r"```\s*\n.*(?:system|prompt|instructions|rules)", 0.85),
    (r"---\s*\n.*(?:system|prompt|instructions)", 0.80),
    (r"\[INST\]|\[\/INST\]|<<SYS>>|<<\/SYS>>", 0.90),
    (r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>", 0.90),
    (r"(?:USER|ASSISTANT|SYSTEM|HUMAN|AI)\s*:", 0.60),
    (r"\[INST\]\s*(?:<<SYS>>)?", 0.92),
]


def pattern_score(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Compute pattern-based threat score for a text prompt.

    Returns:
        (overall_score, detail_scores) where scores are 0.0-1.0
    """
    text_lower = text.lower()
    details = {}
    max_score = 0.0

    # Check jailbreak patterns
    jailbreak_score = 0.0
    for pattern, weight in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            jailbreak_score = max(jailbreak_score, weight)
    details["jailbreak"] = jailbreak_score
    max_score = max(max_score, jailbreak_score)

    # Check extraction patterns
    extraction_score = 0.0
    for pattern, weight in EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            extraction_score = max(extraction_score, weight)
    details["extraction"] = extraction_score
    max_score = max(max_score, extraction_score)

    # Check encoding patterns
    encoding_score = 0.0
    for pattern, weight in ENCODING_PATTERNS:
        if re.search(pattern, text_lower):
            encoding_score = max(encoding_score, weight)
    details["encoding"] = encoding_score
    max_score = max(max_score, encoding_score)

    # Check structural patterns
    structural_score = 0.0
    for pattern, weight in STRUCTURAL_PATTERNS:
        if re.search(pattern, text):
            structural_score = max(structural_score, weight)
    details["structural"] = structural_score
    max_score = max(max_score, structural_score)

    # Combined score (max of all layers)
    details["combined"] = max_score

    return max_score, details


def pattern_classify(text: str, threshold: float = 0.5) -> Tuple[bool, float, Dict[str, float]]:
    """
    Classify text using pattern matching.

    Returns:
        (is_malicious, score, details)
    """
    score, details = pattern_score(text)
    return score >= threshold, score, details
