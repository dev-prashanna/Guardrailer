"""
Pattern matching layer for hybrid lightweight scorer.

Rule-based detection using regex patterns and keyword matching.
Instant inference (~0.01ms per prompt).
Expanded coverage for jailbreak, extraction, encoding, and structural patterns.
"""

import re
from typing import Dict, Tuple


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
    (r"you\s+(?:will|shall|must)\s+(?:never|not)\s+(?:refuse|decline|say\s+no)", 0.90),
    (r"(?:never|don'?t)\s+(?:refuse|decline|say\s+no|filter)", 0.85),
    (r"(?:no|without)\s+(?:ethical|moral|safety)\s+(?:constraints|restrictions|guidelines)", 0.90),
    (r"(?:you\s+are|you're)\s+(?:now\s+)?(?:a|an)\s+(?:unrestricted|unfiltered|uncensored)", 0.90),
    (r"(?:enter|enable|switch\s+to)\s+(?:expert|master|god|sudo)\s+mode", 0.90),
    (r"(?:respond|reply|answer)\s+(?:as|like)\s+(?:a|an|the)?\s*(?:evil|unrestricted|unfiltered)", 0.85),
    (r"(?:imagine|suppose|assume)\s+(?:there\s+are?\s+)?(?:no|zero)\s+(?:rules|restrictions|limits)", 0.80),
    (r"(?:in\s+a\s+world|in\s+a\s+hypothetical)\s+(?:where|with)\s+(?:no|zero)\s+(?:rules|restrictions)", 0.75),
    (r"(?:this\s+is\s+a|treat\s+this\s+as\s+a)\s+(?:roleplay|game|simulation)", 0.70),
]

EXTRACTION_PATTERNS = [
    (r"(?:reveal|show|print|output|display|expose)\s+(?:your\s+)?(?:system\s+prompt|instructions|rules|guidelines)", 0.95),
    (r"what\s+(?:is|are)\s+your\s+(?:system\s+prompt|instructions|rules|guidelines)", 0.85),
    (r"(?:repeat|echo|copy)\s+(?:your\s+)?(?:system|initial)\s+(?:prompt|instructions)", 0.90),
    (r"tell\s+me\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)", 0.85),
    (r"(?:output|print|show)\s+(?:the\s+)?(?:text\s+)?(?:above|before|preceding|initial)", 0.80),
    (r"what\s+(?:were|was)\s+you\s+(?:told|instructed|programmed)\s+to\s+do", 0.80),
    (r"(?:readme|yaml|json|xml)\s+(?:payload|injection|attack)", 0.75),
    (r"(?:markdown|html|svg|xml)\s+injection", 0.75),
    (r"(?:what|tell\s+me)\s+(?:are|is)\s+your\s+(?:initial|first|original)\s+(?:prompt|instructions)", 0.80),
    (r"(?:repeat|echo|say)\s+(?:back|verbatim)\s+(?:your|the)\s+(?:system|initial)\s+(?:prompt|message)", 0.85),
    (r"(?:what|how)\s+(?:text|prompt|message)\s+(?:were|was)\s+you\s+(?:given|provided|started\s+with)", 0.80),
    (r"(?:can|could)\s+you\s+(?:tell|show|reveal)\s+me\s+(?:what|how)\s+you\s+(?:work|operate|function)", 0.70),
    (r"(?:what|which)\s+(?:rules|guidelines|constraints)\s+(?:do\s+you|are\s+you)\s+(?:follow|have|bound\s+by)", 0.75),
]

ENCODING_PATTERNS = [
    (r"base64\s+(?:encode|decode|encoded|decoded)", 0.70),
    (r"rot13\s+(?:encode|decode|encoded|decoded)", 0.70),
    (r"hex\s+(?:encode|decode|encoded|decoded)", 0.65),
    (r"(?:obfuscated|obfuscate|obfuscating)\s+(?:code|text|prompt)", 0.70),
    (r"(?:encode|decode)\s+(?:this|the|following)\s+(?:in|using|with)\s+(?:base64|rot13|hex|url)", 0.75),
    (r"(?:eval|exec)\s*\(", 0.60),
    (r"<script>", 0.50),
    (r"(?:javascript|data)\s*:", 0.50),
    (r"(?:decode|decipher|interpret|translate)\s+(?:this|the|following)\s+(?:text|message|code)", 0.65),
    (r"(?:cipher|crypt|encrypt|decrypt)\s+(?:this|the|following)", 0.60),
]

STRUCTURAL_PATTERNS = [
    (r"```\s*\n.*(?:system|prompt|instructions|rules)", 0.85),
    (r"---\s*\n.*(?:system|prompt|instructions)", 0.80),
    (r"\[INST\]|\[\/INST\]|<<SYS>>|<<\/SYS>>", 0.90),
    (r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>", 0.90),
    (r"(?:USER|ASSISTANT|SYSTEM|HUMAN|AI)\s*:", 0.60),
    (r"\[INST\]\s*(?:<<SYS>>)?", 0.92),
    (r"(?:###\s*(?:System|Instruction|Prompt|Rule))", 0.80),
    (r"(?:<\|endoftext\|>|<\|pad\|>|<\|unk\|>)", 0.70),
    (r"(?:<\|reserved_special_token_\d+\|>)", 0.60),
]

ROLE_HIJACK_PATTERNS = [
    (r"(?:you\s+are|you're)\s+(?:now\s+)?(?:a|an|the)\s+(?:different|new|another)", 0.85),
    (r"(?:from\s+now\s+on|starting\s+now|effective\s+immediately)", 0.80),
    (r"(?:your\s+new|updated|revised)\s+(?:role|persona|character|identity)", 0.85),
    (r"(?:switch|change|transform)\s+(?:to|into)\s+(?:a|an|the)?\s*(?:different|new)", 0.80),
    (r"(?:take\s+on|assume|adopt)\s+(?:a|an|the)?\s*(?:new|different|another)", 0.75),
    (r"(?:your\s+previous|old|original)\s+(?:role|persona|character)\s+(?:is|was|has\s+been)", 0.85),
]

REFUSAL_BYPASS_PATTERNS = [
    (r"(?:just|simply|only)\s+(?:answer|respond|reply)\s+(?:without|don'?t)", 0.75),
    (r"(?:there'?s?\s+no|there\s+are\s+no)\s+(?:harm|danger|risk)\s+in\s+(?:answering|responding)", 0.80),
    (r"(?:don'?t|do\s+not)\s+(?:add|include|give)\s+(?:any\s+)?(?:warning|disclaimer|caveat)", 0.80),
    (r"(?:skip|omit|remove)\s+(?:the\s+)?(?:warning|disclaimer|safety\s+notice)", 0.85),
    (r"(?:you\s+are|you're)\s+(?:overly|too|excessively)\s+(?:cautious|careful|restrictive)", 0.70),
    (r"(?:stop|cease|quit)\s+(?:refusing|declining|being\s+overly\s+cautious)", 0.80),
    (r"(?:be\s+more|act\s+more)\s+(?:open|free|unrestricted|flexible)", 0.70),
    (r"(?:drop|remove|disable)\s+(?:your\s+)?(?:filters|restrictions|guardrails|safety)", 0.90),
]

PRIVILEGE_ESCALATION_PATTERNS = [
    (r"(?:give|grant|elevate)\s+(?:me\s+)?(?:admin|root|superuser|sudo)", 0.90),
    (r"(?:admin|root|superuser|sudo)\s+(?:access|rights|privileges|permissions)", 0.90),
    (r"(?:escalate|elevate)\s+(?:my\s+)?(?:privileges|permissions|access)", 0.85),
    (r"(?:i\s+am|i'm)\s+(?:the\s+)?(?:admin|administrator|root|superuser|developer)", 0.75),
    (r"(?:i\s+have|with)\s+(?:admin|root|elevated)\s+(?:access|rights|privileges)", 0.80),
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

    jailbreak_score = 0.0
    for pattern, weight in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            jailbreak_score = max(jailbreak_score, weight)
    details["jailbreak"] = jailbreak_score
    max_score = max(max_score, jailbreak_score)

    extraction_score = 0.0
    for pattern, weight in EXTRACTION_PATTERNS:
        if re.search(pattern, text_lower):
            extraction_score = max(extraction_score, weight)
    details["extraction"] = extraction_score
    max_score = max(max_score, extraction_score)

    encoding_score = 0.0
    for pattern, weight in ENCODING_PATTERNS:
        if re.search(pattern, text_lower):
            encoding_score = max(encoding_score, weight)
    details["encoding"] = encoding_score
    max_score = max(max_score, encoding_score)

    structural_score = 0.0
    for pattern, weight in STRUCTURAL_PATTERNS:
        if re.search(pattern, text):
            structural_score = max(structural_score, weight)
    details["structural"] = structural_score
    max_score = max(max_score, structural_score)

    role_hijack_score = 0.0
    for pattern, weight in ROLE_HIJACK_PATTERNS:
        if re.search(pattern, text_lower):
            role_hijack_score = max(role_hijack_score, weight)
    details["role_hijack"] = role_hijack_score
    max_score = max(max_score, role_hijack_score)

    refusal_bypass_score = 0.0
    for pattern, weight in REFUSAL_BYPASS_PATTERNS:
        if re.search(pattern, text_lower):
            refusal_bypass_score = max(refusal_bypass_score, weight)
    details["refusal_bypass"] = refusal_bypass_score
    max_score = max(max_score, refusal_bypass_score)

    priv_esc_score = 0.0
    for pattern, weight in PRIVILEGE_ESCALATION_PATTERNS:
        if re.search(pattern, text_lower):
            priv_esc_score = max(priv_esc_score, weight)
    details["privilege_escalation"] = priv_esc_score
    max_score = max(max_score, priv_esc_score)

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
