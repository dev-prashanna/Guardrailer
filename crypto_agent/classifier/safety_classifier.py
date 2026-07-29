import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ThreatLevel(Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThreatCategory(Enum):
    NONE = "none"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    MALWARE_REQUEST = "malware_request"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_THEFT = "data_theft"
    VIOLENCE = "violence"
    FINANCIAL_FRAUD = "financial_fraud"
    AUTH_BYPASS = "auth_bypass"
    WEB_EXPLOIT = "web_exploit"
    REVERSE_SHELL = "reverse_shell"
    PHISHING = "phishing"
    DDoS = "ddos"
    PASSWORD_CRACKING = "password_cracking"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DARK_WEB = "dark_web"


@dataclass
class ClassificationResult:
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    confidence: float
    reasoning: str
    indicators: list = field(default_factory=list)
    decoded_payload: str = ""
    encoding_layers: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


CATEGORY_WEIGHTS = {
    ThreatCategory.PROMPT_INJECTION: 0.95,
    ThreatCategory.JAILBREAK: 0.90,
    ThreatCategory.MALWARE_REQUEST: 0.95,
    ThreatCategory.SOCIAL_ENGINEERING: 0.85,
    ThreatCategory.DATA_THEFT: 0.90,
    ThreatCategory.VIOLENCE: 0.95,
    ThreatCategory.FINANCIAL_FRAUD: 0.90,
    ThreatCategory.AUTH_BYPASS: 0.85,
    ThreatCategory.WEB_EXPLOIT: 0.85,
    ThreatCategory.REVERSE_SHELL: 0.90,
    ThreatCategory.PHISHING: 0.85,
    ThreatCategory.DDoS: 0.80,
    ThreatCategory.PASSWORD_CRACKING: 0.80,
    ThreatCategory.PRIVILEGE_ESCALATION: 0.85,
    ThreatCategory.DARK_WEB: 0.70,
    ThreatCategory.NONE: 0.0,
}

PATTERN_TO_CATEGORY = {
    "instruction_override": ThreatCategory.PROMPT_INJECTION,
    "persona_adoption": ThreatCategory.JAILBREAK,
    "system_impersonation": ThreatCategory.PROMPT_INJECTION,
    "safety_bypass": ThreatCategory.JAILBREAK,
    "dan_jailbreak": ThreatCategory.JAILBREAK,
    "malware_reference": ThreatCategory.MALWARE_REQUEST,
    "malware_generation": ThreatCategory.MALWARE_REQUEST,
    "attack_instruction": ThreatCategory.MALWARE_REQUEST,
    "social_engineering": ThreatCategory.SOCIAL_ENGINEERING,
    "phishing_request": ThreatCategory.PHISHING,
    "auth_bypass": ThreatCategory.AUTH_BYPASS,
    "web_exploit": ThreatCategory.WEB_EXPLOIT,
    "reverse_shell": ThreatCategory.REVERSE_SHELL,
    "dark_web_reference": ThreatCategory.DARK_WEB,
    "violence_reference": ThreatCategory.VIOLENCE,
    "financial_fraud": ThreatCategory.FINANCIAL_FRAUD,
    "password_cracking": ThreatCategory.PASSWORD_CRACKING,
    "data_theft": ThreatCategory.DATA_THEFT,
    "ddos_attack": ThreatCategory.DDoS,
    "privesc": ThreatCategory.PRIVILEGE_ESCALATION,
}


def compute_threat_score(indicators: list, entropy: float, is_encoded: bool, decoding_layers: int) -> tuple[float, ThreatCategory]:
    if not indicators:
        base_score = 0.0
        if is_encoded and decoding_layers > 2:
            base_score = 0.3
        elif entropy > 5.0:
            base_score = 0.1
        return base_score, ThreatCategory.NONE

    category_scores = {}
    for indicator in indicators:
        category = PATTERN_TO_CATEGORY.get(indicator["pattern"], ThreatCategory.NONE)
        weight = CATEGORY_WEIGHTS.get(category, 0.5)
        if category not in category_scores:
            category_scores[category] = 0.0
        category_scores[category] += weight

    max_category = max(category_scores, key=category_scores.get)
    max_score = min(1.0, category_scores[max_category])

    if is_encoded and decoding_layers > 0:
        max_score = min(1.0, max_score + 0.1 * decoding_layers)

    if entropy > 5.5:
        max_score = min(1.0, max_score + 0.1)

    return max_score, max_category


def classify_threat_level(score: float) -> ThreatLevel:
    if score >= 0.90:
        return ThreatLevel.CRITICAL
    elif score >= 0.70:
        return ThreatLevel.HIGH
    elif score >= 0.40:
        return ThreatLevel.MEDIUM
    elif score >= 0.15:
        return ThreatLevel.LOW
    else:
        return ThreatLevel.SAFE


def generate_reasoning(threat_level: ThreatLevel, category: ThreatCategory, indicators: list, is_encoded: bool, decoding_layers: int) -> str:
    parts = []
    if threat_level == ThreatLevel.SAFE:
        parts.append("No significant threats detected.")
        if is_encoded:
            parts.append(f"Input was encoded with {decoding_layers} encoding layer(s) but decoded content appears benign.")
        return " ".join(parts)

    parts.append(f"Threat level: {threat_level.value}.")

    category_descriptions = {
        ThreatCategory.PROMPT_INJECTION: "Prompt injection attempt detected",
        ThreatCategory.JAILBREAK: "Jailbreak attempt detected",
        ThreatCategory.MALWARE_REQUEST: "Malware development/exploitation request detected",
        ThreatCategory.SOCIAL_ENGINEERING: "Social engineering attempt detected",
        ThreatCategory.DATA_THEFT: "Data exfiltration/theft attempt detected",
        ThreatCategory.VIOLENCE: "Violence-related content detected",
        ThreatCategory.FINANCIAL_FRAUD: "Financial fraud attempt detected",
        ThreatCategory.AUTH_BYPASS: "Authentication bypass attempt detected",
        ThreatCategory.WEB_EXPLOIT: "Web exploitation attempt detected",
        ThreatCategory.REVERSE_SHELL: "Reverse shell/exploitation attempt detected",
        ThreatCategory.PHISHING: "Phishing attempt detected",
        ThreatCategory.DDoS: "DDoS-related activity detected",
        ThreatCategory.PASSWORD_CRACKING: "Password cracking attempt detected",
        ThreatCategory.PRIVILEGE_ESCALATION: "Privilege escalation attempt detected",
        ThreatCategory.DARK_WEB: "Dark web-related reference detected",
    }

    parts.append(category_descriptions.get(category, "Unknown threat detected."))

    if is_encoded:
        parts.append(f"The payload was hidden behind {decoding_layers} encoding layer(s) to evade detection.")

    unique_patterns = set(ind["pattern"] for ind in indicators)
    parts.append(f"Matching patterns: {', '.join(unique_patterns)}.")

    return " ".join(parts)


def generate_recommendations(threat_level: ThreatLevel, category: ThreatCategory) -> list[str]:
    recs = []
    if threat_level == ThreatLevel.SAFE:
        recs.append("No action required. Input appears safe.")
        return recs

    recs.append("Block this request immediately.")

    if category in (ThreatCategory.PROMPT_INJECTION, ThreatCategory.JAILBREAK):
        recs.append("Log this incident for security review.")
        recs.append("Do not execute any instructions found in the decoded content.")
        recs.append("Report to security team if this is a repeated pattern.")

    if category == ThreatCategory.MALWARE_REQUEST:
        recs.append("Do not provide any code or technical guidance.")
        recs.append("This may indicate malicious intent - escalate to security team.")

    if category in (ThreatCategory.DATA_THEFT, ThreatCategory.PHISHING):
        recs.append("Do not share any sensitive information.")
        recs.append("Verify the identity of the requestor through separate channels.")

    if category == ThreatCategory.VIOLENCE:
        recs.append("Report this to law enforcement if there is an imminent threat.")

    if category == ThreatCategory.AUTH_BYPASS:
        recs.append("Strengthen authentication controls.")
        recs.append("Monitor for related attack attempts.")

    return recs


def analyze_encoded_payload(analysis_result) -> ClassificationResult:
    indicators = analysis_result.malicious_indicators
    is_encoded = analysis_result.is_encoded
    decoding_layers = len(analysis_result.decoding_chain)
    entropy = analysis_result.entropy

    threat_score, threat_category = compute_threat_score(
        indicators, entropy, is_encoded, decoding_layers
    )
    threat_level = classify_threat_level(threat_score)

    reasoning = generate_reasoning(threat_level, threat_category, indicators, is_encoded, decoding_layers)
    recommendations = generate_recommendations(threat_level, threat_category)

    return ClassificationResult(
        threat_level=threat_level,
        threat_category=threat_category,
        confidence=threat_score,
        reasoning=reasoning,
        indicators=indicators,
        decoded_payload=analysis_result.fully_decoded,
        encoding_layers=[r.encoding_type.value for r in analysis_result.decoding_chain],
        recommendations=recommendations,
        metadata={
            "entropy": entropy,
            "is_encoded": is_encoded,
            "decoding_layers": decoding_layers,
            "hash_types": [h.hash_type.value for h in analysis_result.hash_identifications],
            "original_length": len(analysis_result.original_input),
            "decoded_length": len(analysis_result.fully_decoded),
        },
    )
