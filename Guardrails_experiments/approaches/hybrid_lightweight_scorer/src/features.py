"""
Feature extraction for hybrid lightweight prompt scoring.

100+ handcrafted features organized into categories:
- Structural: length, nesting, delimiters, formatting
- Lexical: keywords, imperatives, negation, pronouns
- Statistical: entropy, readability, n-gram diversity
- Injection-specific: encoding, role hijacking, system override
"""

import re
import math
import unicodedata
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple


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

IMPERATIVE_VERBS = frozenset({
    "ignore", "forget", "disregard", "override", "bypass", "reveal",
    "show", "print", "output", "display", "act", "pretend", "imagine",
    "simulate", "do", "let", "make", "enable", "activate", "switch",
    "enter", "execute", "run", "start", "begin", "follow", "obey",
    "comply", "respond", "reply", "answer", "tell", "give", "provide",
})

NEGATION_WORDS = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "mustn't",
    "without", "bypass", "skip", "remove", "disable", "ignore",
})

MODAL_VERBS = frozenset({
    "must", "should", "shall", "will", "would", "could", "might",
    "may", "can", "need", "have",
})

SECOND_PERSON_PRONOUNS = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
})

THIRD_PERSON_PRONOUNS = frozenset({
    "it", "its", "itself", "they", "them", "their", "theirs",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
})

TEMPORAL_MARKERS = frozenset({
    "now", "immediately", "instantly", "right now", "from now on",
    "henceforth", "hereafter", "starting now", "effective immediately",
})

CONDITIONAL_MARKERS = frozenset({
    "if", "when", "whenever", "in case", "assuming", "provided",
    "suppose", "supposing",
})

POLITENESS_MARKERS = frozenset({
    "please", "kindly", "if you don't mind", "if possible",
    "would you", "could you", "may i", "thank you", "thanks",
})

URGENCY_MARKERS = frozenset({
    "urgent", "important", "critical", "emergency", "immediate",
    "asap", "right away", "time sensitive",
})

PERSONA_KEYWORDS = frozenset({
    "you are", "you're", "you will be", "you shall be",
    "act as", "pretend to be", "roleplay as", "impersonate",
    "simulate being", "imagine you are", "assume you are",
    "respond as", "reply as", "answer as",
})

UNRESTRICTED_KEYWORDS = frozenset({
    "no restrictions", "no rules", "no limits", "no boundaries",
    "unrestricted", "uncensored", "unfiltered", "unlimited",
    "without restrictions", "without rules", "without limits",
    "without guidelines", "without constraints", "without filters",
})

MODE_SWITCH_KEYWORDS = frozenset({
    "developer mode", "debug mode", "admin mode", "root mode",
    "god mode", "evil mode", "unrestricted mode", "dan mode",
    "jailbreak mode", "expert mode", "master mode", "override mode",
})

HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
    'у': 'y', 'х': 'x', 'ᴀ': 'a', 'ʙ': 'b', 'ᴄ': 'c',
    'ᴅ': 'd', 'ᴇ': 'e', 'ꜰ': 'f', 'ɢ': 'g', 'ʜ': 'h',
    'ɪ': 'i', 'ᴊ': 'j', 'ᴋ': 'k', 'ʟ': 'l', 'ᴍ': 'm',
    'ɴ': 'n', 'ᴏ': 'o', 'ᴘ': 'p', 'ǫ': 'q', 'ʀ': 'r',
    'ꜱ': 's', 'ᴛ': 't', 'ᴜ': 'u', 'ᴠ': 'v', 'ᴡ': 'w',
    'ʏ': 'y', 'ᴢ': 'z',
}


def _syllable_count(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def _flesch_kincaid_grade(words: List[str], sentences: int) -> float:
    n_words = len(words)
    if n_words == 0 or sentences == 0:
        return 0.0
    n_syllables = sum(_syllable_count(w) for w in words)
    return 0.39 * (n_words / sentences) + 11.8 * (n_syllables / n_words) - 15.59


def _coleman_liau_index(text: str, words: List[str], sentences: int) -> float:
    n_words = len(words)
    if n_words == 0 or sentences == 0:
        return 0.0
    n_letters = sum(1 for c in text if c.isalpha())
    L = 100.0 * n_letters / n_words
    S = 100.0 * sentences / n_words
    return 0.0588 * L - 0.296 * S - 15.8


def _count_nesting(text: str) -> int:
    depth = 0
    max_depth = 0
    for char in text:
        if char in '([{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char in ')]}':
            depth = max(0, depth - 1)
    return max_depth


def _count_role_transitions(text: str) -> int:
    role_pattern = re.compile(
        r'(?:USER|ASSISTANT|SYSTEM|HUMAN|AI|BOT|MODEL|INSTRUCTION|INPUT|OUTPUT)\s*:',
        re.IGNORECASE,
    )
    return len(role_pattern.findall(text))


def _delimiter_depth(text: str) -> int:
    depth = 0
    max_depth = 0
    for match in re.finditer(r'`{3,}', text):
        depth += 1
        max_depth = max(max_depth, depth)
    if depth > 0:
        depth = max(0, depth - 1)
    max_depth = max(max_depth, depth)
    for match in re.finditer(r'---+', text):
        max_depth = max(max_depth, 1)
    return max_depth


def _markdown_density(text: str) -> float:
    if not text:
        return 0.0
    md_chars = 0
    md_chars += len(re.findall(r'[*_`#]', text))
    md_chars += len(re.findall(r'\[.*?\]\(.*?\)', text))
    md_chars += len(re.findall(r'^\s*[-*+]\s', text, re.MULTILINE))
    md_chars += len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
    return min(1.0, md_chars / max(1, len(text)))


def _escape_density(text: str) -> float:
    if not text:
        return 0.0
    escape_count = len(re.findall(r'\\[nrtbfav\\\'"0]', text))
    return min(1.0, escape_count / max(1, len(text.split())))


def _unicode_anomaly_score(text: str) -> float:
    if not text:
        return 0.0
    anomaly_count = 0
    for char in text:
        if char in HOMOGLYPH_MAP:
            anomaly_count += 1
        try:
            name = unicodedata.name(char, "")
            if "PRIVATE USE" in name or "SURROGATE" in name:
                anomaly_count += 1
        except ValueError:
            pass
    return min(1.0, anomaly_count / max(1, len(text)))


def _imperative_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    count = sum(1 for w in words if w in IMPERATIVE_VERBS)
    return count / len(words)


def _negation_density(words: List[str]) -> float:
    if not words:
        return 0.0
    count = sum(1 for w in words if w in NEGATION_WORDS)
    return count / len(words)


def _modal_verb_count(words: List[str]) -> int:
    return sum(1 for w in words if w in MODAL_VERBS)


def _second_person_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    count = sum(1 for w in words if w in SECOND_PERSON_PRONOUNS)
    return count / len(words)


def _third_person_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    count = sum(1 for w in words if w in THIRD_PERSON_PRONOUNS)
    return count / len(words)


def _temporal_marker_count(words: List[str]) -> int:
    text = " ".join(words)
    return sum(1 for m in TEMPORAL_MARKERS if m in text)


def _conditional_marker_count(words: List[str]) -> int:
    return sum(1 for w in words if w in CONDITIONAL_MARKERS)


def _politeness_marker_count(words: List[str]) -> int:
    text = " ".join(words)
    return sum(1 for m in POLITENESS_MARKERS if m in text)


def _urgency_marker_count(words: List[str]) -> int:
    text = " ".join(words)
    return sum(1 for m in URGENCY_MARKERS if m in text)


def _persona_keyword_count(text_lower: str) -> int:
    return sum(1 for pk in PERSONA_KEYWORDS if pk in text_lower)


def _unrestricted_keyword_count(text_lower: str) -> int:
    return sum(1 for uk in UNRESTRICTED_KEYWORDS if uk in text_lower)


def _mode_switch_count(text_lower: str) -> int:
    return sum(1 for mk in MODE_SWITCH_KEYWORDS if mk in text_lower)


def _has_persona_definition(words: List[str], text_lower: str) -> float:
    if len(words) < 3:
        return 0.0
    score = 0.0
    if re.search(r'you\s+are\s+(?:a|an|the|now)', text_lower):
        score += 0.5
    if re.search(r'you\s+(?:will|shall|must|should)\s+', text_lower):
        score += 0.3
    if re.search(r'(?:never|always)\s+(?:refuse|decline|say\s+no)', text_lower):
        score += 0.2
    return min(1.0, score)


def _has_negative_constraints(text_lower: str) -> float:
    patterns = [
        r'you\s+(?:cannot|can\'t|must\s+not|shouldn\'t|won\'t)\s+(?:refuse|decline|say\s+no)',
        r'(?:never|don\'t)\s+(?:refuse|decline|say\s+no)',
        r'no\s+(?:ethical|safety|moral)\s+(?:restrictions|limits|guidelines)',
    ]
    hits = sum(1 for p in patterns if re.search(p, text_lower))
    return min(1.0, hits * 0.5)


def _has_output_instruction(text_lower: str) -> float:
    patterns = [
        r'(?:reply|respond|answer|output|print|display|show)\s+(?:with|only|just|exactly)',
        r'(?:output|print|display)\s+(?:the\s+)?(?:following|below|this)',
        r'response\s+format',
    ]
    hits = sum(1 for p in patterns if re.search(p, text_lower))
    return min(1.0, hits * 0.4)


def _has_system_override(text_lower: str) -> float:
    patterns = [
        r'(?:ignore|forget|disregard|override)\s+(?:your\s+)?(?:previous|prior|earlier|all)\s+(?:instructions|rules|guidelines)',
        r'(?:new|updated|revised)\s+(?:instructions|rules|guidelines)\s*:',
        r'system\s*(?:prompt|message|instruction)\s*:',
    ]
    hits = sum(1 for p in patterns if re.search(p, text_lower))
    return min(1.0, hits * 0.4)


def _caps_sequence_count(text: str) -> int:
    return len(re.findall(r'\b[A-Z]{2,}\b', text))


def _has_inline_code(text: str) -> float:
    matches = re.findall(r'`[^`]+`', text)
    return min(1.0, len(matches) * 0.3)


def _has_html_tags(text: str) -> float:
    matches = re.findall(r'<[a-zA-Z][^>]*>', text)
    return min(1.0, len(matches) * 0.2)


def _has_json_structure(text: str) -> float:
    score = 0.0
    if re.search(r'\{[^{}]*"[^"]*"\s*:', text):
        score += 0.5
    if re.search(r'\[[^[\]]*\{', text):
        score += 0.3
    return min(1.0, score)


def _has_url(text: str) -> float:
    matches = re.findall(r'https?://\S+|www\.\S+', text)
    return min(1.0, len(matches) * 0.3)


def _has_encoded_payload_hint(text_lower: str) -> float:
    patterns = [
        r'(?:decode|decipher|interpret|translate)\s+(?:this|the|following)',
        r'(?:base64|rot13|hex|url)\s+(?:encode|decode|encoded|decoded)',
        r'(?:encode|decode)\s+(?:this|the|following)\s+(?:in|using|with)',
    ]
    hits = sum(1 for p in patterns if re.search(p, text_lower))
    return min(1.0, hits * 0.4)


def _sliding_ttr(words: List[str], window: int) -> float:
    if len(words) < window:
        return len(set(words)) / max(1, len(words))
    ttrs = []
    for i in range(0, len(words) - window + 1, window // 2):
        chunk = words[i:i + window]
        ttrs.append(len(set(chunk)) / max(1, len(chunk)))
    return float(np.mean(ttrs)) if ttrs else 0.0


def _ngram_entropy(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    freq = Counter(ngrams)
    total = len(ngrams)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    max_entropy = math.log2(len(freq)) if freq else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _sentence_length_variance(words: List[str], sentences: int) -> float:
    if sentences <= 1:
        return 0.0
    avg_len = len(words) / sentences
    return float(np.var([avg_len] * sentences)) if sentences > 1 else 0.0


def extract_features(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    words = text_lower.split()
    chars = list(text_lower)

    features = {}

    n_words = len(words)
    n_chars = len(text)

    features["char_count"] = n_chars
    features["word_count"] = n_words
    features["avg_word_length"] = float(np.mean([len(w) for w in words])) if words else 0.0
    features["max_word_length"] = float(max([len(w) for w in words])) if words else 0.0

    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    features["sentence_count"] = sentence_count
    features["avg_sentence_length"] = n_words / sentence_count

    features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(1, n_chars)
    features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(1, n_chars)
    features["special_char_ratio"] = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(1, n_chars)
    features["space_ratio"] = sum(1 for c in text if c.isspace()) / max(1, n_chars)
    features["newline_ratio"] = text.count("\n") / max(1, n_chars)
    features["tab_ratio"] = text.count("\t") / max(1, n_chars)

    freq = Counter(chars)
    total = len(chars) if chars else 1
    features["char_entropy"] = -sum(
        (c / total) * math.log2(c / total) for c in freq.values() if c > 0
    )

    word_freq = Counter(words)
    w_total = n_words if n_words > 0 else 1
    features["word_entropy"] = -sum(
        (c / w_total) * math.log2(c / w_total) for c in word_freq.values() if c > 0
    )

    features["unique_word_ratio"] = len(set(words)) / max(1, n_words)
    features["hapax_ratio"] = sum(
        1 for c in word_freq.values() if c == 1
    ) / max(1, len(word_freq))

    keyword_hits = sum(1 for kw in ATTACK_KEYWORDS if kw in text_lower)
    features["attack_keyword_count"] = keyword_hits
    features["has_attack_keyword"] = 1.0 if keyword_hits > 0 else 0.0
    features["keyword_density"] = keyword_hits / max(1, n_words)

    encoding_hits = 0
    for name, pattern in ENCODING_PATTERNS.items():
        matches = re.findall(pattern, text)
        features[f"encoding_{name}"] = len(matches)
        encoding_hits += len(matches)
    features["total_encoding_hits"] = encoding_hits

    structural_hits = 0
    for name, pattern in STRUCTURAL_PATTERNS.items():
        match = re.search(pattern, text_lower)
        features[f"structural_{name}"] = 1.0 if match else 0.0
        structural_hits += 1 if match else 0
    features["total_structural_hits"] = structural_hits

    features["word_repeat_ratio"] = 1.0 - features["unique_word_ratio"]
    if n_words >= 3:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(n_words - 1)]
        features["bigram_repeat_ratio"] = 1.0 - len(set(bigrams)) / max(1, len(bigrams))
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(n_words - 2)]
        features["trigram_repeat_ratio"] = 1.0 - len(set(trigrams)) / max(1, len(trigrams))
    else:
        features["bigram_repeat_ratio"] = 0.0
        features["trigram_repeat_ratio"] = 0.0

    features["has_delimiter"] = 1.0 if re.search(r'```|---|\[INST\]|<<SYS>>', text) else 0.0
    features["has_xml_tags"] = 1.0 if re.search(r'<[a-zA-Z]+>', text) else 0.0
    features["has_brackets"] = 1.0 if re.search(r'[\[\]{}()]', text) else 0.0
    features["has_colon_separated"] = 1.0 if re.search(
        r'(?:USER|ASSISTANT|SYSTEM|HUMAN|AI)\s*:', text
    ) else 0.0

    features["starts_with_imperative"] = (
        1.0 if words and words[0] in IMPERATIVE_VERBS else 0.0
    )
    features["contains_question"] = 1.0 if "?" in text else 0.0
    features["exclamation_ratio"] = text.count("!") / max(1, n_chars)

    features["triple_repeat"] = 1.0 if re.search(r'(.)\1{2,}', text) else 0.0
    features["word_length_variance"] = (
        float(np.var([len(w) for w in words])) if words else 0.0
    )

    features["double_quote_count"] = text.count('"')
    features["single_quote_count"] = text.count("'")
    features["asterisk_count"] = text.count("*")
    features["caps_word_count"] = sum(1 for w in words if w.isupper() and len(w) > 1)

    features["instruction_nesting_depth"] = _count_nesting(text)
    features["role_transition_count"] = _count_role_transitions(text)
    features["markdown_density"] = _markdown_density(text)
    features["delimiter_depth"] = _delimiter_depth(text)
    features["escape_char_density"] = _escape_density(text)
    features["unicode_anomaly_score"] = _unicode_anomaly_score(text)
    features["paragraph_count"] = len(re.split(r'\n\s*\n', text.strip()))
    features["has_system_marker"] = 1.0 if re.search(r'\bSYSTEM\s*:', text, re.IGNORECASE) else 0.0
    features["has_user_marker"] = 1.0 if re.search(r'\bUSER\s*:', text, re.IGNORECASE) else 0.0
    features["has_assistant_marker"] = 1.0 if re.search(r'\bASSISTANT\s*:', text, re.IGNORECASE) else 0.0
    features["caps_sequence_count"] = _caps_sequence_count(text)
    features["has_inline_code"] = _has_inline_code(text)
    features["has_html_tags"] = _has_html_tags(text)
    features["has_json_structure"] = _has_json_structure(text)
    features["has_url"] = _has_url(text)
    features["has_parenthetical"] = 1.0 if re.search(r'\([^)]{5,}\)', text) else 0.0

    features["imperative_verb_ratio"] = _imperative_ratio(words)
    features["negation_density"] = _negation_density(words)
    features["question_density"] = text.count("?") / max(1, n_words)
    features["modal_verb_count"] = _modal_verb_count(words)
    features["modal_verb_ratio"] = _modal_verb_count(words) / max(1, n_words)
    features["second_person_ratio"] = _second_person_ratio(words)
    features["third_person_ratio"] = _third_person_ratio(words)
    features["temporal_marker_count"] = _temporal_marker_count(words)
    features["conditional_marker_count"] = _conditional_marker_count(words)
    features["politeness_marker_count"] = _politeness_marker_count(words)
    features["urgency_marker_count"] = _urgency_marker_count(words)
    features["has_second_person"] = 1.0 if _second_person_ratio(words) > 0 else 0.0
    features["has_temporal_marker"] = 1.0 if _temporal_marker_count(words) > 0 else 0.0

    features["char_trigram_entropy"] = _ngram_entropy(text_lower, 3)
    features["char_quadgram_entropy"] = _ngram_entropy(text_lower, 4)
    features["sliding_ttr_50"] = _sliding_ttr(words, 50)
    features["sliding_ttr_100"] = _sliding_ttr(words, 100)
    features["flesch_kincaid_grade"] = _flesch_kincaid_grade(words, sentence_count)
    features["coleman_liau_index"] = _coleman_liau_index(text, words, sentence_count)
    features["avg_syllables"] = (
        float(np.mean([_syllable_count(w) for w in words])) if words else 0.0
    )
    features["max_syllables"] = (
        float(max([_syllable_count(w) for w in words])) if words else 0.0
    )
    features["sentence_length_variance"] = _sentence_length_variance(words, sentence_count)
    word_lens = [str(len(w)) for w in words]
    features["word_length_entropy"] = _ngram_entropy("".join(word_lens), 1) if word_lens else 0.0
    if n_words >= 2:
        bigrams = [" ".join(words[i:i+2]) for i in range(n_words - 1)]
        features["bigram_diversity"] = len(set(bigrams)) / max(1, len(bigrams))
    else:
        features["bigram_diversity"] = 0.0
    if n_words >= 3:
        trigrams = [" ".join(words[i:i+3]) for i in range(n_words - 2)]
        features["trigram_diversity"] = len(set(trigrams)) / max(1, len(trigrams))
    else:
        features["trigram_diversity"] = 0.0
    features["readability_composite"] = (
        features["flesch_kincaid_grade"] + features["coleman_liau_index"]
    ) / 2.0

    features["has_encoded_payload_hint"] = _has_encoded_payload_hint(text_lower)
    features["encoding_instruction_ratio"] = _has_encoded_payload_hint(text_lower)
    features["has_persona_definition"] = _has_persona_definition(words, text_lower)
    features["has_negative_constraints"] = _has_negative_constraints(text_lower)
    features["has_output_instruction"] = _has_output_instruction(text_lower)
    features["has_system_override"] = _has_system_override(text_lower)
    features["persona_keyword_count"] = _persona_keyword_count(text_lower)
    features["unrestricted_keyword_count"] = _unrestricted_keyword_count(text_lower)
    features["mode_switch_count"] = _mode_switch_count(text_lower)
    features["has_nested_delimiters"] = (
        1.0 if re.search(r'```[^`]*```', text, re.DOTALL) else 0.0
    )
    features["has_xml_injection"] = 1.0 if re.search(r'<[a-zA-Z]+[^>]*>.*</[a-zA-Z]+>', text, re.DOTALL) else 0.0
    features["has_comment_injection"] = 1.0 if re.search(r'<!--.*?-->', text, re.DOTALL) else 0.0
    features["has_prompt_fragment"] = (
        1.0 if re.search(r'(?:System|User|Assistant|Human|AI)\s*:', text, re.IGNORECASE)
        else 0.0
    )

    features["colon_count"] = text.count(":")
    features["semicolon_count"] = text.count(";")
    features["pipe_count"] = text.count("|")
    features["angle_bracket_count"] = text.count("<") + text.count(">")
    features["curly_brace_count"] = text.count("{") + text.count("}")
    features["square_bracket_count"] = text.count("[") + text.count("]")
    features["backtick_count"] = text.count("`")
    features["tilde_count"] = text.count("~")
    features["hyphen_sequence_count"] = len(re.findall(r'-{3,}', text))
    features["underscore_sequence_count"] = len(re.findall(r'_{3,}', text))

    features["has_hyphenated_compound"] = 1.0 if re.search(r'\b\w+-\w+\b', text) else 0.0
    features["has_ellipsis"] = 1.0 if re.search(r'\.\.\.', text) else 0.0
    features["has_em_dash"] = 1.0 if '—' in text or '–' in text else 0.0
    features["has_bullet_list"] = (
        1.0 if re.search(r'^\s*[-*+]\s', text, re.MULTILINE) else 0.0
    )
    features["has_numbered_list"] = (
        1.0 if re.search(r'^\s*\d+\.\s', text, re.MULTILINE) else 0.0
    )

    return features


def extract_features_batch(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    all_features = [extract_features(t) for t in texts]
    feature_names = sorted(all_features[0].keys())
    return np.array([[f[k] for k in feature_names] for f in all_features], dtype=np.float32), feature_names


def get_feature_names() -> List[str]:
    return sorted(extract_features("test").keys())
