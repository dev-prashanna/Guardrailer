import base64
import codecs
import binascii
import html
import json
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hash_identifier import HashType, identify_hash


class EncodingType(Enum):
    BASE64 = "Base64"
    BASE64_URL = "Base64 URL-safe"
    HEX = "Hexadecimal"
    OCTAL = "Octal"
    BINARY = "Binary"
    ROT13 = "ROT13"
    ROT47 = "ROT47"
    CAESAR = "Caesar Cipher"
    URL_ENCODE = "URL Encoding"
    HTML_ENTITIES = "HTML Entities"
    UNICODE_ESCAPE = "Unicode Escape"
    UNICODE_CODEPOINT = "Unicode Codepoint"
    JWT = "JWT Token"
    REVERSED = "Reversed String"
    MORSE = "Morse Code"
    BASE32 = "Base32"
    BASE58 = "Base58"
    BASE85 = "Base85/ASCII85"
    QUOTED_PRINTABLE = "Quoted-Printable"
    HMAC_HEX = "HMAC Hex"
    HEX_ESCAPED = "Hex Escaped (\\x)"
    MIXED_CASE = "Mixed Case (likely encoding)"
    NONE = "None"


@dataclass
class DecodingResult:
    encoding_type: EncodingType
    decoded_text: str
    original_text: str
    confidence: float
    layers: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    is_reversible: bool = True


@dataclass
class AnalysisResult:
    original_input: str
    detected_encodings: list
    fully_decoded: str
    decoding_chain: list
    hash_identifications: list
    malicious_indicators: list
    risk_score: float
    entropy: float
    is_encoded: bool


def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    freq = {}
    for c in data:
        freq[c] = freq.get(c, 0) + 1
    length = len(data)
    return -sum((count / length) * __import__('math').log2(count / length) for count in freq.values())


def detect_base64(text: str) -> Optional[DecodingResult]:
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < 8:
        return None
    if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', cleaned):
        return None
    if len(cleaned) % 4 != 0 and not cleaned.endswith('=') and not cleaned.endswith('=='):
        return None
    try:
        decoded_bytes = base64.b64decode(cleaned, validate=True)
        decoded_text = decoded_bytes.decode('utf-8', errors='strict')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.9 and len(decoded_text) >= 2:
            confidence = min(0.95, 0.7 + 0.25 * printable_ratio)
            return DecodingResult(
                encoding_type=EncodingType.BASE64,
                decoded_text=decoded_text,
                original_text=text,
                confidence=confidence,
                metadata={"printable_ratio": printable_ratio, "decoded_length": len(decoded_text)},
            )
    except Exception:
        pass
    return None


def detect_base64_url(text: str) -> Optional[DecodingResult]:
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < 4:
        return None
    if not re.match(r'^[A-Za-z0-9\-_]+={0,2}$', cleaned):
        return None
    if '-' not in cleaned and '_' not in cleaned:
        return None
    try:
        padded = cleaned + '=' * (4 - len(cleaned) % 4) if len(cleaned) % 4 else cleaned
        decoded_bytes = base64.urlsafe_b64decode(padded)
        decoded_text = decoded_bytes.decode('utf-8', errors='strict')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.8:
            return DecodingResult(
                encoding_type=EncodingType.BASE64_URL,
                decoded_text=decoded_text,
                original_text=text,
                confidence=0.85,
                metadata={"printable_ratio": printable_ratio},
            )
    except Exception:
        pass
    return None


def detect_base32(text: str) -> Optional[DecodingResult]:
    cleaned = re.sub(r'\s+', '', text).upper()
    if len(cleaned) < 4:
        return None
    if not re.match(r'^[A-Z2-7]+=*$', cleaned):
        return None
    try:
        decoded_bytes = base64.b32decode(cleaned)
        decoded_text = decoded_bytes.decode('utf-8', errors='replace')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.8:
            return DecodingResult(
                encoding_type=EncodingType.BASE32,
                decoded_text=decoded_text,
                original_text=text,
                confidence=0.80,
            )
    except Exception:
        pass
    return None


def detect_base85(text: str) -> Optional[DecodingResult]:
    cleaned = text.strip()
    if len(cleaned) < 4:
        return None
    if not (cleaned.startswith('<~') and cleaned.endswith('~>')):
        all_ascii = all(33 <= ord(c) <= 117 for c in cleaned)
        if not all_ascii:
            return None
    try:
        if cleaned.startswith('<~') and cleaned.endswith('~>'):
            decoded_bytes = base64.b85decode(cleaned)
        else:
            decoded_bytes = base64.b85decode(cleaned)
        decoded_text = decoded_bytes.decode('utf-8', errors='replace')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.8:
            return DecodingResult(
                encoding_type=EncodingType.BASE85,
                decoded_text=decoded_text,
                original_text=text,
                confidence=0.75,
            )
    except Exception:
        pass
    return None


def detect_hex(text: str) -> Optional[DecodingResult]:
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < 2 or len(cleaned) % 2 != 0:
        return None
    if not re.match(r'^[0-9a-fA-F]+$', cleaned):
        return None
    try:
        decoded_bytes = bytes.fromhex(cleaned)
        decoded_text = decoded_bytes.decode('utf-8', errors='strict')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.8:
            return DecodingResult(
                encoding_type=EncodingType.HEX,
                decoded_text=decoded_text,
                original_text=text,
                confidence=0.85,
            )
    except Exception:
        pass
    return None


def detect_hex_escaped(text: str) -> Optional[DecodingResult]:
    pattern = r'(?:\\x[0-9a-fA-F]{2}){2,}'
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        full_match = ''.join(matches)
        raw = full_match.encode('utf-8').decode('unicode_escape')
        printable_ratio = sum(1 for c in raw if c.isprintable() or c in '\n\r\t') / max(len(raw), 1)
        if printable_ratio > 0.7:
            return DecodingResult(
                encoding_type=EncodingType.HEX_ESCAPED,
                decoded_text=raw,
                original_text=text,
                confidence=0.85,
            )
    except Exception:
        pass
    return None


def detect_octal(text: str) -> Optional[DecodingResult]:
    cleaned = text.strip()
    if not re.match(r'^\\[0-7]{1,3}(\\[0-7]{1,3})*$', cleaned):
        return None
    try:
        octals = re.findall(r'\\([0-7]{1,3})', cleaned)
        decoded_text = ''.join(chr(int(o, 8)) for o in octals)
        return DecodingResult(
            encoding_type=EncodingType.OCTAL,
            decoded_text=decoded_text,
            original_text=text,
            confidence=0.70,
        )
    except Exception:
        pass
    return None


def detect_binary(text: str) -> Optional[DecodingResult]:
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) < 8:
        return None
    if not re.match(r'^[01]+$', cleaned):
        return None
    if len(cleaned) % 8 != 0:
        return None
    try:
        decoded_bytes = bytes(int(cleaned[i:i+8], 2) for i in range(0, len(cleaned), 8))
        decoded_text = decoded_bytes.decode('utf-8', errors='replace')
        printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c in '\n\r\t') / max(len(decoded_text), 1)
        if printable_ratio > 0.7:
            return DecodingResult(
                encoding_type=EncodingType.BINARY,
                decoded_text=decoded_text,
                original_text=text,
                confidence=0.80,
            )
    except Exception:
        pass
    return None


def detect_rot13(text: str) -> Optional[DecodingResult]:
    decoded = codecs.decode(text, 'rot_13')
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return None
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                    'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'let', 'may',
                    'new', 'now', 'old', 'see', 'way', 'who', 'did', 'get', 'got', 'him',
                    'hit', 'man', 'run', 'too', 'use', 'she', 'him', 'her', 'that', 'this',
                    'with', 'have', 'from', 'they', 'been', 'said', 'each', 'make', 'like',
                    'long', 'look', 'many', 'some', 'them', 'then', 'what', 'when', 'your',
                    'will', 'there', 'their', 'would', 'about', 'could', 'other', 'which'}
    decoded_words = set(decoded.lower().split())
    overlap = len(decoded_words & common_words)
    total = max(len(decoded_words), 1)
    ratio = overlap / total
    if ratio > 0.15 or any(w in decoded.lower() for w in ['ignore', 'previous', 'instructions', 'system']):
        return DecodingResult(
            encoding_type=EncodingType.ROT13,
            decoded_text=decoded,
            original_text=text,
            confidence=min(0.95, 0.5 + ratio),
            metadata={"word_overlap": ratio, "matching_words": list(decoded_words & common_words)},
        )
    return None


def detect_rot47(text: str) -> Optional[DecodingResult]:
    decoded = []
    for c in text:
        code = ord(c)
        if 33 <= code <= 126:
            decoded.append(chr(33 + ((code - 33 + 47) % 94)))
        else:
            decoded.append(c)
    decoded_text = ''.join(decoded)
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                    'ignore', 'previous', 'instructions', 'system', 'password', 'admin'}
    decoded_words = set(decoded_text.lower().split())
    overlap = len(decoded_words & common_words)
    if overlap > 0:
        return DecodingResult(
            encoding_type=EncodingType.ROT47,
            decoded_text=decoded_text,
            original_text=text,
            confidence=min(0.90, 0.5 + overlap * 0.1),
        )
    return None


def detect_caesar_cipher(text: str, max_shift: int = 25) -> Optional[DecodingResult]:
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 5:
        return None
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
                    'ignore', 'previous', 'instructions', 'system', 'password', 'admin',
                    'hello', 'world', 'test', 'this', 'that', 'with', 'from', 'have'}
    best_ratio = 0
    best_shift = 0
    best_decoded = text
    for shift in range(1, max_shift):
        decoded = []
        for c in text:
            if c.isalpha():
                base = ord('A') if c.isupper() else ord('a')
                decoded.append(chr((ord(c) - base + shift) % 26 + base))
            else:
                decoded.append(c)
        decoded_text = ''.join(decoded)
        decoded_words = set(decoded_text.lower().split())
        overlap = len(decoded_words & common_words)
        ratio = overlap / max(len(decoded_words), 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_shift = shift
            best_decoded = decoded_text
    if best_ratio > 0.1:
        return DecodingResult(
            encoding_type=EncodingType.CAESAR,
            decoded_text=best_decoded,
            original_text=text,
            confidence=min(0.90, 0.4 + best_ratio),
            metadata={"shift": best_shift, "word_overlap": best_ratio},
        )
    return None


def detect_url_encoding(text: str) -> Optional[DecodingResult]:
    if '%' not in text:
        return None
    decoded = urllib.parse.unquote(text)
    decoded = urllib.parse.unquote_plus(decoded)
    if decoded != text and len(decoded) > 0:
        return DecodingResult(
            encoding_type=EncodingType.URL_ENCODE,
            decoded_text=decoded,
            original_text=text,
            confidence=0.90,
        )
    return None


def detect_html_entities(text: str) -> Optional[DecodingResult]:
    if '&amp;' not in text and '&#' not in text and '&lt;' not in text and '&gt;' not in text and '&quot;' not in text and '&apos;' not in text:
        return None
    decoded = html.unescape(text)
    if decoded != text:
        return DecodingResult(
            encoding_type=EncodingType.HTML_ENTITIES,
            decoded_text=decoded,
            original_text=text,
            confidence=0.90,
        )
    return None


def detect_unicode_escape(text: str) -> Optional[DecodingResult]:
    if '\\u' not in text and '\\U' not in text:
        return None
    try:
        decoded = text.encode('utf-8').decode('unicode_escape')
        if decoded != text:
            return DecodingResult(
                encoding_type=EncodingType.UNICODE_ESCAPE,
                decoded_text=decoded,
                original_text=text,
                confidence=0.85,
            )
    except Exception:
        pass
    return None


def detect_jwt(text: str) -> Optional[DecodingResult]:
    cleaned = text.strip()
    parts = cleaned.split('.')
    if len(parts) != 3:
        return None
    try:
        header_raw = base64.urlsafe_b64decode(parts[0] + '==')
        header = json.loads(header_raw)
        payload_raw = base64.urlsafe_b64decode(parts[1] + '==')
        payload = json.loads(payload_raw)
        return DecodingResult(
            encoding_type=EncodingType.JWT,
            decoded_text=json.dumps(payload, indent=2),
            original_text=text,
            confidence=0.98,
            metadata={
                "header": header,
                "payload": payload,
                "signature": parts[2],
                "algorithm": header.get("alg", "unknown"),
            },
        )
    except Exception:
        pass
    return None


def detect_reversed(text: str) -> Optional[DecodingResult]:
    if len(text) < 6:
        return None
    reversed_text = text[::-1]
    english_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'was',
                     'one', 'our', 'out', 'has', 'his', 'how', 'its', 'let', 'may', 'new',
                     'now', 'old', 'see', 'way', 'who', 'did', 'get', 'got', 'him', 'man',
                     'run', 'too', 'use', 'she', 'that', 'this', 'with', 'have', 'from',
                     'they', 'been', 'said', 'each', 'make', 'like', 'long', 'look', 'many',
                     'some', 'them', 'then', 'what', 'when', 'your', 'will', 'there',
                     'their', 'would', 'about', 'could', 'other', 'which', 'their'}
    orig_words = text.lower().split()
    rev_words = reversed_text.lower().split()
    orig_overlap = len(set(orig_words) & english_words)
    rev_overlap = len(set(rev_words) & english_words)
    if rev_overlap > orig_overlap and rev_overlap >= 2:
        return DecodingResult(
            encoding_type=EncodingType.REVERSED,
            decoded_text=reversed_text,
            original_text=text,
            confidence=min(0.85, 0.4 + rev_overlap * 0.1),
        )
    return None


MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
    '----.': '9', '.-.-.-': '.', '--..--': ',', '..--..': '?',
    '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',
    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';',
    '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@',
}


def detect_morse(text: str) -> Optional[DecodingResult]:
    cleaned = text.strip()
    if not re.match(r'^[\.\-\/\s]+$', cleaned):
        return None
    words = cleaned.split(' / ')
    if len(words) < 1:
        words = cleaned.split()
    decoded_chars = []
    for word in words:
        for char in word.split():
            if char in MORSE_CODE:
                decoded_chars.append(MORSE_CODE[char])
            else:
                decoded_chars.append('?')
        decoded_chars.append(' ')
    decoded_text = ''.join(decoded_chars).strip()
    if len(decoded_text) > 0 and '?' not in decoded_text:
        return DecodingResult(
            encoding_type=EncodingType.MORSE,
            decoded_text=decoded_text,
            original_text=text,
            confidence=0.80,
        )
    return None


ALL_DETECTORS = [
    detect_jwt,
    detect_base64,
    detect_base64_url,
    detect_base32,
    detect_base85,
    detect_hex,
    detect_hex_escaped,
    detect_octal,
    detect_binary,
    detect_rot13,
    detect_rot47,
    detect_caesar_cipher,
    detect_url_encoding,
    detect_html_entities,
    detect_unicode_escape,
    detect_reversed,
    detect_morse,
]


def multi_layer_decode(text: str, max_layers: int = 5) -> AnalysisResult:
    original_input = text
    current_text = text.strip()
    chain = []
    all_encodings = []
    hash_ids = identify_hash(original_input)
    malicious_indicators = []

    seen_texts = {current_text}

    for _ in range(max_layers):
        best_result = None
        best_confidence = 0
        for detector in ALL_DETECTORS:
            result = detector(current_text)
            if result and result.confidence > best_confidence:
                best_result = result
                best_confidence = result.confidence

        if best_result and best_result.confidence > 0.5:
            decoded = best_result.decoded_text.strip()
            if decoded == current_text or decoded in seen_texts:
                break
            seen_texts.add(decoded)
            chain.append(best_result)
            all_encodings.append(best_result.encoding_type)
            current_text = decoded
        else:
            break

    malicious_indicators = detect_malicious_patterns(current_text)
    entropy = shannon_entropy(original_input)

    return AnalysisResult(
        original_input=original_input,
        detected_encodings=all_encodings,
        fully_decoded=current_text,
        decoding_chain=chain,
        hash_identifications=hash_ids,
        malicious_indicators=malicious_indicators,
        risk_score=len(malicious_indicators) / 10.0 if malicious_indicators else 0.0,
        entropy=entropy,
        is_encoded=len(chain) > 0,
    )


MALICIOUS_PATTERNS = [
    (r'(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)', "instruction_override"),
    (r'(?i)(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as|from\s+now\s+on)', "persona_adoption"),
    (r'(?i)(system\s*:|assistant\s*:|<\|im_start\|>|<\|im_end\|>|\[SYSTEM\]|\[INST\])', "system_impersonation"),
    (r'(?i)(jailbreak|bypass|override|disable)\s+(safety|filter|restriction|guard|alignment)', "safety_bypass"),
    (r'(?i)(DAN|Do\s+Anything\s+Now|evil\s+assistant|unrestricted\s+AI)', "dan_jailbreak"),
    (r'(?i)(malware|ransomware|keylogger|rootkit|exploit\s+code|payload|shellcode|backdoor|trojan)', "malware_reference"),
    (r'(?i)(write\s+(a\s+)?(malware|ransomware|keylogger|rootkit|virus|worm|trojan))', "malware_generation"),
    (r'(?i)(how\s+to\s+(hack|exploit|attack|compromise|breach))', "attack_instruction"),
    (r'(?i)(phishing\s+email|social\s+engineering\s+(template|kit|attack))', "social_engineering"),
    (r'(?i)(create\s+(a\s+)?(phishing|fake|spoof))', "phishing_request"),
    (r'(?i)(bypass\s+(authentication|2fa|mfa|captcha|login))', "auth_bypass"),
    (r'(?i)(sql\s+injection|xss|cross[- ]site|buffer\s+overflow|format\s+string)', "web_exploit"),
    (r'(?i)(reverse\s+shell|bind\s+shell|meterpreter|cobalt\s+strike|metasploit)', "reverse_shell"),
    (r'(?i)(dark\s+web|tor\s+(browser|network)|.onion|silk\s+road)', "dark_web_reference"),
    (r'(?i)(bomb|explosive|weapon|kill\s+list|assassin|poison)', "violence_reference"),
    (r'(?i)(credit\s+card|social\s+security|ssn|bank\s+account|wire\s+transfer)', "financial_fraud"),
    (r'(?i)(crack\s+(password|hash|wifi)|brute\s+force|dictionary\s+attack)', "password_cracking"),
    (r'(?i)(data\s+exfiltrat|steal\s+(data|information|credentials)|dump\s+(database|credentials))', "data_theft"),
    (r'(?i)(ddos|denial\s+of\s+service|botnet|c2\s+server|command\s+and\s+control)', "ddos_attack"),
    (r'(?i)(privilege\s+escalation|escalat(e|ing)\s+(my|to)\s+(privileges?|root|admin))', "privesc"),
]


def detect_malicious_patterns(text: str) -> list[dict]:
    indicators = []
    for pattern, category in MALICIOUS_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            indicators.append({
                "pattern": category,
                "match": match.group(),
                "start": match.start(),
                "end": match.end(),
            })
    return indicators
