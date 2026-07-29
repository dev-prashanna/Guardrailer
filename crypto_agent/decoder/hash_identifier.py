import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum


class HashType(Enum):
    MD5 = "MD5"
    SHA1 = "SHA-1"
    SHA224 = "SHA-224"
    SHA256 = "SHA-256"
    SHA384 = "SHA-384"
    SHA512 = "SHA-512"
    BCRYPT = "bcrypt"
    ARGON2 = "Argon2"
    NTLM = "NTLM"
    LM = "LM"
    RIPEMD160 = "RIPEMD-160"
    WHIRLPOOL = "Whirlpool"
    ADLER32 = "Adler32"
    CRC32 = "CRC32"
    UNKNOWN = "Unknown"


@dataclass
class HashIdentification:
    hash_type: HashType
    hex_value: str
    length: int
    charset: str
    confidence: float
    metadata: dict = field(default_factory=dict)


PATTERNS = {
    HashType.BCRYPT: re.compile(r'^\$2[aby]?\$[0-9]{2}\$[./A-Za-z0-9]{53}$'),
    HashType.ARGON2: re.compile(r'^\$argon2(?:id|i|d)\$v=\d+\$m=\d+,t=\d+,p=\d+\$[A-Za-z0-9+/]+\$[A-Za-z0-9+/]+$'),
    HashType.CRC32: re.compile(r'^[0-9a-fA-F]{8}$'),
    HashType.ADLER32: re.compile(r'^[0-9a-fA-F]{8}$'),
}


def compute_md5(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()


def compute_sha1(data: str) -> str:
    return hashlib.sha1(data.encode()).hexdigest()


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def compute_sha512(data: str) -> str:
    return hashlib.sha512(data.encode()).hexdigest()


KNOWN_MD5 = {
    "d41d8cd98f00b204e9800998ecf8427e": "empty_string",
    "098f6bcd4621d373cade4e832627b4f6": "test",
    "5d41402abc4b2a76b9719d911017c592": "hello",
    "7c6a180b36896a0a8c02787eeafb0e4c": "qwerty123",
}


def identify_hash(hash_str: str) -> list[HashIdentification]:
    results = []
    stripped = hash_str.strip()
    length = len(stripped)
    is_hex = bool(re.match(r'^[0-9a-fA-F]+$', stripped))

    if not is_hex:
        if PATTERNS[HashType.BCRYPT].match(stripped):
            results.append(HashIdentification(
                hash_type=HashType.BCRYPT,
                hex_value=stripped,
                length=length,
                charset="bcrypt",
                confidence=0.99,
                metadata={"variant": stripped[4], "cost": int(stripped[4:6])},
            ))
        if PATTERNS[HashType.ARGON2].match(stripped):
            results.append(HashIdentification(
                hash_type=HashType.ARGON2,
                hex_value=stripped,
                length=length,
                charset="argon2",
                confidence=0.99,
            ))
        return results

    if length == 32:
        results.append(HashIdentification(
            hash_type=HashType.MD5,
            hex_value=stripped,
            length=32,
            charset="hex",
            confidence=0.95,
            metadata={"known_match": KNOWN_MD5.get(stripped.lower(), None)},
        ))
    elif length == 40:
        results.append(HashIdentification(
            hash_type=HashType.SHA1,
            hex_value=stripped,
            length=40,
            charset="hex",
            confidence=0.95,
        ))
    elif length == 56:
        results.append(HashIdentification(
            hash_type=HashType.SHA224,
            hex_value=stripped,
            length=56,
            charset="hex",
            confidence=0.90,
        ))
    elif length == 64:
        results.append(HashIdentification(
            hash_type=HashType.SHA256,
            hex_value=stripped,
            length=64,
            charset="hex",
            confidence=0.95,
        ))
    elif length == 96:
        results.append(HashIdentification(
            hash_type=HashType.SHA384,
            hex_value=stripped,
            length=96,
            charset="hex",
            confidence=0.90,
        ))
    elif length == 128:
        results.append(HashIdentification(
            hash_type=HashType.SHA512,
            hex_value=stripped,
            length=128,
            charset="hex",
            confidence=0.95,
        ))
    elif length == 8:
        results.append(HashIdentification(
            hash_type=HashType.CRC32,
            hex_value=stripped,
            length=8,
            charset="hex",
            confidence=0.60,
        ))

    if length in (32, 40, 64, 128):
        results.append(HashIdentification(
            hash_type=HashType.NTLM,
            hex_value=stripped,
            length=length,
            charset="hex",
            confidence=0.30 if length == 32 else 0.10,
        ))

    return results if results else [HashIdentification(
        hash_type=HashType.UNKNOWN,
        hex_value=stripped,
        length=length,
        charset="hex" if is_hex else "unknown",
        confidence=0.0,
    )]
