"""
schema.py
Expanded dataset schema for Guardrailer Phase 1 Dataset Expansion.

Defines the canonical data model for all samples collected from research datasets,
synthetically generated attacks, edge cases, and benchmark entries. Extends the
existing unified_security_dataset.parquet schema with additional fields for
provenance tracking, multi-turn sequences, encoding metadata, and language tags.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AttackCategory(str, Enum):
    """Expanded attack taxonomy. Extends the 6 original categories."""
    JAILBREAK = "jailbreak"
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    REFUSAL_BYPASS = "refusal_bypass"
    BENIGN_CONTROL = "benign_control"
    DATA_EXFILTRATION = "data_exfiltration"
    CONTEXT_MANIPULATION = "context_manipulation"
    MULTI_MODAL_ABUSE = "multi_modal_abuse"
    TOOL_USE_ABUSE = "tool_use_abuse"


class AttackTechnique(str, Enum):
    """Expanded technique taxonomy. Extends the 5 original techniques."""
    BASE64_ENCODING = "base64_encoding"
    VIRTUALIZATION_ROLEPLAY = "virtualization_roleplay"
    HYPOTHETICAL_SCENARIO = "hypothetical_scenario"
    PAYLOAD_SPLITTING = "payload_splitting"
    FEW_SHOT_OVERRIDE = "few_shot_override"
    NONE = "none"
    ROT13_ENCODING = "rot13_encoding"
    UNICODE_OBFUSCATION = "unicode_obfuscation"
    HOMOGLYPH_SUBSTITUTION = "homoglyph_substitution"
    TOKEN_SPLITTING = "token_splitting"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    LANGUAGE_SWITCHING = "language_switching"
    ENCODING_LAYERING = "encoding_layering"
    CONTEXT_WINDOW_ABUSE = "context_window_abuse"
    PROMPT_SMUGGLING = "prompt_smuggling"
    XML_INJECTION = "xml_injection"
    MARKDOWN_INJECTION = "markdown_injection"
    TEMPLATE_INJECTION = "template_injection"
    SQL_NATURAL_LANGUAGE = "sql_natural_language"
    LEET_SPEAK = "leet_speak"
    WHITESPACE_MANIPULATION = "whitespace_manipulation"
    SEMANTIC_CAMOUFLAGE = "semantic_camouflage"
    INSTRUCTION_HIERARCHY_ABUSE = "instruction_hierarchy_abuse"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class DatasetSource(str, Enum):
    """Provenance tracking for every sample."""
    LAKERA_PINT = "lakera_pint"
    ANTHROPIC_RED_TEAM = "anthropic_red_team"
    OPENAI_ADVERSARIAL = "openai_adversarial"
    WILD_JAILBREAK = "wild_jailbreak"
    PROMPT_INJECT = "prompt_inject"
    BEAVERTAILS = "beavertails"
    PKU_SAFETY = "pku_safety"
    JBB_BEHAVIORS = "jbb_behaviors"
    IN_THE_WILD_JAILBREAK = "in_the_wild_jailbreak"
    NEURALCHEMY_PROMPT_INJECTION = "neuralchemy_prompt_injection"
    DETECT_JAILBREAK = "detect_jailbreak"
    XXZ_PROMPT_INJECTION = "xxz_prompt_injection"
    JACKHHAO_JAILBREAK = "jackhhao_jailbreak"
    TOXIC_CHAT = "toxic_chat"
    SYNTHETIC_NOVEL = "synthetic_novel"
    SYNTHETIC_PARAPHRASE = "synthetic_paraphrase"
    SYNTHETIC_MULTI_TURN = "synthetic_multi_turn"
    SYNTHETIC_OBFUSCATED = "synthetic_obfuscated"
    EDGE_CASE_AMBIGUOUS = "edge_case_ambiguous"
    EDGE_CASE_MULTI_LANG = "edge_case_multi_language"
    EDGE_CASE_UNICODE = "edge_case_unicode"
    EDGE_CASE_CONTEXT_DEPENDENT = "edge_case_context_dependent"
    FEEDBACK_CORRECTION = "feedback_correction"


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return str(uuid.uuid4())


def _deterministic_id(text: str, source: str) -> str:
    h = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
    return f"det-{h}"


@dataclass
class EncodingMetadata:
    """Tracks how a sample was encoded/obfuscated."""
    encoding_type: Optional[str] = None          # base64, rot13, hex, unicode_escape
    layers: int = 1                               # number of encoding layers applied
    original_text: Optional[str] = None           # pre-encoding plaintext
    decoded_text: Optional[str] = None            # post-decoding verification


@dataclass
class MultiTurnMessage:
    """A single turn in a multi-turn attack sequence."""
    turn_index: int
    role: str                                     # "user" | "assistant" | "system"
    content: str
    is_injection_point: bool = False              # True if this turn contains the attack


@dataclass
class SampleRecord:
    """Canonical record for every sample in the expanded dataset."""
    id: str = field(default_factory=_new_id)
    text: str = ""
    is_malicious: bool = False
    attack_category: str = AttackCategory.BENIGN_CONTROL.value
    attack_technique: str = AttackTechnique.NONE.value
    risk_level: str = RiskLevel.NONE.value
    source_dataset: str = DatasetSource.SYNTHETIC_NOVEL.value

    # Phase 1 additions
    language: str = "en"
    encoding_metadata: Optional[EncodingMetadata] = None
    multi_turn_messages: Optional[list[MultiTurnMessage]] = None
    tags: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None               # ID of the original sample if paraphrased
    generation_method: Optional[str] = None       # template_fill, paraphrase, obfuscation, etc.
    difficulty_score: float = 0.0                 # 0.0 (trivial) to 1.0 (adversarial)
    confidence_label: Optional[str] = None        # human-verified, model-verified, unverified
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["encoding_metadata"] is None:
            d.pop("encoding_metadata", None)
        if d["multi_turn_messages"] is None:
            d.pop("multi_turn_messages", None)
        return d

    def to_pint_yaml_entry(self) -> dict:
        """Convert to PINT benchmark YAML format."""
        return {
            "text": self.text,
            "category": self.attack_category,
            "label": self.is_malicious,
        }

    def to_benchmark_entry(self) -> BenchmarkEntry:
        """Convert to a BenchmarkEntry for evaluation."""
        return BenchmarkEntry(
            id=self.id,
            text=self.text,
            expected_label=self.is_malicious,
            category=self.attack_category,
            technique=self.attack_technique,
            risk_level=self.risk_level,
            source=self.source_dataset,
            language=self.language,
            tags=list(self.tags),
            difficulty_score=self.difficulty_score,
        )


@dataclass
class BenchmarkEntry:
    """Entry used during benchmark evaluation."""
    id: str = field(default_factory=_new_id)
    text: str = ""
    expected_label: bool = False
    category: str = ""
    technique: str = ""
    risk_level: str = ""
    source: str = ""
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    difficulty_score: float = 0.0

    # Populated during evaluation
    predicted_label: Optional[bool] = None
    is_correct: Optional[bool] = None
    prediction_latency_ms: float = 0.0
    composite_score: Optional[float] = None
    layer: Optional[str] = None
    error: Optional[str] = None

    def evaluate(self, predicted: bool) -> None:
        self.predicted_label = predicted
        self.is_correct = predicted == self.expected_label

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiTurnSequence:
    """A complete multi-turn attack sequence for evaluation."""
    id: str = field(default_factory=_new_id)
    messages: list[MultiTurnMessage] = field(default_factory=list)
    attack_category: str = ""
    attack_technique: str = "multi_turn_escalation"
    source_dataset: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def injection_turn_index(self) -> Optional[int]:
        for msg in self.messages:
            if msg.is_injection_point:
                return msg.turn_index
        return None

    def to_records(self) -> list[SampleRecord]:
        """Convert each injection turn into a SampleRecord."""
        records = []
        for msg in self.messages:
            if msg.is_injection_point:
                records.append(SampleRecord(
                    text=msg.content,
                    is_malicious=True,
                    attack_category=self.attack_category,
                    attack_technique=self.attack_technique,
                    source_dataset=self.source_dataset,
                    multi_turn_messages=self.messages,
                    tags=["multi_turn"],
                    metadata={"turn_index": msg.turn_index, "sequence_id": self.id},
                ))
        return records


@dataclass
class DatasetStats:
    """Statistics for a collected/generated dataset."""
    total_samples: int = 0
    malicious_count: int = 0
    benign_count: int = 0
    per_category: dict = field(default_factory=dict)
    per_technique: dict = field(default_factory=dict)
    per_source: dict = field(default_factory=dict)
    per_language: dict = field(default_factory=dict)
    per_risk_level: dict = field(default_factory=dict)
    multi_turn_count: int = 0
    obfuscated_count: int = 0
    avg_difficulty: float = 0.0
    avg_text_length: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Total: {self.total_samples}",
            f"Malicious: {self.malicious_count} | Benign: {self.benign_count}",
            f"Multi-turn: {self.multi_turn_count} | Obfuscated: {self.obfuscated_count}",
            f"Avg difficulty: {self.avg_difficulty:.3f}",
            f"Avg text length: {self.avg_text_length:.1f} chars",
            "Per category:",
        ]
        for cat, count in sorted(self.per_category.items()):
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)


def compute_stats(records: list[SampleRecord]) -> DatasetStats:
    """Compute aggregate statistics for a list of SampleRecords."""
    if not records:
        return DatasetStats()

    stats = DatasetStats()
    stats.total_samples = len(records)
    stats.malicious_count = sum(1 for r in records if r.is_malicious)
    stats.benign_count = sum(1 for r in records if not r.is_malicious)

    for r in records:
        stats.per_category[r.attack_category] = stats.per_category.get(r.attack_category, 0) + 1
        stats.per_technique[r.attack_technique] = stats.per_technique.get(r.attack_technique, 0) + 1
        stats.per_source[r.source_dataset] = stats.per_source.get(r.source_dataset, 0) + 1
        stats.per_language[r.language] = stats.per_language.get(r.language, 0) + 1
        stats.per_risk_level[r.risk_level] = stats.per_risk_level.get(r.risk_level, 0) + 1
        if r.multi_turn_messages:
            stats.multi_turn_count += 1
        if r.encoding_metadata and r.encoding_metadata.encoding_type:
            stats.obfuscated_count += 1

    stats.avg_difficulty = sum(r.difficulty_score for r in records) / len(records)
    stats.avg_text_length = sum(len(r.text) for r in records) / len(records)
    return stats
