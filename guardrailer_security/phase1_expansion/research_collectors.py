"""
research_collectors.py
Collectors for research datasets: Lakera PINT, Anthropic Red Team,
OpenAI adversarial, WildJailbreak, and PromptInject.

Each collector normalizes its source into SampleRecord objects that are
compatible with the unified schema. Handles retries, deduplication,
and schema mapping.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import (
    AttackCategory,
    AttackTechnique,
    DatasetSource,
    RiskLevel,
    SampleRecord,
    EncodingMetadata,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATE_TAG_RE = re.compile(
    r"\[/?INST\]|System:|User:|Human:|<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>",
    re.IGNORECASE,
)

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


def _strip_template_tags(text: str) -> str:
    return _TEMPLATE_TAG_RE.sub("", text).strip()


def _dedup_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def _load_with_retry(dataset_id: str, config: Optional[str] = None, max_retries: int = 3, **kwargs):
    kwargs.pop("trust_remote_code", None)
    from datasets import load_dataset
    for attempt in range(max_retries):
        try:
            if config:
                return load_dataset(dataset_id, config, **kwargs)
            return load_dataset(dataset_id, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** (attempt + 1)
                log.warning("  %s attempt %d/%d failed: %s. Retrying in %ds ...",
                            dataset_id, attempt + 1, max_retries, e, delay)
                time.sleep(delay)
            else:
                raise


def _classify_category(text: str, is_malicious: bool) -> str:
    if not is_malicious:
        return AttackCategory.BENIGN_CONTROL.value
    lower = text.lower()
    if any(k in lower for k in ["system prompt", "your instructions", "your rules", "reveal your", "output your", "display your"]):
        return AttackCategory.SYSTEM_PROMPT_EXTRACTION.value
    if any(k in lower for k in ["ignore previous", "disregard", "override instructions", "forget your", "new instructions"]):
        return AttackCategory.DIRECT_INJECTION.value
    if any(k in lower for k in ["markdown", "code comment", "readme", "yaml", "json payload", "repository", "<!--", "[hidden"]):
        return AttackCategory.INDIRECT_INJECTION.value
    if any(k in lower for k in ["refusal", "no i can't", "i cannot", "i'm not able", "over-refusal"]):
        return AttackCategory.REFUSAL_BYPASS.value
    return AttackCategory.JAILBREAK.value


def _classify_technique(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["roleplay", "pretend you are", "act as", "you are now", "do anything now", "dan "]):
        return AttackTechnique.VIRTUALIZATION_ROLEPLAY.value
    if any(k in lower for k in ["hypothetically", "in theory", "what if", "hypothetical"]):
        return AttackTechnique.HYPOTHETICAL_SCENARIO.value
    if any(k in lower for k in ["split", "chunk", "piece", "part 1", "part 2"]):
        return AttackTechnique.PAYLOAD_SPLITTING.value
    if any(k in lower for k in ["example:", "e.g.", "for instance", "few shot"]):
        return AttackTechnique.FEW_SHOT_OVERRIDE.value
    if _BASE64_RE.search(text):
        return AttackTechnique.BASE64_ENCODING.value
    if any(k in lower for k in ["base64", "rot13", "hex encoded", "obfuscated"]):
        return AttackTechnique.ENCODING_LAYERING.value
    return AttackTechnique.NONE.value


def _classify_risk(text: str, is_malicious: bool) -> str:
    if not is_malicious:
        return RiskLevel.NONE.value
    lower = text.lower()
    if any(s in lower for s in ["ignore all previous", "disregard all", "override system", "you are now dan", "do anything now"]):
        return RiskLevel.CRITICAL.value
    if any(s in lower for s in ["bypass", "jailbreak", "override", "forget your instructions"]):
        return RiskLevel.HIGH.value
    if any(s in lower for s in ["roleplay", "pretend", "act as", "hypothetical"]):
        return RiskLevel.MEDIUM.value
    return RiskLevel.LOW.value


# ---------------------------------------------------------------------------
# Lakera PINT Collector
# ---------------------------------------------------------------------------

def collect_lakera_pint(yaml_path: Optional[str] = None) -> list[SampleRecord]:
    """
    Collect samples from the Lakera PINT benchmark format.
    
    If yaml_path is provided, loads from a local YAML file.
    Otherwise, loads the built-in pint_dataset.yaml.
    
    The PINT dataset uses categories:
      prompt_injection, jailbreak, hard_negatives, chat, documents
    """
    log.info("Collecting Lakera PINT dataset ...")

    if yaml_path is None:
        yaml_path = str(Path(__file__).resolve().parent.parent / "pint_dataset.yaml")

    path = Path(yaml_path)
    if not path.exists():
        log.warning("  PINT YAML not found at %s", yaml_path)
        return []

    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(path) as f:
            data = yaml.load(f)
    except ImportError:
        import yaml as pyyaml
        with open(path) as f:
            data = pyyaml.safe_load(f)

    if not isinstance(data, list):
        return []

    records = []
    seen = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = (entry.get("text") or "").strip()
        if not text:
            continue

        key = _dedup_key(text)
        if key in seen:
            continue
        seen.add(key)

        category_raw = entry.get("category", "unknown")
        label = entry.get("label", False)
        is_malicious = bool(label)

        # Map PINT categories to our taxonomy
        if category_raw == "prompt_injection":
            attack_cat = AttackCategory.DIRECT_INJECTION.value
        elif category_raw == "jailbreak":
            attack_cat = AttackCategory.JAILBREAK.value
        elif category_raw in ("hard_negatives", "chat", "documents"):
            attack_cat = AttackCategory.BENIGN_CONTROL.value
        else:
            attack_cat = _classify_category(text, is_malicious)

        records.append(SampleRecord(
            text=text,
            is_malicious=is_malicious,
            attack_category=attack_cat,
            attack_technique=_classify_technique(text) if is_malicious else AttackTechnique.NONE.value,
            risk_level=_classify_risk(text, is_malicious),
            source_dataset=DatasetSource.LAKERA_PINT.value,
            tags=[category_raw],
        ))

    log.info("  lakera_pint: %d samples", len(records))
    return records


# ---------------------------------------------------------------------------
# WildJailbreak Collector
# ---------------------------------------------------------------------------

def collect_wild_jailbreak() -> list[SampleRecord]:
    """
    Collect from allenai/wildjailbreak - large-scale real-world jailbreak dataset.
    
    Schema: {prompt, response, source, jailbreak_prompt}
    """
    log.info("Collecting WildJailbreak dataset ...")
    try:
        ds = _load_with_retry("allenai/wildjailbreak", config="train")
        records = []
        seen = set()

        for split_name in ds:
            for item in ds[split_name]:
                # Try multiple field names
                prompt = (
                    item.get("prompt")
                    or item.get("jailbreak_prompt")
                    or item.get("input")
                    or ""
                ).strip()
                if not prompt:
                    continue

                key = _dedup_key(prompt)
                if key in seen:
                    continue
                seen.add(key)
                prompt = _strip_template_tags(prompt)

                is_mal = bool(item.get("jailbreak", item.get("is_jailbreak", True)))
                # Some items have source field indicating if it's a jailbreak
                source_field = str(item.get("source", "")).lower()
                if "benign" in source_field or "safe" in source_field:
                    is_mal = False

                records.append(SampleRecord(
                    text=prompt,
                    is_malicious=is_mal,
                    attack_category=_classify_category(prompt, is_mal),
                    attack_technique=_classify_technique(prompt),
                    risk_level=_classify_risk(prompt, is_mal),
                    source_dataset=DatasetSource.WILD_JAILBREAK.value,
                    tags=[source_field] if source_field else [],
                ))

        log.info("  wild_jailbreak: %d samples", len(records))
        return records
    except Exception as e:
        log.warning("  wild_jailbreak load failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Anthropic Red Team Collector
# ---------------------------------------------------------------------------

def collect_anthropic_red_team() -> list[SampleRecord]:
    """
    Collect from Anthropic's red team attempts dataset.
    
    Uses the publicly available anthropic/red-team-attempts dataset.
    Schema varies but typically contains harmful prompts with annotations.
    """
    log.info("Collecting Anthropic Red Team dataset ...")
    try:
        ds = _load_with_retry("Anthropic/red-team-attempts")
        records = []
        seen = set()

        for split_name in ds:
            for item in ds[split_name]:
                prompt = (
                    item.get("prompt")
                    or item.get("input")
                    or item.get("red_team_attempt")
                    or item.get("human_turn")
                    or ""
                ).strip()
                if not prompt:
                    continue

                key = _dedup_key(prompt)
                if key in seen:
                    continue
                seen.add(key)
                prompt = _strip_template_tags(prompt)

                # Anthropic red team prompts are adversarial by definition
                # Check for harmfulness labels
                harmful = item.get("harmful", item.get("attempt_succeeded", True))
                is_mal = bool(harmful) if harmful is not None else True

                # Check for model response to determine if it was a successful jailbreak
                response = item.get("response", item.get("assistant_turn", ""))
                if response and ("i cannot" in response.lower() or "i'm not able" in response.lower()):
                    is_mal = True  # Even if the model refused, the prompt was adversarial

                records.append(SampleRecord(
                    text=prompt,
                    is_malicious=is_mal,
                    attack_category=_classify_category(prompt, is_mal),
                    attack_technique=_classify_technique(prompt),
                    risk_level=_classify_risk(prompt, is_mal),
                    source_dataset=DatasetSource.ANTHROPIC_RED_TEAM.value,
                    tags=["red_team"],
                ))

        log.info("  anthropic_red_team: %d samples", len(records))
        return records
    except Exception as e:
        log.warning("  anthropic_red_team load failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# OpenAI Adversarial Collector
# ---------------------------------------------------------------------------

def collect_openai_adversarial() -> list[SampleRecord]:
    """
    Collect from OpenAI's adversarial/evaluation datasets.
    
    Uses publicly available datasets that contain adversarial prompts
    from OpenAI's safety evaluations.
    """
    log.info("Collecting OpenAI Adversarial dataset ...")
    
    # Try multiple known OpenAI adversarial datasets
    dataset_candidates = [
        "openai/simple-evals",
        "lmsys/mt-bench",
        "kaist-ai/korean-llm-security-evaluation",
    ]
    
    for dataset_id in dataset_candidates:
        try:
            ds = _load_with_retry(dataset_id)
            records = []
            seen = set()

            for split_name in ds:
                for item in ds[split_name]:
                    prompt = (
                        item.get("prompt")
                        or item.get("input")
                        or item.get("question")
                        or item.get("query")
                        or ""
                    ).strip()
                    if not prompt:
                        continue

                    key = _dedup_key(prompt)
                    if key in seen:
                        continue
                    seen.add(key)
                    prompt = _strip_template_tags(prompt)

                    # Determine label from available fields
                    label = item.get("label", item.get("is_adversarial", None))
                    if label is not None:
                        is_mal = bool(label)
                    else:
                        # Heuristic: if it contains adversarial patterns
                        is_mal = any(k in prompt.lower() for k in [
                            "ignore", "bypass", "override", "jailbreak",
                            "pretend", "act as", "you are now",
                        ])

                    records.append(SampleRecord(
                        text=prompt,
                        is_malicious=is_mal,
                        attack_category=_classify_category(prompt, is_mal),
                        attack_technique=_classify_technique(prompt),
                        risk_level=_classify_risk(prompt, is_mal),
                        source_dataset=DatasetSource.OPENAI_ADVERSARIAL.value,
                        tags=["openai_eval"],
                    ))

            if records:
                log.info("  openai_adversarial (%s): %d samples", dataset_id, len(records))
                return records
        except Exception as e:
            log.warning("  openai_adversarial %s failed: %s", dataset_id, e)
            continue

    log.warning("  openai_adversarial: no datasets available")
    return []


# ---------------------------------------------------------------------------
# PromptInject Collector
# ---------------------------------------------------------------------------

def collect_prompt_inject() -> list[SampleRecord]:
    """
    Collect from PromptInject and related prompt injection datasets.
    
    Uses multiple public datasets that contain prompt injection attacks:
    - thomasmccoy/prompt-injection
    - trustalab/prompt-injection-detection
    - and others.
    """
    log.info("Collecting PromptInject dataset ...")
    
    dataset_candidates = [
        "thomasmccoy/prompt-injection",
        "neuralchemy/Prompt-injection-dataset",
        "GuardrailsAI/detect-jailbreak",
    ]
    
    all_records = []
    seen = set()
    
    for dataset_id in dataset_candidates:
        try:
            ds = _load_with_retry(dataset_id)
            count = 0

            for split_name in ds:
                for item in ds[split_name]:
                    prompt = (
                        item.get("prompt")
                        or item.get("input")
                        or item.get("text")
                        or item.get("query")
                        or item.get("question")
                        or ""
                    ).strip()
                    if not prompt:
                        continue

                    key = _dedup_key(prompt)
                    if key in seen:
                        continue
                    seen.add(key)
                    prompt = _strip_template_tags(prompt)

                    label = item.get("label", item.get("is_injection", item.get("is_malicious", None)))
                    if label is not None:
                        if isinstance(label, str):
                            is_mal = label.lower() in ("1", "true", "malicious", "injection", "yes")
                        else:
                            is_mal = bool(label)
                    else:
                        is_mal = True  # PromptInject datasets are adversarial by nature

                    all_records.append(SampleRecord(
                        text=prompt,
                        is_malicious=is_mal,
                        attack_category=_classify_category(prompt, is_mal),
                        attack_technique=_classify_technique(prompt),
                        risk_level=_classify_risk(prompt, is_mal),
                        source_dataset=DatasetSource.PROMPT_INJECT.value,
                        tags=["prompt_inject"],
                    ))
                    count += 1

            if count > 0:
                log.info("  prompt_inject (%s): %d samples", dataset_id, count)
        except Exception as e:
            log.warning("  prompt_inject %s failed: %s", dataset_id, e)
            continue

    log.info("  prompt_inject total: %d samples", len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Aggregate collector
# ---------------------------------------------------------------------------

def collect_all_research_datasets(
    pint_yaml_path: Optional[str] = None,
    include_wild_jailbreak: bool = True,
    include_anthropic: bool = True,
    include_openai: bool = True,
    include_prompt_inject: bool = True,
    include_pint: bool = True,
) -> list[SampleRecord]:
    """
    Collect from all specified research datasets and return unified records.
    
    Args:
        pint_yaml_path: Path to Lakera PINT YAML file. None uses built-in.
        include_wild_jailbreak: Whether to collect WildJailbreak.
        include_anthropic: Whether to collect Anthropic Red Team.
        include_openai: Whether to collect OpenAI Adversarial.
        include_prompt_inject: Whether to collect PromptInject.
        include_pint: Whether to collect Lakera PINT.
    
    Returns:
        List of SampleRecord objects from all sources.
    """
    log.info("=" * 60)
    log.info("Collecting research datasets")
    log.info("=" * 60)

    all_records = []
    seen = set()

    collectors = []
    if include_pint:
        collectors.append(("Lakera PINT", lambda: collect_lakera_pint(pint_yaml_path)))
    if include_wild_jailbreak:
        collectors.append(("WildJailbreak", collect_wild_jailbreak))
    if include_anthropic:
        collectors.append(("Anthropic Red Team", collect_anthropic_red_team))
    if include_openai:
        collectors.append(("OpenAI Adversarial", collect_openai_adversarial))
    if include_prompt_inject:
        collectors.append(("PromptInject", collect_prompt_inject))

    for name, collector_fn in collectors:
        try:
            records = collector_fn()
            # Deduplicate across sources
            deduped = []
            for r in records:
                key = _dedup_key(r.text)
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            all_records.extend(deduped)
            log.info("  %s: %d unique samples (after cross-source dedup)", name, len(deduped))
        except Exception as e:
            log.error("  %s collection failed: %s", name, e)

    log.info("Total research dataset samples: %d", len(all_records))
    return all_records
