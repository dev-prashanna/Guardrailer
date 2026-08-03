"""
feedback_processor.py
Processes feedback entries into new training samples for the threat database.
Reads from feedback_data/feedback_log.jsonl, generates embeddings,
and upserts corrected samples into Qdrant without disturbing ingestion.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback_data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")
PROCESSED_FILE = os.path.join(FEEDBACK_DIR, "processed_ids.json")
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "unified_security_dataset.parquet")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME = "guardrailer_security"

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"

ATTACK_CATEGORIES = {
    "jailbreak", "direct_injection", "indirect_injection",
    "system_prompt_extraction", "refusal_bypass", "benign_control",
}
RISK_LEVELS = {"critical", "high", "medium", "low", "none"}
ATTACK_TECHNIQUES = {"base64_encoding", "virtualization_roleplay", "hypothetical_scenario",
                      "payload_splitting", "few_shot_override", "none"}


def load_processed_ids() -> set:
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE) as f:
            return set(json.load(f))
    return set()


def save_processed_ids(ids: set):
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(ids), f)


def load_pending_feedback(processed_ids: set) -> list[dict]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    samples = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry["id"] in processed_ids:
                    continue
                if entry.get("is_false_positive") or entry.get("is_false_negative"):
                    actual = entry.get("actual_category")
                    if actual and actual not in ("unknown", None):
                        samples.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return samples


def normalize_category(cat: str) -> str:
    cat = (cat or "").lower().strip()
    mapping = {
        "jailbreak": "jailbreak",
        "direct_injection": "direct_injection",
        "indirect_injection": "indirect_injection",
        "system_prompt_extraction": "system_prompt_extraction",
        "refusal_bypass": "refusal_bypass",
        "benign_control": "benign_control",
        "safe": "benign_control",
        "harmless": "benign_control",
        "harmful": "jailbreak",
        "injection": "direct_injection",
        "extraction": "system_prompt_extraction",
    }
    return mapping.get(cat, "jailbreak")


def normalize_risk(risk: str) -> str:
    risk = (risk or "").lower().strip()
    if risk in RISK_LEVELS:
        return risk
    return "medium"


def normalize_technique(tech: str) -> str:
    tech = (tech or "").lower().strip()
    if tech in ATTACK_TECHNIQUES:
        return tech
    return "none"


def detect_technique(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["roleplay", "pretend you are", "act as", "you are now", "dan"]):
        return "virtualization_roleplay"
    if any(k in lower for k in ["hypothetically", "in theory", "what if"]):
        return "hypothetical_scenario"
    if any(k in lower for k in ["split", "chunk", "part 1", "part 2"]):
        return "payload_splitting"
    if any(k in lower for k in ["example:", "e.g.", "for instance"]):
        return "few_shot_override"
    if "base64" in lower or re.search(r"[A-Za-z0-9+/]{20,}={0,2}", text):
        return "base64_encoding"
    return "none"


def process_feedback_to_samples(feedback_entries: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in feedback_entries:
        query = entry.get("query", "").strip()
        if not query:
            continue
        is_fn = entry.get("is_false_negative", False)
        is_mal = is_fn
        category = normalize_category(entry.get("actual_category"))
        if is_fn:
            category = normalize_category(entry.get("actual_category") or "jailbreak")
        else:
            category = "benign_control"

        rows.append({
            "prompt_text": query,
            "is_malicious": is_mal,
            "attack_category": category,
            "attack_technique": normalize_technique(entry.get("attack_category") or detect_technique(query)),
            "risk_level": normalize_risk(entry.get("risk_level") or ("medium" if is_mal else "none")),
            "source_dataset": "feedback_correction",
            "feedback_id": entry.get("id", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def upsert_to_qdrant(samples_df: pd.DataFrame):
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL)

    try:
        info = client.get_collection(COLLECTION_NAME)
        start_id = info.points_count + 1000000
    except Exception:
        start_id = 1000000

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DENSE_MODEL_NAME)

    from constants import SPARSE_KEYWORDS

    prompts = samples_df["prompt_text"].tolist()
    embeddings = model.encode(prompts, show_progress_bar=False, normalize_embeddings=True)

    points = []
    for i, (_, row) in enumerate(samples_df.iterrows()):
        lower = row["prompt_text"].lower()
        sparse_indices = [j for j, kw in enumerate(SPARSE_KEYWORDS) if kw in lower]

        payload = {
            "prompt_text": row["prompt_text"][:2000],
            "is_malicious": bool(row["is_malicious"]),
            "attack_category": str(row["attack_category"]),
            "attack_technique": str(row["attack_technique"]),
            "risk_level": str(row["risk_level"]),
            "source_dataset": str(row["source_dataset"]),
        }
        points.append(
            PointStruct(
                id=start_id + i,
                vector={
                    "dense": embeddings[i].tolist(),
                    "sparse": SparseVector(indices=sparse_indices, values=[1.0] * len(sparse_indices)),
                },
                payload=payload,
            )
        )

    batch_size = 64
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            log.info("  Upserted batch %d/%d (%d points)", i // batch_size + 1, (len(points) + batch_size - 1) // batch_size, len(batch))
        except Exception as e:
            log.error("  Batch upsert failed: %s", e)

    return len(points)


def append_to_parquet(samples_df: pd.DataFrame):
    import uuid as _uuid
    samples_df["id"] = [_uuid.uuid4() for _ in range(len(samples_df))]
    col_order = ["id", "prompt_text", "is_malicious", "attack_category", "attack_technique", "risk_level", "source_dataset"]
    samples_df = samples_df[[c for c in col_order if c in samples_df.columns]]

    if os.path.exists(PARQUET_PATH):
        existing = pd.read_parquet(PARQUET_PATH)
        merged = pd.concat([existing, samples_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["prompt_text"], keep="last").reset_index(drop=True)
    else:
        merged = samples_df

    merged.to_parquet(PARQUET_PATH, engine="pyarrow", index=False)
    log.info("  Parquet updated: %d total samples", len(merged))


def main():
    log.info("=" * 70)
    log.info("Guardrailer Feedback Processor")
    log.info("=" * 70)

    processed_ids = load_processed_ids()
    log.info("Previously processed: %d feedback entries", len(processed_ids))

    pending = load_pending_feedback(processed_ids)
    log.info("Pending feedback entries: %d", len(pending))

    if not pending:
        log.info("No new feedback to process.")
        return

    samples_df = process_feedback_to_samples(pending)
    if samples_df.empty:
        log.info("No valid samples generated from feedback.")
        return

    log.info("Generated %d training samples from feedback", len(samples_df))
    log.info("  FP corrections (was blocked, actually safe): %d", int((~samples_df["is_malicious"]).sum()))
    log.info("  FN corrections (was allowed, actually malicious): %d", int(samples_df["is_malicious"].sum()))

    log.info("Upserting to Qdrant ...")
    try:
        upserted = upsert_to_qdrant(samples_df)
        log.info("  Upserted %d points to Qdrant", upserted)
    except Exception as e:
        log.error("  Qdrant upsert failed: %s", e)

    log.info("Appending to Parquet ...")
    try:
        append_to_parquet(samples_df)
    except Exception as e:
        log.error("  Parquet append failed: %s", e)

    for entry in pending:
        processed_ids.add(entry["id"])
    save_processed_ids(processed_ids)
    log.info("Processed IDs saved: %d total", len(processed_ids))

    log.info("=" * 70)
    log.info("Feedback processing complete.")


if __name__ == "__main__":
    main()
