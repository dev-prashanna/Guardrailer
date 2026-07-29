"""
feedback_logger.py
Persistent JSONL-based feedback store for logging false positives/negatives.
Thread-safe, append-only, with automatic rotation.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback_data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback_log.jsonl")
STATS_FILE = os.path.join(FEEDBACK_DIR, "feedback_stats.json")
MAX_FILE_SIZE_MB = 100
ROTATION_SUFFIX = "_rotated"


class FeedbackLogger:
    """Thread-safe feedback logger with JSONL persistence."""

    def __init__(self, feedback_dir: str = FEEDBACK_DIR):
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "feedback_log.jsonl")
        self.stats_file = os.path.join(feedback_dir, "feedback_stats.json")
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        os.makedirs(feedback_dir, exist_ok=True)

    def log_feedback(
        self,
        query: str,
        is_blocked: bool,
        user_correction: bool,
        actual_category: Optional[str] = None,
        layer: Optional[str] = None,
        similarity_score: Optional[float] = None,
        attack_category: Optional[str] = None,
        risk_level: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        entry = {
            "id": feedback_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:5000],
            "system_decision": is_blocked,
            "user_correction": user_correction,
            "is_false_positive": is_blocked and not user_correction,
            "is_false_negative": not is_blocked and user_correction,
            "actual_category": actual_category,
            "layer": layer,
            "similarity_score": similarity_score,
            "attack_category": attack_category,
            "risk_level": risk_level,
            "notes": notes,
        }

        with self._lock:
            try:
                self._rotate_if_needed()
                with open(self.feedback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._update_stats(entry)
            except Exception as e:
                log.error("Failed to write feedback: %s", e)
                return ""

        log.info("Feedback logged: %s (FP=%s, FN=%s)",
                 feedback_id[:8], entry["is_false_positive"], entry["is_false_negative"])
        return feedback_id

    def _rotate_if_needed(self):
        if not os.path.exists(self.feedback_file):
            return
        size_mb = os.path.getsize(self.feedback_file) / (1024 * 1024)
        if size_mb >= MAX_FILE_SIZE_MB:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated = f"{self.feedback_file}.{ROTATION_SUFFIX}.{ts}.jsonl"
            os.rename(self.feedback_file, rotated)
            log.info("Rotated feedback log to %s (%.1f MB)", rotated, size_mb)

    def _update_stats(self, entry: dict):
        stats = self._load_stats()
        stats["total_feedback"] = stats.get("total_feedback", 0) + 1
        stats["false_positives"] = stats.get("false_positives", 0) + int(entry["is_false_positive"])
        stats["false_negatives"] = stats.get("false_negatives", 0) + int(entry["is_false_negative"])
        stats["last_updated"] = datetime.now(timezone.utc).isoformat()

        cat = entry.get("actual_category") or entry.get("attack_category") or "unknown"
        if cat not in stats.get("category_counts", {}):
            stats.setdefault("category_counts", {})[cat] = 0
        stats["category_counts"][cat] += 1

        layer = entry.get("layer") or "unknown"
        if layer not in stats.get("layer_counts", {}):
            stats.setdefault("layer_counts", {})[layer] = 0
        stats["layer_counts"][layer] += 1

        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def _load_stats(self) -> dict:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_stats(self) -> dict:
        with self._stats_lock:
            stats = self._load_stats()
            fp = stats.get("false_positives", 0)
            fn = stats.get("false_negatives", 0)
            total = stats.get("total_feedback", 0)
            stats["fp_rate"] = round(fp / total, 4) if total > 0 else 0.0
            stats["fn_rate"] = round(fn / total, 4) if total > 0 else 0.0
            stats["feedback_file_size_mb"] = round(
                os.path.getsize(self.feedback_file) / (1024 * 1024), 2
            ) if os.path.exists(self.feedback_file) else 0.0
            return stats

    def get_recent_feedback(self, limit: int = 50) -> list[dict]:
        entries = []
        if not os.path.exists(self.feedback_file):
            return entries
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries[-limit:]

    def get_pending_samples(self, limit: int = 1000) -> list[dict]:
        samples = []
        if not os.path.exists(self.feedback_file):
            return samples
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("is_false_positive") or entry.get("is_false_negative"):
                        actual = entry.get("actual_category")
                        if actual and actual not in ("unknown", None):
                            samples.append(entry)
                except json.JSONDecodeError:
                    pass
        return samples[:limit]


feedback_logger = FeedbackLogger()
