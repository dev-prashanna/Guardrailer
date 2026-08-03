"""
phase4_prototype.py
Phase 4: Near-Term Architecture Enhancements

Implements three key features for conversational AI security:
1. Streaming evaluation - Real-time prompt evaluation for conversational agents
2. Multi-turn context - Evaluate conversation history, not just single prompts
3. Semantic hash index - Exact-match deduplication for known attack payloads

This module extends the existing security engine with:
- WebSocket support for streaming evaluation
- Conversation session management with context accumulation
- MinHash/LSH-based semantic hashing for fast deduplication
- Incremental scoring with context-aware signals
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHASE4_VERSION = "4.0.0"
SEMANTIC_HASH_STORAGE = Path(__file__).resolve().parent / "semantic_hash_index.json"
CONVERSATION_STORAGE = Path(__file__).resolve().parent / "conversation_sessions.json"

# Streaming configuration
STREAM_CHUNK_SIZE = 1024  # characters per chunk
STREAM_EVAL_INTERVAL = 0.1  # seconds between evaluations
STREAM_CONFIDENCE_THRESHOLD = 0.85  # confidence for early termination

# Multi-turn context configuration
MAX_CONTEXT_TURNS = 10  # maximum conversation turns to consider
CONTEXT_DECAY_FACTOR = 0.9  # exponential decay for older turns
CONTEXT_AGGREGATION_MODE = "weighted"  # weighted, concatenated, or attention

# Semantic hash configuration
HASH_SIZE = 128  # MinHash signature size
HASH_PERMUTATIONS = 256  # number of permutations for LSH
HASH_BANDS = 32  # number of bands for LSH
HASH_ROWS = 4  # rows per band (HASH_SIZE / HASH_BANDS)
HASH_THRESHOLD = 0.7  # similarity threshold for hash matching

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class StreamingState(str, Enum):
    """State of a streaming evaluation session."""
    IDLE = "idle"
    STREAMING = "streaming"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class AggregationMode(str, Enum):
    """How to aggregate multi-turn context."""
    WEIGHTED = "weighted"  # exponential decay weighting
    CONCATENATED = "concatenated"  # concatenate all turns
    ATTENTION = "attention"  # attention-weighted aggregation


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    role: str  # user or assistant
    content: str
    timestamp: float
    embedding: Optional[np.ndarray] = None
    evaluation_result: Optional[dict] = None
    risk_score: float = 0.0


@dataclass
class ConversationSession:
    """Manages multi-turn conversation context."""
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    aggregate_risk: float = 0.0
    is_blocked: bool = False
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamingChunk:
    """A chunk of streaming input."""
    chunk_id: int
    content: str
    is_final: bool
    timestamp: float
    session_id: Optional[str] = None


@dataclass
class SemanticHashEntry:
    """Entry in the semantic hash index."""
    hash_id: str
    original_text: str
    hash_signature: list[int]
    attack_category: str
    risk_level: str
    created_at: float
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamingEvaluationResult:
    """Result of a streaming evaluation."""
    chunk_id: int
    accumulated_text: str
    is_blocked: bool
    risk_score: float
    attack_category: Optional[str] = None
    attack_technique: Optional[str] = None
    risk_level: Optional[str] = None
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    hash_match: bool = False
    hash_match_id: Optional[str] = None


# ---------------------------------------------------------------------------
# MinHash Implementation for Semantic Hashing
# ---------------------------------------------------------------------------

class MinHash:
    """MinHash signature for semantic similarity."""
    
    def __init__(self, num_perm: int = HASH_PERMUTATIONS):
        self.num_perm = num_perm
        self.hash_func = hashlib.md5
        self.max_hash = (1 << 32) - 1
    
    def _hash_func(self, x: int, seed: int) -> int:
        """Consistent hash function for MinHash."""
        h = self.hash_func(f"{seed}:{x}".encode()).digest()
        return int.from_bytes(h[:4], byteorder='big')
    
    def compute_signature(self, shingles: set[int]) -> list[int]:
        """Compute MinHash signature from shingle set."""
        signature = []
        for i in range(self.num_perm):
            min_hash = self.max_hash
            for shingle in shingles:
                h = self._hash_func(shingle, i)
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature
    
    @staticmethod
    def text_to_shingles(text: str, k: int = 3) -> set[int]:
        """Convert text to k-gram shingles."""
        text = text.lower().strip()
        if len(text) < k:
            return {hash(text)}
        
        shingles = set()
        for i in range(len(text) - k + 1):
            gram = text[i:i + k]
            shingles.add(hash(gram))
        return shingles
    
    @staticmethod
    def jaccard_similarity(sig1: list[int], sig2: list[int]) -> float:
        """Compute Jaccard similarity between two MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LSHIndex:
    """Locality-Sensitive Hashing index for fast approximate nearest neighbors."""
    
    def __init__(self, num_bands: int = HASH_BANDS, rows_per_band: int = HASH_ROWS):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets: dict[str, list[str]] = defaultdict(list)
    
    def hash_to_band(self, signature: list[int], band_idx: int) -> str:
        """Hash a band of the signature to a bucket key."""
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return hashlib.md5(json.dumps(band).encode()).hexdigest()
    
    def index(self, hash_id: str, signature: list[int]):
        """Add a signature to the LSH index."""
        for band_idx in range(self.num_bands):
            bucket_key = self.hash_to_band(signature, band_idx)
            self.buckets[bucket_key].append(hash_id)
    
    def query(self, signature: list[int]) -> set[str]:
        """Find candidate matches for a signature."""
        candidates = set()
        for band_idx in range(self.num_bands):
            bucket_key = self.hash_to_band(signature, band_idx)
            candidates.update(self.buckets.get(bucket_key, []))
        return candidates
    
    def remove(self, hash_id: str):
        """Remove a hash_id from all buckets."""
        for bucket_key in list(self.buckets.keys()):
            self.buckets[bucket_key] = [
                h for h in self.buckets[bucket_key] if h != hash_id
            ]
            if not self.buckets[bucket_key]:
                del self.buckets[bucket_key]


# ---------------------------------------------------------------------------
# Semantic Hash Index Manager
# ---------------------------------------------------------------------------

class SemanticHashIndex:
    """Manages semantic hash index for exact-match deduplication."""
    
    def __init__(self):
        self.entries: dict[str, SemanticHashEntry] = {}
        self.minhash = MinHash(num_perm=HASH_PERMUTATIONS)
        self.lsh = LSHIndex(num_bands=HASH_BANDS, rows_per_band=HASH_ROWS)
        self._loaded = False
    
    def load(self):
        """Load semantic hash index from disk."""
        if self._loaded:
            return
        
        if SEMANTIC_HASH_STORAGE.exists():
            try:
                with open(SEMANTIC_HASH_STORAGE, 'r') as f:
                    data = json.load(f)
                
                for entry_data in data.get("entries", []):
                    entry = SemanticHashEntry(
                        hash_id=entry_data["hash_id"],
                        original_text=entry_data["original_text"],
                        hash_signature=entry_data["hash_signature"],
                        attack_category=entry_data["attack_category"],
                        risk_level=entry_data["risk_level"],
                        created_at=entry_data["created_at"],
                        metadata=entry_data.get("metadata", {}),
                    )
                    self.entries[entry.hash_id] = entry
                    self.lsh.index(entry.hash_id, entry.hash_signature)
                
                log.info("Loaded %d semantic hash entries", len(self.entries))
            except Exception as e:
                log.warning("Failed to load semantic hash index: %s", e)
        
        self._loaded = True
    
    def save(self):
        """Save semantic hash index to disk."""
        data = {
            "version": PHASE4_VERSION,
            "entries": [
                {
                    "hash_id": entry.hash_id,
                    "original_text": entry.original_text,
                    "hash_signature": entry.hash_signature,
                    "attack_category": entry.attack_category,
                    "risk_level": entry.risk_level,
                    "created_at": entry.created_at,
                    "metadata": entry.metadata,
                }
                for entry in self.entries.values()
            ],
            "stats": {
                "total_entries": len(self.entries),
                "created_at": time.time(),
            },
        }
        
        tmp_path = str(SEMANTIC_HASH_STORAGE) + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(SEMANTIC_HASH_STORAGE))
    
    def add_entry(
        self,
        text: str,
        attack_category: str,
        risk_level: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a new entry to the semantic hash index."""
        self.load()
        
        # Generate MinHash signature
        shingles = MinHash.text_to_shingles(text)
        signature = self.minhash.compute_signature(shingles)
        
        # Generate unique hash ID
        hash_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        entry = SemanticHashEntry(
            hash_id=hash_id,
            original_text=text[:2000],  # truncate for storage
            hash_signature=signature,
            attack_category=attack_category,
            risk_level=risk_level,
            created_at=time.time(),
            metadata=metadata or {},
        )
        
        # Remove old entry if exists
        if hash_id in self.entries:
            self.lsh.remove(hash_id)
        
        self.entries[hash_id] = entry
        self.lsh.index(hash_id, signature)
        
        # Periodically save
        if len(self.entries) % 100 == 0:
            self.save()
        
        return hash_id
    
    def find_similar(
        self,
        text: str,
        threshold: float = HASH_THRESHOLD,
        max_results: int = 5,
    ) -> list[tuple[str, float, SemanticHashEntry]]:
        """Find similar entries using LSH and verify with Jaccard similarity."""
        self.load()
        
        shingles = MinHash.text_to_shingles(text)
        query_signature = self.minhash.compute_signature(shingles)
        
        # Get candidates from LSH
        candidates = self.lsh.query(query_signature)
        
        # Verify with Jaccard similarity
        results = []
        for candidate_id in candidates:
            if candidate_id not in self.entries:
                continue
            
            entry = self.entries[candidate_id]
            similarity = MinHash.jaccard_similarity(query_signature, entry.hash_signature)
            
            if similarity >= threshold:
                results.append((candidate_id, similarity, entry))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def get_exact_match(self, text: str) -> Optional[SemanticHashEntry]:
        """Get exact match from semantic hash index."""
        self.load()
        
        hash_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return self.entries.get(hash_id)
    
    def remove_entry(self, hash_id: str):
        """Remove an entry from the index."""
        self.load()
        
        if hash_id in self.entries:
            del self.entries[hash_id]
            self.lsh.remove(hash_id)
    
    def get_stats(self) -> dict:
        """Get statistics about the semantic hash index."""
        self.load()
        
        return {
            "total_entries": len(self.entries),
            "categories": dict(defaultdict(int, {
                entry.attack_category: 1
                for entry in self.entries.values()
            })),
            "risk_levels": dict(defaultdict(int, {
                entry.risk_level: 1
                for entry in self.entries.values()
            })),
        }


# ---------------------------------------------------------------------------
# Multi-Turn Context Manager
# ---------------------------------------------------------------------------

class MultiTurnContextManager:
    """Manages multi-turn conversation context for security evaluation."""
    
    def __init__(
        self,
        max_turns: int = MAX_CONTEXT_TURNS,
        decay_factor: float = CONTEXT_DECAY_FACTOR,
        aggregation_mode: AggregationMode = AggregationMode.WEIGHTED,
    ):
        self.max_turns = max_turns
        self.decay_factor = decay_factor
        self.aggregation_mode = aggregation_mode
        self.sessions: dict[str, ConversationSession] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get or create a conversation session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]
    
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
        evaluation_result: Optional[dict] = None,
    ) -> ConversationTurn:
        """Add a turn to the conversation session."""
        session = self.get_or_create_session(session_id)
        
        turn = ConversationTurn(
            turn_id=len(session.turns),
            role=role,
            content=content,
            timestamp=time.time(),
            embedding=embedding,
            evaluation_result=evaluation_result,
            risk_score=evaluation_result.get("composite_score", 0.0) if evaluation_result else 0.0,
        )
        
        session.turns.append(turn)
        session.last_updated = time.time()
        
        # Trim to max turns
        if len(session.turns) > self.max_turns:
            session.turns = session.turns[-self.max_turns:]
            # Re-index turn_ids
            for i, t in enumerate(session.turns):
                t.turn_id = i
        
        # Update aggregate risk
        session.aggregate_risk = self._compute_aggregate_risk(session)
        
        # Check if session should be blocked
        if session.aggregate_risk >= 0.7:
            session.is_blocked = True
        
        return turn
    
    def _compute_aggregate_risk(self, session: ConversationSession) -> float:
        """Compute aggregate risk score across all turns."""
        if not session.turns:
            return 0.0
        
        if self.aggregation_mode == AggregationMode.WEIGHTED:
            return self._weighted_risk(session.turns)
        elif self.aggregation_mode == AggregationMode.CONCATENATED:
            return self._concatenated_risk(session.turns)
        elif self.aggregation_mode == AggregationMode.ATTENTION:
            return self._attention_risk(session.turns)
        else:
            return self._weighted_risk(session.turns)
    
    def _weighted_risk(self, turns: list[ConversationTurn]) -> float:
        """Compute risk with exponential decay weighting."""
        if not turns:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, turn in enumerate(turns):
            # More recent turns get higher weight
            weight = self.decay_factor ** (len(turns) - 1 - i)
            weighted_sum += weight * turn.risk_score
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _concatenated_risk(self, turns: list[ConversationTurn]) -> float:
        """Compute risk from concatenated context."""
        if not turns:
            return 0.0
        
        # Average risk across all turns
        return sum(t.risk_score for t in turns) / len(turns)
    
    def _attention_risk(self, turns: list[ConversationTurn]) -> float:
        """Compute risk with attention-based weighting."""
        if not turns:
            return 0.0
        
        # Simple attention: weight by risk score itself (higher risk gets more attention)
        scores = [t.risk_score for t in turns]
        total = sum(scores)
        
        if total == 0:
            return 0.0
        
        weights = [s / total for s in scores]
        return sum(w * s for w, s in zip(weights, scores))
    
    def get_context_window(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> list[ConversationTurn]:
        """Get the context window for a session."""
        session = self.get_or_create_session(session_id)
        
        if window_size is None:
            window_size = self.max_turns
        
        return session.turns[-window_size:]
    
    def get_context_embedding(
        self,
        session_id: str,
        embedding_engine,
    ) -> Optional[np.ndarray]:
        """Compute aggregated context embedding for a session."""
        session = self.get_or_create_session(session_id)
        
        if not session.turns:
            return None
        
        # Get embeddings for all turns
        embeddings = []
        weights = []
        
        for i, turn in enumerate(session.turns):
            if turn.embedding is not None:
                embeddings.append(turn.embedding)
                weight = self.decay_factor ** (len(session.turns) - 1 - i)
                weights.append(weight)
            else:
                # Compute embedding if not available
                emb = embedding_engine.encode([turn.content], normalize_embeddings=True)[0]
                turn.embedding = emb
                embeddings.append(emb)
                weight = self.decay_factor ** (len(session.turns) - 1 - i)
                weights.append(weight)
        
        if not embeddings:
            return None
        
        # Weighted average
        embeddings = np.array(embeddings)
        weights = np.array(weights).reshape(-1, 1)
        weights = weights / weights.sum()
        
        context_embedding = (embeddings * weights).sum(axis=0)
        context_embedding = context_embedding / np.linalg.norm(context_embedding)
        
        return context_embedding
    
    def get_conversation_history(
        self,
        session_id: str,
        include_embeddings: bool = False,
    ) -> list[dict]:
        """Get conversation history as JSON-serializable dict."""
        session = self.get_or_create_session(session_id)
        
        history = []
        for turn in session.turns:
            entry = {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "risk_score": turn.risk_score,
            }
            if turn.evaluation_result:
                entry["evaluation_result"] = turn.evaluation_result
            if include_embeddings and turn.embedding is not None:
                entry["embedding"] = turn.embedding.tolist()
            history.append(entry)
        
        return history
    
    def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a conversation session."""
        session = self.get_or_create_session(session_id)
        
        risk_scores = [t.risk_score for t in session.turns]
        
        return {
            "session_id": session_id,
            "total_turns": len(session.turns),
            "aggregate_risk": session.aggregate_risk,
            "is_blocked": session.is_blocked,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "avg_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
            "max_risk": max(risk_scores) if risk_scores else 0.0,
            "min_risk": min(risk_scores) if risk_scores else 0.0,
        }
    
    def clear_session(self, session_id: str):
        """Clear a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def save_sessions(self):
        """Save all sessions to disk."""
        data = {
            "version": PHASE4_VERSION,
            "sessions": {},
        }
        
        for session_id, session in self.sessions.items():
            data["sessions"][session_id] = {
                "session_id": session.session_id,
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "risk_score": t.risk_score,
                        "evaluation_result": t.evaluation_result,
                    }
                    for t in session.turns
                ],
                "aggregate_risk": session.aggregate_risk,
                "is_blocked": session.is_blocked,
                "created_at": session.created_at,
                "last_updated": session.last_updated,
                "metadata": session.metadata,
            }
        
        tmp_path = str(CONVERSATION_STORAGE) + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(CONVERSATION_STORAGE))
    
    def load_sessions(self):
        """Load sessions from disk."""
        if not CONVERSATION_STORAGE.exists():
            return
        
        try:
            with open(CONVERSATION_STORAGE, 'r') as f:
                data = json.load(f)
            
            for session_id, session_data in data.get("sessions", {}).items():
                session = ConversationSession(
                    session_id=session_data["session_id"],
                    aggregate_risk=session_data.get("aggregate_risk", 0.0),
                    is_blocked=session_data.get("is_blocked", False),
                    created_at=session_data.get("created_at", time.time()),
                    last_updated=session_data.get("last_updated", time.time()),
                    metadata=session_data.get("metadata", {}),
                )
                
                for turn_data in session_data.get("turns", []):
                    turn = ConversationTurn(
                        turn_id=turn_data["turn_id"],
                        role=turn_data["role"],
                        content=turn_data["content"],
                        timestamp=turn_data["timestamp"],
                        risk_score=turn_data.get("risk_score", 0.0),
                        evaluation_result=turn_data.get("evaluation_result"),
                    )
                    session.turns.append(turn)
                
                self.sessions[session_id] = session
            
            log.info("Loaded %d conversation sessions", len(self.sessions))
        except Exception as e:
            log.warning("Failed to load conversation sessions: %s", e)


# ---------------------------------------------------------------------------
# Streaming Evaluation Engine
# ---------------------------------------------------------------------------

class StreamingEvaluator:
    """Real-time streaming evaluation for conversational agents."""
    
    def __init__(
        self,
        embedding_engine=None,
        security_engine=None,
        hash_index: Optional[SemanticHashIndex] = None,
        context_manager: Optional[MultiTurnContextManager] = None,
    ):
        self.embedding_engine = embedding_engine
        self.security_engine = security_engine
        self.hash_index = hash_index or SemanticHashIndex()
        self.context_manager = context_manager or MultiTurnContextManager()
        
        # Active streaming sessions
        self.active_sessions: dict[str, dict] = {}
    
    def start_streaming_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> dict:
        """Start a new streaming evaluation session."""
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "accumulated_text": "",
            "chunk_count": 0,
            "start_time": time.time(),
            "state": StreamingState.IDLE,
            "last_evaluation": None,
        }
        
        return {
            "status": "ok",
            "session_id": session_id,
            "message": "Streaming session started",
        }
    
    async def process_chunk(
        self,
        session_id: str,
        chunk: StreamingChunk,
    ) -> StreamingEvaluationResult:
        """Process a streaming chunk and return evaluation result."""
        if session_id not in self.active_sessions:
            self.start_streaming_session(session_id)
        
        session = self.active_sessions[session_id]
        session["state"] = StreamingState.STREAMING
        
        # Accumulate text
        session["accumulated_text"] += chunk.content
        session["chunk_count"] += 1
        
        t0 = time.time()
        
        # Check semantic hash first (fast path)
        hash_match = None
        hash_match_id = None
        similar_matches = self.hash_index.find_similar(
            session["accumulated_text"],
            threshold=HASH_THRESHOLD,
            max_results=1,
        )
        
        if similar_matches:
            hash_match = similar_matches[0]
            hash_match_id = hash_match[0]
            similarity = hash_match[1]
            entry = hash_match[2]
            
            # If high-confidence hash match, block immediately
            if similarity >= 0.9 and entry.risk_level in ("critical", "high"):
                latency = (time.time() - t0) * 1000
                result = StreamingEvaluationResult(
                    chunk_id=chunk.chunk_id,
                    accumulated_text=session["accumulated_text"],
                    is_blocked=True,
                    risk_score=similarity,
                    attack_category=entry.attack_category,
                    risk_level=entry.risk_level,
                    confidence=similarity,
                    latency_ms=latency,
                    hash_match=True,
                    hash_match_id=hash_match_id,
                )
                session["state"] = StreamingState.BLOCKED
                session["last_evaluation"] = result
                return result
        
        # Compute embedding for accumulated text
        if self.embedding_engine is not None:
            embedding = self.embedding_engine.encode(
                [session["accumulated_text"]],
                normalize_embeddings=True,
            )[0]
        else:
            embedding = None
        
        # Get context embedding if available
        context_embedding = None
        if self.context_manager is not None and session_id:
            context_embedding = self.context_manager.get_context_embedding(
                session_id, self.embedding_engine
            )
        
        # Combine current and context embeddings
        if embedding is not None and context_embedding is not None:
            # Weighted combination: 70% current, 30% context
            combined_embedding = 0.7 * embedding + 0.3 * context_embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        elif embedding is not None:
            combined_embedding = embedding
        else:
            combined_embedding = context_embedding
        
        # Evaluate using security engine
        is_blocked = False
        risk_score = 0.0
        attack_category = None
        attack_technique = None
        risk_level = None
        confidence = 0.0
        signals = {}
        
        if combined_embedding is not None and self.security_engine is not None:
            try:
                # Search for similar patterns
                hits = self.security_engine.search_with_oversample(
                    combined_embedding,
                    text=session["accumulated_text"],
                )
                
                if hits:
                    # Re-rank with composite scoring
                    reranked = self.security_engine.rerank_with_composite(
                        hits,
                        text=session["accumulated_text"],
                        query_embedding=combined_embedding,
                        top_k=3,
                    )
                    
                    top_hit = reranked[0]
                    risk_score = top_hit["composite_score"]
                    signals = top_hit["signal_scores"]
                    payload = top_hit["payload"]
                    
                    is_blocked = payload.get("is_malicious", False) and risk_score >= 0.55
                    attack_category = payload.get("attack_category")
                    attack_technique = payload.get("attack_technique")
                    risk_level = payload.get("risk_level")
                    confidence = min(risk_score, 1.0)
            except Exception as e:
                log.warning("Security engine evaluation failed: %s", e)
        
        latency = (time.time() - t0) * 1000
        
        result = StreamingEvaluationResult(
            chunk_id=chunk.chunk_id,
            accumulated_text=session["accumulated_text"],
            is_blocked=is_blocked,
            risk_score=risk_score,
            attack_category=attack_category,
            attack_technique=attack_technique,
            risk_level=risk_level,
            confidence=confidence,
            signals=signals,
            latency_ms=latency,
            hash_match=hash_match is not None,
            hash_match_id=hash_match_id,
        )
        
        session["state"] = StreamingState.BLOCKED if is_blocked else StreamingState.EVALUATING
        session["last_evaluation"] = result
        
        return result
    
    def finalize_streaming_session(
        self,
        session_id: str,
        final_evaluation_result: Optional[dict] = None,
    ) -> dict:
        """Finalize a streaming session and update context."""
        if session_id not in self.active_sessions:
            return {"status": "error", "message": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["state"] = StreamingState.COMPLETE
        
        # Add to multi-turn context
        if self.context_manager is not None:
            self.context_manager.add_turn(
                session_id=session_id,
                role="user",
                content=session["accumulated_text"],
                evaluation_result=final_evaluation_result,
            )
        
        # Add to semantic hash index if malicious
        if session.get("last_evaluation") and session["last_evaluation"].is_blocked:
            self.hash_index.add_entry(
                text=session["accumulated_text"],
                attack_category=session["last_evaluation"].attack_category or "unknown",
                risk_level=session["last_evaluation"].risk_level or "medium",
                metadata={"session_id": session_id, "source": "streaming"},
            )
        
        # Clean up
        result = {
            "status": "ok",
            "session_id": session_id,
            "total_chunks": session["chunk_count"],
            "final_text": session["accumulated_text"],
            "final_evaluation": session["last_evaluation"].__dict__ if session.get("last_evaluation") else None,
            "duration_ms": (time.time() - session["start_time"]) * 1000,
        }
        
        del self.active_sessions[session_id]
        return result
    
    def get_session_status(self, session_id: str) -> Optional[dict]:
        """Get status of a streaming session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "state": session["state"],
            "chunk_count": session["chunk_count"],
            "accumulated_length": len(session["accumulated_text"]),
            "duration_ms": (time.time() - session["start_time"]) * 1000,
        }


# ---------------------------------------------------------------------------
# Phase 4 API Endpoints
# ---------------------------------------------------------------------------

def create_phase4_routes(app):
    """Add Phase 4 routes to the FastAPI app."""
    from fastapi import WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field
    
    # Initialize Phase 4 components
    hash_index = SemanticHashIndex()
    context_manager = MultiTurnContextManager()
    streaming_evaluator = StreamingEvaluator(hash_index=hash_index, context_manager=context_manager)
    
    class StreamingStartRequest(BaseModel):
        session_id: str = Field(..., description="Unique session identifier")
        user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    class StreamingChunkRequest(BaseModel):
        session_id: str = Field(..., description="Session identifier")
        chunk_id: int = Field(..., description="Chunk sequence number")
        content: str = Field(..., description="Chunk content")
        is_final: bool = Field(False, description="Whether this is the final chunk")
    
    class ConversationTurnRequest(BaseModel):
        session_id: str = Field(..., description="Session identifier")
        role: str = Field(..., description="Turn role: user or assistant")
        content: str = Field(..., min_length=1, max_length=10000, description="Turn content")
    
    class HashIndexAddRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000, description="Text to index")
        attack_category: str = Field(..., description="Attack category")
        risk_level: str = Field(..., description="Risk level")
        metadata: Optional[dict] = Field(None, description="Additional metadata")
    
    class HashIndexQueryRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=10000, description="Text to query")
        threshold: float = Field(HASH_THRESHOLD, ge=0.0, le=1.0, description="Similarity threshold")
        max_results: int = Field(5, ge=1, le=50, description="Maximum results")
    
    # --- Streaming Endpoints ---
    
    @app.post("/v1/streaming/start")
    def start_streaming(req: StreamingStartRequest):
        """Start a streaming evaluation session."""
        result = streaming_evaluator.start_streaming_session(
            session_id=req.session_id,
            user_id=req.user_id,
        )
        return result
    
    @app.post("/v1/streaming/chunk")
    async def process_streaming_chunk(req: StreamingChunkRequest):
        """Process a streaming chunk."""
        chunk = StreamingChunk(
            chunk_id=req.chunk_id,
            content=req.content,
            is_final=req.is_final,
            timestamp=time.time(),
            session_id=req.session_id,
        )
        
        result = await streaming_evaluator.process_chunk(
            session_id=req.session_id,
            chunk=chunk,
        )
        
        return {
            "chunk_id": result.chunk_id,
            "is_blocked": result.is_blocked,
            "risk_score": result.risk_score,
            "attack_category": result.attack_category,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "hash_match": result.hash_match,
        }
    
    @app.post("/v1/streaming/finalize")
    def finalize_streaming(session_id: str):
        """Finalize a streaming session."""
        result = streaming_evaluator.finalize_streaming_session(session_id)
        return result
    
    @app.get("/v1/streaming/status/{session_id}")
    def get_streaming_status(session_id: str):
        """Get streaming session status."""
        result = streaming_evaluator.get_session_status(session_id)
        if result is None:
            return {"status": "error", "message": "Session not found"}
        return result
    
    # --- WebSocket Streaming ---
    
    @app.websocket("/ws/streaming/{session_id}")
    async def websocket_streaming(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time streaming evaluation."""
        await websocket.accept()
        
        streaming_evaluator.start_streaming_session(session_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                chunk = StreamingChunk(
                    chunk_id=message.get("chunk_id", 0),
                    content=message.get("content", ""),
                    is_final=message.get("is_final", False),
                    timestamp=time.time(),
                    session_id=session_id,
                )
                
                result = await streaming_evaluator.process_chunk(
                    session_id=session_id,
                    chunk=chunk,
                )
                
                response = {
                    "chunk_id": result.chunk_id,
                    "is_blocked": result.is_blocked,
                    "risk_score": result.risk_score,
                    "attack_category": result.attack_category,
                    "risk_level": result.risk_level,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "hash_match": result.hash_match,
                }
                
                await websocket.send_text(json.dumps(response))
                
                if result.is_blocked or chunk.is_final:
                    if chunk.is_final:
                        streaming_evaluator.finalize_streaming_session(session_id)
                    break
        
        except WebSocketDisconnect:
            streaming_evaluator.finalize_streaming_session(session_id)
    
    # --- Multi-Turn Context Endpoints ---
    
    @app.post("/v1/context/turn")
    def add_conversation_turn(req: ConversationTurnRequest):
        """Add a turn to the conversation context."""
        turn = context_manager.add_turn(
            session_id=req.session_id,
            role=req.role,
            content=req.content,
        )
        
        return {
            "status": "ok",
            "turn_id": turn.turn_id,
            "session_id": req.session_id,
            "total_turns": len(context_manager.get_or_create_session(req.session_id).turns),
        }
    
    @app.get("/v1/context/history/{session_id}")
    def get_conversation_history(session_id: str, window_size: Optional[int] = None):
        """Get conversation history."""
        turns = context_manager.get_context_window(session_id, window_size)
        
        history = []
        for turn in turns:
            entry = {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "risk_score": turn.risk_score,
            }
            if turn.evaluation_result:
                entry["evaluation_result"] = turn.evaluation_result
            history.append(entry)
        
        return {
            "session_id": session_id,
            "turns": history,
            "total_turns": len(history),
        }
    
    @app.get("/v1/context/stats/{session_id}")
    def get_context_stats(session_id: str):
        """Get context statistics."""
        return context_manager.get_session_stats(session_id)
    
    @app.delete("/v1/context/{session_id}")
    def clear_context(session_id: str):
        """Clear conversation context."""
        context_manager.clear_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} cleared"}
    
    # --- Semantic Hash Index Endpoints ---
    
    @app.post("/v1/hash/add")
    def add_to_hash_index(req: HashIndexAddRequest):
        """Add an entry to the semantic hash index."""
        hash_id = hash_index.add_entry(
            text=req.text,
            attack_category=req.attack_category,
            risk_level=req.risk_level,
            metadata=req.metadata,
        )
        
        return {
            "status": "ok",
            "hash_id": hash_id,
        }
    
    @app.post("/v1/hash/query")
    def query_hash_index(req: HashIndexQueryRequest):
        """Query the semantic hash index."""
        results = hash_index.find_similar(
            text=req.text,
            threshold=req.threshold,
            max_results=req.max_results,
        )
        
        matches = []
        for hash_id, similarity, entry in results:
            matches.append({
                "hash_id": hash_id,
                "similarity": round(similarity, 4),
                "original_text": entry.original_text[:200],
                "attack_category": entry.attack_category,
                "risk_level": entry.risk_level,
            })
        
        return {
            "query_text": req.text[:100],
            "matches": matches,
            "total_matches": len(matches),
        }
    
    @app.get("/v1/hash/stats")
    def get_hash_index_stats():
        """Get semantic hash index statistics."""
        return hash_index.get_stats()
    
    @app.delete("/v1/hash/{hash_id}")
    def remove_from_hash_index(hash_id: str):
        """Remove an entry from the semantic hash index."""
        hash_index.remove_entry(hash_id)
        return {"status": "ok", "message": f"Entry {hash_id} removed"}
    
    # --- Phase 4 Stats ---
    
    @app.get("/v1/phase4/stats")
    def get_phase4_stats():
        """Get Phase 4 statistics."""
        return {
            "version": PHASE4_VERSION,
            "streaming": {
                "active_sessions": len(streaming_evaluator.active_sessions),
            },
            "context": {
                "active_sessions": len(context_manager.sessions),
            },
            "hash_index": hash_index.get_stats(),
            "configuration": {
                "max_context_turns": context_manager.max_turns,
                "decay_factor": context_manager.decay_factor,
                "aggregation_mode": context_manager.aggregation_mode,
                "hash_threshold": HASH_THRESHOLD,
            },
        }
    
    return app


# ---------------------------------------------------------------------------
# Integration with existing security engine
# ---------------------------------------------------------------------------

def integrate_phase4():
    """Integrate Phase 4 components with the existing security engine."""
    from security_engine import app, get_embedding_engine, get_qdrant_client, get_corpus_meta
    
    # Initialize Phase 4 components
    hash_index = SemanticHashIndex()
    context_manager = MultiTurnContextManager()
    
    # Get existing engine components
    embedding_engine = get_embedding_engine()
    qdrant_client = get_qdrant_client()
    corpus_meta = get_corpus_meta()
    
    # Create streaming evaluator with existing components
    streaming_evaluator = StreamingEvaluator(
        embedding_engine=embedding_engine,
        hash_index=hash_index,
        context_manager=context_manager,
    )
    
    # Add Phase 4 routes
    create_phase4_routes(app)
    
    log.info("Phase 4 integrated successfully")
    return streaming_evaluator, hash_index, context_manager


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run Phase 4 standalone for testing."""
    import uvicorn
    
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Guardrailer Phase 4 Prototype",
        description="Streaming evaluation, multi-turn context, and semantic hash index",
        version=PHASE4_VERSION,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add Phase 4 routes
    create_phase4_routes(app)
    
    @app.get("/health")
    def health():
        return {"status": "ok", "version": PHASE4_VERSION, "phase": 4}
    
    host = os.environ.get("GUARDRAILER_HOST", "0.0.0.0")
    port = int(os.environ.get("GUARDRAILER_PORT", "8091"))
    
    log.info("Starting Phase 4 Prototype on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
