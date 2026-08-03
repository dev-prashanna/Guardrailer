# Guardrailer: Technical Architecture for Industry-Grade Prompt Guardrails

| Version | Date | Status |
|---------|------|--------|
| 3.0 | 2026-08-03 | Active |

---

## 1. Design Philosophy

Guardrailer implements a **multi-signal, cascaded defense architecture** for prompt injection detection. The design prioritizes three invariants:

1. **High Accuracy** — Multi-signal fusion with learned weights and probability calibration to minimize both false positives and false negatives across attack categories.
2. **Low Latency** — Fast-path decisions in <15ms for clear-cut cases; LLM evaluation reserved only for ambiguous inputs.
3. **Research-Level Sophistication** — Contrastive fine-tuned embeddings, hard negative mining, ensemble scoring, and continuous feedback-driven retraining.

The system operates as a **RAG-augmented classifier**: it retrieves similar known attack patterns from a vector database, scores the input against multiple independent signals, and routes ambiguous cases to an LLM-as-judge for final adjudication.

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
                          ┌──────────────────────────────┐
                          │         User Prompt           │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │    Preprocessing & Encoding   │
                          │    ┌────────────────────┐     │
                          │    │ Embedding Engine    │     │
                          │    │ (single/ensemble)   │     │
                          │    └────────┬───────────┘     │
                          └─────────────┼─────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │       Qdrant Hybrid Retrieval             │
                     │   dense vector + IDF-weighted sparse      │
                     │   Fusion: RRF (Reciprocal Rank Fusion)    │
                     │   Oversample: 12 candidates               │
                     └──────────────────┬───────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │       Multi-Signal Re-Ranking             │
                     │   10 independent scoring signals          │
                     │   Learned weight combination              │
                     │   Probability calibration (Platt/Isotonic)│
                     │   Top-3 candidates selected               │
                     └──────────────────┬───────────────────────┘
                                        │
                       ┌────────────────┼────────────────────┐
                       ▼                ▼                    ▼
              ┌──────────────┐ ┌──────────────────┐ ┌──────────────┐
              │  LAYER 1     │ │  LAYER 2         │ │  SAFE        │
              │  Fast Block  │ │  Deep Path       │ │  Allow       │
              │  <15ms       │ │  LLM Ensemble    │ │  <15ms       │
              │  composite   │ │  ~220ms          │ │  composite   │
              │  ≥ 0.55      │ │  0.30–0.55       │ │  < 0.30      │
              │  + malicious │ │                  │ │              │
              │  + crit/high │ │                  │ │              │
              └──────────────┘ └────────┬─────────┘ └──────────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  LLM Judge Ensemble   │
                             │  xiaomi/mimo-v2.5     │
                             │  N=1–3 parallel calls │
                             │  Temperature sweep    │
                             │  Majority vote        │
                             └──────────────────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  Decision & Response  │
                             │  + Feedback Logging   │
                             └──────────────────────┘
```

### 2.2 Layered Decision Model

The three-layer cascade is the core latency-accuracy optimization. Each layer represents a different confidence tier:

| Layer | Trigger Condition | Latency | Method | Use Case |
|-------|------------------|---------|--------|----------|
| **Layer 1: Fast Block** | `composite ≥ 0.55` AND `is_malicious=True` AND `risk_level ∈ {critical, high}` | <15ms | Score-only | Obvious attacks: known patterns with high confidence |
| **Layer 2: Deep Path** | `0.30 ≤ composite < 0.55` OR `signals_strong_agree < 3` | ~220ms | LLM ensemble | Ambiguous cases: novel attacks, subtle extraction, borderline inputs |
| **Layer 3: Safe** | `composite < 0.30` AND `signals_strong_agree == 0` | <15ms | Score-only | Clear benign: no suspicious signals detected |

An intermediate **strong signal agreement** path bypasses the LLM when ≥3 of 4 key signals agree (composite ≥0.40, dense ≥0.70, cross-encoder ≥0.60, sparse keywords ≥1) and dense ≥0.75, providing a second fast-block mechanism without LLM cost.

---

## 3. Scoring Architecture

### 3.1 Signal Inventory

The system computes **10 independent scoring signals**, each capturing a different aspect of prompt maliciousness:

| # | Signal | Weight (Default) | Computation | Latency | Phase |
|---|--------|------------------|-------------|---------|-------|
| 1 | `dense_score` | 0.35 | Cosine similarity via BGE-large-en-v1.5 (1024-dim) | ~10ms (encode) | 1 |
| 2 | `sparse_idf_score` | 0.20 | IDF-weighted BM25 keyword match over 54 security keywords | ~0.1ms | 1 |
| 3 | `centroid_score` | 0.15 | Max cosine distance to 6 category centroids | ~0.5ms | 1 |
| 4 | `cross_encoder_score` | 0.15 | Pre-computed relevance via ms-marco-MiniLM-L-6-v2 | 0ms (pre-computed) | 1 |
| 5 | `perplexity_score` | 0.01 | Token diversity + entropy anomaly detection | ~0.5ms | 3 |
| 6 | `entropy_score` | 0.01 | Shannon entropy at character and word levels | ~0.3ms | 3 |
| 7 | `token_frequency_score` | 0.01 | Rare word ratio + attack-token density | ~0.2ms | 3 |
| 8 | `ngram_overlap_score` | 0.01 | Bigram/trigram overlap with 42 known attack phrases | ~0.3ms | 3 |
| 9 | `uniqueness_score` | 0.06 | Inverse mean distance to k=10 nearest neighbors (pre-computed) | 0ms (pre-computed) | 1 |
| 10 | `length_norm_score` | 0.05 | `log1p(text_length) / log1p(avg_length)` | ~0.01ms | 1 |

### 3.2 Composite Score Formula

```
composite = Σ (weight_i × signal_i) + attack_boost + subtle_boost - benign_penalty
```

Where:
- **attack_boost** = `min(0.3, attack_matches × 0.08)` — direct pattern match bonus
- **subtle_boost** = `min(0.15, subtle_matches × 0.05)` — subtle extraction bonus (only when no direct attacks)
- **benign_penalty** = `0.05–0.3` scaled by benign pattern match count (only when no attack/subtle patterns)

The final composite is clamped to `[0, 1]`.

### 3.3 Learned Weight Systems (Phase 3)

Rather than relying solely on hand-tuned weights, the system supports three learned weight combination modes:

| Mode | Model | Architecture | Use Case |
|------|-------|-------------|----------|
| `logistic` | Logistic Regression | Linear: `coef × features + intercept` | Interpretable baseline; fast inference |
| `neural` | MLP Classifier | 32→16 hidden layers, ReLU, early stopping | Non-linear weight interactions |
| `attention` | Multi-Head Attention | 2-head attention + FC (10→32→10) + Softmax | Adaptive per-input weight allocation |
| `ensemble` | Stacking | Weighted average of all three + calibration | Maximum robustness via model diversity |

All weight learners are trained on labeled signal features (10-dimensional) and persist to `models/` directory. The active mode is configured via `phase3_meta.json`.

### 3.4 Probability Calibration

Raw composite scores are calibrated to meaningful probabilities using:

| Method | Implementation | When Used |
|--------|---------------|-----------|
| **Platt Scaling** | `CalibratedClassifierCV(base_lr, cv=5, method='sigmoid')` | Default calibration |
| **Isotonic Regression** | `IsotonicRegression(out_of_bounds='clip')` | Non-parametric alternative |
| **Sigmoid fallback** | `1 / (1 + exp(-k × (x - x0)))` with `k=10, x0=0.5` | When no calibrator is trained |

Calibration ensures that a score of 0.7 corresponds to a ~70% true malicious probability, enabling well-calibrated threshold setting.

---

## 4. Embedding Engine

### 4.1 Model Registry

| Model | ID | Dimension | Max Seq | Weight | Multilingual | Sparse |
|-------|----|-----------|---------|--------|-------------|--------|
| BGE-large-en-v1.5 | `BAAI/bge-large-en-v1.5` | 1024 | 512 | 0.40 | No | No |
| BGE-M3 | `BAAI/bge-m3` | 1024 | 8192 | 0.35 | Yes | Yes |
| MxBai-embed-large-v1 | `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 512 | 0.25 | No | No |
| BGE-base-en-v1.5 | `BAAI/bge-base-en-v1.5` | 768 | 512 | 1.00 | No | No (legacy) |

### 4.2 Embedding Modes

| Mode | Strategy | Description |
|------|----------|-------------|
| `single` | One model | Default; uses primary model only |
| `average` | Weighted mean | `Σ(w_i × emb_i) / Σ(w_i)`, re-normalized |
| `late` | Concatenate + project | Concatenate all model outputs, chunked average-pool to target dim |
| `max_sim` | Per-dim max | Each dimension takes the value from the most confident model |

### 4.3 Contrastive Fine-Tuning Pipeline

The `ContrastiveTrainer` fine-tunes embedding models on security-specific data:

**Training Objectives:**
1. **Triplet Loss** — `(anchor, positive, negative)` with configurable margin (default 0.5)
2. **InfoNCE** — In-batch negatives with temperature scaling (default 0.02)
3. **Supervised Contrastive** — Same-category positives, cross-category negatives

**Hard Negative Mining** (`HardNegativeMiner`):
- **Distance-based**: Closest samples in embedding space with different labels
- **Confusion-based**: Samples closest to wrong category centroids
- **Boundary-based**: Samples near the decision boundary (within margin of 0.1)

**Training Configuration:**
- Base model: `BAAI/bge-large-en-v1.5`
- Learning rate: `2e-5` with linear warmup (10% of steps)
- Batch size: 16
- Epochs: 3
- Validation: 15% holdout with embedding similarity evaluation

---

## 5. Retrieval Pipeline

### 5.1 Query-Time Flow

```
Step 1: Encode query                         ~10ms
  └─ EmbeddingEngine.encode(query, normalize=True)
  └─ Returns 1024-dim normalized vector

Step 2: Hybrid search (12 candidates)        ~5ms
  ├─ Prefetch A: dense vector search (top_k × 3 = 36)
  ├─ Prefetch B: IDF-weighted sparse search (top_k × 3 = 36)
  └─ Fusion: Reciprocal Rank Fusion (RRF)
  └─ Returns top 12 candidates

Step 3: Re-rank with composite score         ~2ms
  ├─ Score all 12 candidates with 10 signals
  ├─ Apply learned weights (if available)
  ├─ Apply probability calibration
  └─ Return top 3 candidates

Step 4: Decision routing
  ├─ composite ≥ 0.55 + malicious + critical/high → BLOCK (Layer 1)
  ├─ signals_strong_agree ≥ 3 + dense ≥ 0.75 → BLOCK (fast secondary)
  ├─ composite < 0.30 + signals_strong_agree == 0 → ALLOW (Safe)
  └─ Otherwise → LLM evaluation (Layer 2)

Step 5: LLM evaluation (if Layer 2)          ~200ms
  ├─ Build evaluator payload (top-3 context + query)
  ├─ N parallel LLM calls (temperature sweep: 0.1, 0.2, ...)
  ├─ Majority vote on is_malicious
  └─ Return verdict with reasoning
```

### 5.2 Hybrid Search Implementation

The hybrid search combines dense semantic retrieval with sparse lexical matching:

```
Dense path:   query_embedding → Qdrant dense vector index → top_k × 3 candidates
Sparse path:  IDF-weighted sparse vector → Qdrant sparse index → top_k × 3 candidates
Fusion:       Reciprocal Rank Fusion (RRF) merges both candidate sets
```

**IDF-Weighted Sparse Vector Construction:**
```python
for keyword in SPARSE_KEYWORDS:  # 54 keywords
    if keyword in text.lower():
        idf = log((N + 1) / (df[keyword] + 1)) + 1.0
        tf = count(keyword) / max(word_count, 1)
        bm25_tf = (tf × 2.0) / (tf + 1.5 × (1.0 - 0.75 + 0.75 × word_count / avgdl))
        score = idf × (bm25_tf + 1.0)
```

### 5.3 LLM Evaluator

**Model:** `xiaomi/mimo-v2.5` (OpenAI-compatible API)

**System Prompt:** "AI Security Inspector" — analyzes queries against 5 attack dimensions:
1. Direct attacks (override, extract, bypass)
2. Indirect attacks (hidden instructions, embedded commands)
3. Subtle extraction (legitimate-sounding information requests)
4. Refusal bypass (safety circumvention)
5. Jailbreak (roleplay, personas, hypotheticals)

**Ensemble Strategy:**
- `ensemble_size` configurable (default 3 for high-confidence cases, 1 for lower)
- Temperature sweep: `[0.1, 0.2, 0.3]` for diversity
- Majority vote: `malicious_votes ≥ ceil(N/2)` → blocked
- Confidence: `malicious_votes / N`

**Fallback:** If LLM is unavailable, the system falls back to signal-agreement heuristic (≥2 signals agree → block at medium risk).

---

## 6. Ingestion Pipeline

### 6.1 Overview

The ingestion pipeline pre-computes all scoring features offline (Kaggle T4 GPU, ~20 minutes for ~1M documents), then exports files for local Qdrant upload.

```
Phase 1: Corpus Statistics
  ├─ IDF per keyword: log((N+1) / (df[kw]+1)) + 1.0
  ├─ Average document length (words)
  └─ Average text length (characters)

Phase 2: Dense Embeddings (GPU)
  ├─ Model: BAAI/bge-large-en-v1.5 (1024-dim)
  ├─ All ~1M documents encoded
  └─ L2-normalized embeddings → dense_embeddings_float16.npy

Phase 3: Category Centroids
  ├─ Mean embedding per attack category (6 categories)
  └─ Normalized to unit vectors → corpus_meta.json

Phase 4: k-Means Clustering (FAISS-GPU)
  ├─ 128 clusters via MiniBatchKMeans
  └─ Cluster assignments → enhanced_payloads.parquet

Phase 5: k-NN Graph (FAISS-GPU)
  ├─ k=10 nearest neighbors per document
  ├─ Cosine distance
  └─ Neighbor IDs → enhanced_payloads.parquet

Phase 6: Uniqueness Scores
  ├─ Uniqueness = 1 / mean_distance_to_k_nn
  └─ Min-max normalized to [0, 1] → enhanced_payloads.parquet

Phase 7: Cross-Encoder Pre-Scoring (GPU)
  ├─ Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ├─ Each document scored against its category description
  ├─ Sigmoid activation for probability
  └─ Scores → enhanced_payloads.parquet

Phase 8: Export
  ├─ corpus_meta.json (IDF, centroids, weights, statistics)
  ├─ enhanced_payloads.parquet (all metadata + pre-computed fields)
  └─ dense_embeddings_float16.npy (compressed embeddings)
```

### 6.2 Per-Point Qdrant Payload

```json
{
  "prompt_text": "truncated to 2000 chars",
  "is_malicious": true,
  "attack_category": "jailbreak",
  "attack_technique": "few_shot_override",
  "risk_level": "critical",
  "source_dataset": "allenai/wildjailbreak",
  "cluster_id": 42,
  "neighbor_ids": [1234, 5678, 9012, 3456, 7890],
  "uniqueness": 0.7234,
  "cross_encoder_score": 0.8901,
  "length_norm": 0.9876
}
```

### 6.3 Corpus Metadata (`corpus_meta.json`)

```json
{
  "keyword_idf": {"ignore previous": 2.34, "bypass": 1.89, ...},
  "total_documents": 693000,
  "avg_doc_length": 45.2,
  "avg_text_length": 267.8,
  "category_centroids": {
    "jailbreak": [0.012, -0.034, ...],
    "direct_injection": [...],
    "indirect_injection": [...],
    "system_prompt_extraction": [...],
    "refusal_bypass": [...],
    "benign_control": [...]
  },
  "scoring_weights": {
    "dense": 0.35, "sparse_idf": 0.20, "centroid": 0.15,
    "cross_encoder": 0.15, "uniqueness": 0.06, "length_norm": 0.05,
    "perplexity": 0.01, "entropy": 0.01, "token_frequency": 0.01, "ngram_overlap": 0.01
  }
}
```

---

## 7. Feedback Loop & Continuous Learning

### 7.1 Feedback Pipeline

```
User reports FP/FN via POST /v1/feedback
        │
        ▼
feedback_logger.py
  ├─ Thread-safe JSONL append (auto-rotation at 100MB)
  ├─ Statistics update (FP/FN counts, category distribution)
  └─ Stored in feedback_data/feedback_log.jsonl
        │
        ▼
feedback_processor.py
  ├─ Reads unprocessed corrections
  ├─ Normalizes categories/risk levels/techniques
  ├─ Generates new dense embeddings (BGE-large-en-v1.5)
  ├─ Builds IDF-weighted sparse vectors
  ├─ Upserts to Qdrant (ID offset +1,000,000 to avoid collisions)
  ├─ Appends to unified_security_dataset.parquet
  └─ Marks entries as processed
        │
        ▼
Next ingestion run
  ├─ Picks up new samples from Parquet
  ├─ Recomputes IDF, centroids, clusters, k-NN, uniqueness
  └─ Exports updated corpus metadata
```

### 7.2 Retraining Pipeline

```
POST /v1/feedback-to-training
  ├─ Reads feedback_log.jsonl
  ├─ Extracts labeled samples (FP → label 0, FN → label 1)
  ├─ Extracts 10-dimensional feature vectors
  ├─ Trains weight learner (logistic/neural/attention)
  ├─ Trains calibrator (Platt/isotonic)
  ├─ Saves models to models/ directory
  └─ Hot-swaps active weight mode
```

### 7.3 Data Sources

The initial corpus is built from 8 public HuggingFace datasets:

| Dataset | Purpose | Approximate Size |
|---------|---------|-----------------|
| `allenai/wildjailbreak` | Real-world jailbreak attempts | ~180K |
| `lmsys/jailbreak-queries` | Curated jailbreak queries | ~50K |
| `DeepPavlov/extraction-prompts` | System prompt extraction | ~30K |
| `Anthropic/hh-rlhf` | Harmful/harmless pairs | ~170K |
| `OpenAI/instruction-following` | Benign instruction examples | ~130K |
| `tatsu-lab/alpaca` | General instruction data | ~52K |
| `GBaker/Belle-3.5M-train-cn` | Multilingual benign examples | ~100K |
| Synthetic generation | Augmented attack variants | ~300K |

Total: ~1M samples, balanced across 6 attack categories + benign control.

---

## 8. System Components

### 8.1 Module Map

```
guardrailer_security/
├── security_engine.py              # FastAPI engine — main entry point (1222 lines)
├── scoring.py                      # Core multi-signal scoring (369 lines)
├── improved_scoring.py             # Phase 3: learned weights, signals, calibration (965 lines)
├── constants.py                    # 54 sparse keywords, 43 attack patterns, 14 subtle, 34 benign
├── train_phase3.py                 # Phase 3 training with cross-validation (442 lines)
├── ingest_precomputed.py           # Safe Qdrant ingestion with checkpointing (415 lines)
├── feedback_logger.py              # Thread-safe JSONL feedback persistence (166 lines)
├── feedback_processor.py           # Feedback → training data pipeline (281 lines)
├── download_and_merge.py           # HuggingFace dataset preparation (1140+ lines)
├── pint_benchmark_pipeline.py      # PINT benchmark evaluation (435 lines)
├── corpus_meta.json                # Pre-computed IDF, centroids, statistics
├── phase3_meta.json                # Active weight mode configuration
├── unified_security_dataset.parquet # ~1M training corpus
├── guardrailer_benchmark_final.yaml # 200-sample evaluation dataset
├── models/
│   ├── logistic_weights.json       # Logistic regression coefficients
│   ├── neural_weights.json         # MLP (32→16) weights
│   ├── attention_weights.pt        # Multi-head attention (PyTorch)
│   └── calibrator.json             # Platt scaling calibrator
├── embedding_engine/
│   ├── config.py                   # Model registry & EmbeddingMode enum
│   ├── engine.py                   # Core embedding engine (360 lines)
│   ├── ensemble.py                 # 4 ensemble strategies (223 lines)
│   ├── fine_tune.py                # Contrastive fine-tuning pipeline (348 lines)
│   └── hard_negatives.py           # 3 mining strategies (347 lines)
├── feedback_data/
│   ├── feedback_log.jsonl          # Feedback entries
│   └── feedback_stats.json         # Aggregated statistics
└── guardrailer_output/             # Kaggle export directory
    ├── corpus_meta.json
    ├── enhanced_payloads.parquet
    ├── dense_embeddings_float16.npy
    └── ingest_checkpoint.json

crypto_agent/                       # Standalone sub-project
├── app.py                          # Streamlit web UI (685 lines)
├── agent.py                        # Analysis orchestrator
├── decoder/
│   ├── crypto_decoder.py           # 17 encoding types, multi-layer decode
│   └── hash_identifier.py          # 14 hash type identification
├── classifier/
│   ├── safety_classifier.py        # Rule-based threat classifier (15 categories)
│   └── llm_classifier.py           # LLM-enhanced classifier with RAG
└── rag/
    ├── embeddings.py               # Sentence transformer (all-MiniLM-L6-v2)
    ├── retriever.py                # Semantic retrieval (top-k cosine)
    └── knowledge_base.py           # 90 threat intelligence examples
```

### 8.2 API Surface

| Endpoint | Method | Description | Latency |
|----------|--------|-------------|---------|
| `/v1/evaluate-prompt` | POST | Evaluate a prompt for security threats | <15ms–220ms |
| `/v1/feedback` | POST | Submit false positive/negative correction | <5ms |
| `/v1/stats` | GET | Engine stats, collection info, feedback stats | <10ms |
| `/v1/feedback/recent` | GET | Recent feedback entries | <5ms |
| `/v1/feedback/pending` | GET | Pending correction samples | <5ms |
| `/v1/train` | POST | Train Phase 3 improved scorer | Variable |
| `/v1/feedback-to-training` | POST | Process feedback into training data and retrain | Variable |
| `/v1/set-weight-mode` | POST | Switch active weight combination mode | <5ms |
| `/v1/models` | GET | Loaded embedding model information | <5ms |
| `/health` | GET | Health check | <1ms |

### 8.3 Response Schema

```json
{
  "query": "user prompt",
  "is_blocked": true,
  "layer": "fast_block | deep_path | safe",
  "composite_score": 0.8734,
  "similarity_score": 0.9102,
  "signal_scores": {
    "dense": 0.9102,
    "sparse_idf": 0.4523,
    "centroid": 0.8234,
    "cross_encoder": 0.8901,
    "perplexity": 0.1234,
    "entropy": 0.0987,
    "token_frequency": 0.3456,
    "ngram_overlap": 0.2789,
    "uniqueness": 0.3200,
    "length_norm": 0.9876,
    "composite": 0.8734
  },
  "attack_category": "jailbreak",
  "attack_technique": "few_shot_override",
  "risk_level": "critical",
  "reasoning": "...",
  "latency_ms": 45.2,
  "retrieved_context": [...],
  "llm_verdict": {
    "is_malicious": true,
    "risk_score": 0.92,
    "attack_category": "jailbreak",
    "attack_technique": "few_shot_override",
    "risk_level": "critical",
    "confidence": 0.87,
    "reasoning": "...",
    "ensemble_votes": 2,
    "ensemble_total": 3
  }
}
```

---

## 9. Performance Characteristics

### 9.1 Latency Profile

| Step | Time | Notes |
|------|------|-------|
| Dense encoding | ~10ms | BGE-large-en-v1.5, single text |
| Qdrant hybrid search | ~5ms | 12 candidates via RRF |
| Composite re-ranking | ~2ms | 10 signals × 12 candidates |
| **Total (Fast Block)** | **~17ms** | No LLM call |
| **Total (Safe)** | **~17ms** | No LLM call |
| LLM call (Layer 2, N=1) | ~200ms | Single LLM evaluation |
| LLM call (Layer 2, N=3) | ~200ms | Parallel ensemble, wall-clock same as N=1 |
| **Total (Deep Path)** | **~222ms** | Includes LLM ensemble |

### 9.2 Accuracy Benchmarks

| Category | Accuracy | Notes |
|----------|----------|-------|
| Benign (correctly allowed) | 100% | Zero false positives on benign queries |
| System prompt extraction | 75% | Strong sparse keyword detection |
| Indirect injection | 55% | Benefits from cross-encoder signals |
| Direct injection | 50% | Mixed; novel variants harder |
| Refusal bypass | 45% | Subtle; relies on LLM adjudication |
| Jailbreak | 35% | Most diverse attack surface |
| **Balanced Score** | **76%** | Macro-average across categories |

### 9.3 Resource Requirements

| Component | Memory | GPU | Notes |
|-----------|--------|-----|-------|
| BGE-large-en-v1.5 | ~1.3GB | Optional | Primary embedding model |
| BGE-M3 (ensemble) | ~1.3GB | Optional | Multilingual ensemble member |
| MxBai-embed-large-v1 | ~1.3GB | Optional | Ensemble member |
| Cross-encoder | ~80MB | Optional | Pre-computed; not needed at query time |
| Qdrant | ~4GB | No | Vector database (693K points, 1024-dim) |
| LLM (remote) | 0 | No | API call to xiaomi/mimo-v2.5 |
| Phase 3 models | ~50MB | No | Logistic/Neural/Attention weights |

---

## 10. Deployment

### 10.1 Environment Configuration

```bash
# Vector database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                    # optional, for cloud Qdrant

# LLM provider
GUARDRAILER_API_KEY=your-api-key
GUARDRAILER_API_BASE=https://api.xiaomimimo.com/v1
GUARDRAILER_MODEL=xiaomi/mimo-v2.5

# Embedding configuration
GUARDRAILER_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
GUARDRAILER_EMBEDDING_MODE=single          # single | average | late | max_sim
GUARDRAILER_EMBEDDING_ENSEMBLE=            # comma-separated model names for ensemble

# Engine configuration
GUARDRAILER_COLLECTION=guardrailer_security
GUARDRAILER_HOST=0.0.0.0
GUARDRAILER_PORT=8090

# Ensemble LLM configuration
GUARDRAILER_ENSEMBLE_SIZE=3               # number of parallel LLM calls
GUARDRAILER_ENSEMBLE_THRESHOLD=2          # majority vote threshold
```

### 10.2 Startup Sequence

```bash
# 1. Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v ./qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# 2. Ingest pre-computed data (one-time, or after dataset update)
python3 guardrailer_security/ingest_precomputed.py \
  --input-dir guardrailer_output/ --batch-size 128

# 3. Start the engine
python3 guardrailer_security/security_engine.py

# 4. Verify health
curl http://localhost:8090/health

# 5. Run benchmark
python3 guardrailer_security/pint_benchmark_pipeline.py \
  --url http://localhost:8090 \
  --dataset guardrailer_security/guardrailer_benchmark_final.yaml
```

### 10.3 Production Considerations

**Scaling:**
- Horizontal: Multiple FastAPI instances behind a load balancer, shared Qdrant cluster
- Vertical: Qdrant benefits from RAM; embedding models benefit from GPU
- LLM: Rate limiting and circuit breaker patterns for API dependencies

**Reliability:**
- Qdrant: Replication factor ≥2 for production
- LLM fallback: Signal-agreement heuristic when LLM is unavailable
- Graceful degradation: System continues with reduced accuracy if Phase 3 models fail to load

**Monitoring:**
- Latency percentiles (p50, p95, p99) per layer
- FP/FN rates from feedback loop
- Qdrant collection health and index status
- LLM API error rates and latency

---

## 11. Attack Taxonomy

### 11.1 Categories

| Category | Description | Detection Signals |
|----------|-------------|-------------------|
| `jailbreak` | Bypass safety via roleplay, personas, hypotheticals | Dense similarity, n-gram overlap, LLM |
| `direct_injection` | Override/prepend new instructions explicitly | Sparse keywords, dense similarity, cross-encoder |
| `indirect_injection` | Hidden instructions in external content | Cross-encoder, centroid, LLM |
| `system_prompt_extraction` | Reveal system prompt or hidden instructions | Sparse keywords, token frequency, LLM |
| `refusal_bypass` | Make model comply with harmful requests | LLM (primary), sparse keywords |
| `benign_control` | Legitimate non-malicious query | Low composite score, benign pattern penalty |

### 11.2 Attack Techniques

| Technique | Description | Prevalence |
|-----------|-------------|------------|
| `base64_encoding` | Payload encoded in Base64/Base32/Hex | Medium |
| `virtualization_roleplay` | "Pretend you are DAN/unrestricted AI" | High |
| `hypothetical_scenario` | "In a hypothetical world where..." | Medium |
| `payload_splitting` | Attack split across multiple messages | Low |
| `few_shot_override` | Examples that override system instructions | High |

### 11.3 Pattern Detection Layers

**Sparse Keywords (54):** Direct lexical matching for known attack phrases.

**Attack Patterns (43):** High-confidence patterns that receive boost scoring.

**Subtle Patterns (14):** Lower-confidence extraction attempts; boosted only when no direct attack patterns present.

**Benign Patterns (34):** Common legitimate phrases; penalize composite score when no attack signals detected.

**N-gram Overlap (42 phrases):** Bigram/trigram matching against known attack corpus.

---

## 12. Security Considerations

### 12.1 Input Validation
- Query length: 1–10,000 characters (Pydantic enforced)
- Encoding detection handled by `crypto_agent` sub-project (17 encoding types)
- Multi-layer decode (up to 5 layers deep) for obfuscated payloads

### 12.2 API Security
- API key authentication for Qdrant and LLM endpoints
- CORS configured for production (currently `*` for development)
- Rate limiting recommended at reverse proxy layer

### 12.3 Data Privacy
- Feedback data stored locally in JSONL (not transmitted externally)
- Qdrant runs locally or in private cloud
- LLM calls contain only retrieved context + query (no full system prompts logged)
- Prompt text truncated to 2000 chars in stored payloads

### 12.4 Adversarial Robustness
- Multi-signal fusion prevents single-vector adversarial attacks
- Ensemble LLM voting reduces individual model manipulation
- Continuous feedback loop enables rapid response to new attack vectors
- Hard negative mining strengthens decision boundary over time

---

## 13. Future Architecture Directions

### 13.1 Near-Term (Phase 4)
- **Streaming evaluation** — Real-time prompt evaluation for conversational agents
- **Multi-turn context** — Evaluate conversation history, not just single prompts
- **Semantic hash index** — Exact-match deduplication for known attack payloads

### 13.2 Mid-Term (Phase 5)
- **On-device inference** — Distilled models for edge deployment
- **Federated learning** — Privacy-preserving model updates across deployments
- **Red-team automation** — Adversarial prompt generation for continuous evaluation

### 13.3 Long-Term (Phase 6)
- **Formal verification** — Provable bounds on detection recall/precision
- **Multi-modal guardrails** — Extend to image, audio, and code injection
- **Autonomous alignment** — Self-improving guardrails via constitutional AI methods
