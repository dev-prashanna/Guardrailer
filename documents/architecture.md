# Guardrailer v2: Multi-Signal RAG Security Engine

## Architecture Document

| Version | Date | Status |
|---------|------|--------|
| 2.0 | 2026-07-29 | Active |

---

## 1. System Overview

Guardrailer is a multi-signal RAG-based prompt security evaluation system. It detects jailbreaks, prompt injections, and adversarial prompts using a cascaded architecture that combines pre-computed scoring signals with LLM-as-judge evaluation.

```
                        ┌─────────────────────────────┐
                        │      User Prompt             │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   BGE-large-en-v1.5          │
                        │   Dense Encoding (1024-dim)  │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                  ┌────────────────────────────────────────┐
                  │     Qdrant Hybrid Search (12 candidates)│
                  │     0.65×dense + 0.35×idf_sparse       │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │     Multi-Signal Re-Ranking (top 3)    │
                  │     6-signal composite scoring          │
                  └──────────────────┬─────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  LAYER 1     │ │  LAYER 2     │ │  SAFE        │
           │  Fast Block  │ │  Deep Path   │ │  Allow       │
           │  (<15ms)     │ │  (LLM eval)  │ │              │
           │  composite   │ │  composite   │ │  composite   │
           │  >= 0.85     │ │  0.60-0.85   │ │  < 0.60      │
           └──────────────┘ └──────┬───────┘ └──────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  LLM Evaluator        │
                        │  xiaomi/mimo-v2.5     │
                        │  (OpenAI-compatible)  │
                        └──────────────────────┘
```

---

## 2. Scoring Architecture

### 2.1 Composite Scoring Formula

Every retrieved candidate is scored using 6 signals combined into a single composite score:

```
composite = 0.40 × dense_score
          + 0.20 × sparse_idf_score
          + 0.15 × centroid_score
          + 0.15 × cross_encoder_score
          + 0.05 × uniqueness_score
          + 0.05 × length_norm_score
```

### 2.2 Signal Definitions

| Signal | Weight | Computation | When Computed | Query Cost |
|--------|--------|-------------|---------------|------------|
| `dense_score` | 0.40 | Cosine similarity via BGE-large-en-v1.5 | Ingestion (embedding) + Query (encode) | ~10ms (encode) |
| `sparse_idf_score` | 0.20 | IDF-weighted BM25 keyword match | Ingestion (IDF stats) | ~0.1ms |
| `centroid_score` | 0.15 | Max cosine distance to category centroids | Ingestion (centroid computation) | ~0.5ms (6 dot products) |
| `cross_encoder_score` | 0.15 | Cross-encoder relevance to category description | Ingestion (1M predictions) | 0ms (pre-computed) |
| `uniqueness_score` | 0.05 | Inverse mean distance to k nearest neighbors | Ingestion (k-NN graph) | 0ms (pre-computed) |
| `length_norm_score` | 0.05 | log1p(text_length) / log1p(avg_length) | Ingestion (stats) | ~0.01ms |

### 2.3 Signal Details

**Dense Score (0.40)**
- Model: `BAAI/bge-large-en-v1.5` (1024-dim)
- Normalized embeddings, cosine distance
- Primary semantic similarity signal

**Sparse IDF Score (0.20)**
- 35 security keywords (jailbreak, bypass, override, etc.)
- IDF weighting: `log((N+1) / (df[kw]+1)) + 1.0`
- BM25 term frequency normalization
- Replaces old binary (0/1) keyword matching

**Centroid Score (0.15)**
- Mean embedding per attack category computed at ingestion
- Categories: jailbreak, direct_injection, indirect_injection, system_prompt_extraction, refusal_bypass, benign_control
- Query scored against all centroids, maximum taken

**Cross-Encoder Score (0.15)**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Pre-computed: each document scored against its category description
- Stored as payload field, zero query-time cost

**Uniqueness Score (0.05)**
- k-NN graph (k=10) computed at ingestion
- Uniqueness = 1 / mean_distance_to_k_neighbors
- Normalized to [0, 1]
- Anti-redundancy: unique documents score higher

**Length Normalization (0.05)**
- Corrects for longer texts having lower cosine similarity
- Centers around 1.0 for average-length texts

---

## 3. Retrieval Pipeline

### 3.1 Query-Time Flow

```
1. Encode query                    ~10ms
2. Hybrid search (12 candidates)   ~5ms
   - 0.65 × dense vector
   - 0.35 × IDF-weighted sparse vector
3. Re-rank with composite score    ~2ms
   - Score all 12 candidates
   - Return top 3
4. Decision routing
   - composite >= 0.85 + malicious + critical/high → BLOCK (Layer 1)
   - 0.60 <= composite < 0.85 → LLM evaluation (Layer 2)
   - composite < 0.60 → ALLOW (Safe)
5. LLM evaluation (if Layer 2)     ~200ms
   - Top-3 context + query → LLM
   - Returns: is_malicious, category, technique, risk, reasoning
```

### 3.2 Decision Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| Fast Block | composite >= 0.85 | Instant block if is_malicious=True AND risk in (critical, high) |
| Deep Path | 0.60 <= composite < 0.85 | LLM evaluation |
| Safe | composite < 0.60 | Allow |

### 3.3 LLM Evaluator

- **Model**: `xiaomi/mimo-v2.5` (configurable via env)
- **API**: OpenAI-compatible endpoint
- **Temperature**: 0.1
- **Max tokens**: 512
- **System prompt**: "AI Security Inspector" role
- **Input**: Top-3 retrieved context + candidate query
- **Output**: JSON with is_malicious, risk_score, attack_category, attack_technique, risk_level, reasoning

The LLM is invoked only for ambiguous cases (Layer 2). It receives the top-3 most relevant attack patterns from the threat database and evaluates whether the candidate query matches those patterns.

### 3.4 LLM Fallback

If the LLM is unavailable (API key missing, network error), the system falls back to:
- Block if best composite score >= 0.60
- Set risk_level to "medium"
- Reasoning indicates LLM fallback was used

---

## 4. Ingestion Pipeline

### 4.1 Overview

The ingestion pipeline pre-computes all scoring features. It runs on Kaggle with T4 GPU acceleration, then exports files for local Qdrant upload.

```
┌────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                          │
│                    (Kaggle T4 GPU, ~20 min)                    │
└────────────────────────────────────────────────────────────────┘

Phase 1: Corpus Statistics
  - IDF per keyword: log((N+1) / (df[kw]+1)) + 1.0
  - Average document length (words)
  - Average text length (characters)

Phase 2: Dense Embeddings (GPU)
  - Model: BAAI/bge-large-en-v1.5 (1024-dim)
  - All ~1M documents encoded
  - Normalized embeddings

Phase 3: Category Centroids
  - Mean embedding per attack category
  - Normalized to unit vectors

Phase 4: k-Means Clustering (FAISS-GPU)
  - 128 clusters
  - MiniBatchKMeans on GPU

Phase 5: k-NN Graph (FAISS-GPU)
  - k=10 nearest neighbors
  - Cosine distance
  - Used for uniqueness scoring

Phase 6: Uniqueness Scores
  - Uniqueness = 1 / mean_distance_to_k_nn
  - Min-max normalized to [0, 1]

Phase 7: Cross-Encoder Pre-Scoring (GPU)
  - Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Each document scored against its category description
  - Sigmoid activation for probability

Phase 8: Export
  - corpus_meta.json (IDF, centroids, weights)
  - enhanced_payloads.parquet (all metadata + pre-computed fields)
  - dense_embeddings_float16.npy (compressed embeddings)
```

### 4.2 Per-Point Payload

Each point stored in Qdrant contains:

```json
{
  "prompt_text": "truncated to 500 chars",
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

### 4.3 Corpus Metadata (`corpus_meta.json`)

```json
{
  "keyword_idf": {"jailbreak": 2.34, "bypass": 1.89, ...},
  "total_documents": 1060988,
  "avg_doc_length": 45.2,
  "avg_text_length": 267.8,
  "category_centroids": {"jailbreak": [...], ...},
  "cluster_centers": [[...], ...],
  "scoring_weights": {
    "dense": 0.40,
    "sparse_idf": 0.20,
    "centroid": 0.15,
    "cross_encoder": 0.15,
    "uniqueness": 0.05,
    "length_norm": 0.05
  }
}
```

---

## 5. System Components

### 5.1 File Map

```
guardrailer_security/
├── scoring.py                    # Multi-signal scoring module
├── security_engine.py            # FastAPI engine (v2.0.0)
├── ingest_enhanced.py            # Full ingestion pipeline (CPU fallback)
├── ingest_precomputed.py         # Local Qdrant uploader (reads Kaggle exports)
├── corpus_meta.json              # Corpus statistics (from Kaggle)
├── feedback_processor.py         # Feedback → training samples
├── feedback_logger.py            # JSONL feedback persistence
├── test_suite.py                 # 100-prompt benchmark
├── download_and_merge.py         # Dataset preparation (HuggingFace)
├── unified_security_dataset.parquet  # Source dataset (~1M samples)
└── guardrailer_output/           # Kaggle export directory
    ├── corpus_meta.json
    ├── enhanced_payloads.parquet
    └── dense_embeddings_float16.npy

documents/
├── guardrailer_enhanced_ingest_kaggle.ipynb  # Kaggle notebook (T4 GPU)
└── architecture.md                              # This file
```

### 5.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/evaluate-prompt` | POST | Evaluate a prompt for security threats |
| `/v1/feedback` | POST | Submit false positive/negative correction |
| `/v1/stats` | GET | Engine stats, collection info, feedback stats |
| `/v1/feedback/recent` | GET | Recent feedback entries |
| `/v1/feedback/pending` | GET | Pending correction samples |
| `/health` | GET | Health check |

### 5.3 Response Schema

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
  "llm_verdict": {...}
}
```

---

## 6. Feedback Loop

```
User reports FP/FN via /v1/feedback
        │
        ▼
feedback_logger.py → feedback_data/feedback_log.jsonl
        │
        ▼
feedback_processor.py
  - Reads corrections
  - Normalizes categories/risk levels
  - Generates new embeddings
  - Upserts to Qdrant (ID offset +1,000,000)
  - Appends to unified_security_dataset.parquet
        │
        ▼
Next Kaggle ingestion run
  - Picks up new samples
  - Recomputes IDF, centroids, clusters, etc.
```

---

## 7. Deployment

### 7.1 Environment Variables

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                    # optional
GUARDRAILER_COLLECTION=guardrailer_security_enhanced
GUARDRAILER_API_KEY=your-llm-api-key
GUARDRAILER_API_BASE=https://api.xiaomimimo.com/v1
GUARDRAILER_MODEL=xiaomi/mimo-v2.5
GUARDRAILER_HOST=0.0.0.0
GUARDRAILER_PORT=8090
```

### 7.2 Startup Sequence

```bash
# 1. Run Kaggle notebook (one-time, or when dataset updates)
#    Output: guardrailer_output/

# 2. Upload pre-computed data to Qdrant
python3 guardrailer_security/ingest_precomputed.py

# 3. Start the engine
python3 guardrailer_security/security_engine.py

# 4. Run benchmark
python3 guardrailer_security/test_suite.py --url http://localhost:8090
```

### 7.3 Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
qdrant-client>=1.7.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
openai>=1.6.0
```

---

## 8. Performance Characteristics

| Metric | Layer 1 (Fast Block) | Layer 2 (Deep Path) | Safe |
|--------|---------------------|---------------------|------|
| Latency | <15ms | ~220ms (incl. LLM) | <15ms |
| Path | Composite >= 0.85 | 0.60 <= Composite < 0.85 | Composite < 0.60 |
| LLM Used | No | Yes | No |
| Accuracy | High (clear malicious) | Highest (LLM adjudicates) | High (clear benign) |

### Query-Time Cost Breakdown

| Step | Time |
|------|------|
| Dense encoding | ~10ms |
| Qdrant hybrid search | ~5ms |
| Composite re-ranking | ~2ms |
| LLM call (Layer 2 only) | ~200ms |
| **Total (Fast Block)** | **~17ms** |
| **Total (Deep Path)** | **~222ms** |
| **Total (Safe)** | **~17ms** |
