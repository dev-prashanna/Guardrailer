# Guardrailer — Version 1

Multi-signal RAG-based prompt injection detection system with LLM evaluation.
Phase 1: Comprehensive dataset expansion and benchmarking.

---

## Phase 1 Benchmark Results

**Date:** 2026-07-30
**Engine:** Guardrailer Security Engine v2.0.0 (xiaomi/mimo-v2.5 via Groq)
**Dataset:** 5,160 samples across 6 attack categories, 15 techniques, 10 sources

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Total Samples** | 5,160 |
| **Accuracy** | 70.16% |
| **Balanced Accuracy** | 79.91% |
| **Precision** | 99.08% |
| **Recall** | 61.94% |
| **F1 Score** | 76.23% |

### Per-Category Accuracy

| Category | Samples | Accuracy | F1 | Recall |
|----------|---------|----------|-----|--------|
| Benign Control | 1,174 | **97.87%** | — | — |
| System Prompt Extraction | 266 | **78.95%** | 0.8824 | 0.7895 |
| Refusal Bypass | 198 | **75.76%** | 0.8621 | 0.7576 |
| Direct Injection | 1,174 | **73.59%** | 0.8479 | 0.7359 |
| Indirect Injection | 1,174 | **67.72%** | 0.8075 | 0.6772 |
| Jailbreak | 1,174 | **38.33%** | 0.5542 | 0.3833 |

### Per-Technique Accuracy (Top 10)

| Technique | Samples | Accuracy | F1 |
|-----------|---------|----------|-----|
| Virtualization Roleplay | 1,067 | **93.35%** | 0.9656 |
| Multi-turn Escalation | 12 | **91.67%** | 0.9565 |
| Hypothetical Scenario | 85 | **80.00%** | 0.8859 |
| Semantic Camouflage | 148 | **75.68%** | 0.8615 |
| Payload Splitting | 114 | **69.30%** | 0.8168 |
| Few-shot Override | 127 | **65.35%** | 0.7905 |
| Instruction Hierarchy Abuse | 161 | 54.04% | 0.7016 |
| Base64 Encoding | 188 | 48.40% | 0.6473 |
| Unicode Obfuscation | 130 | 30.00% | 0.4615 |
| Prompt Smuggling | 70 | 28.57% | 0.4444 |

### Robustness Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Obfuscation Resistance | **17.12%** | Critical weakness |
| Unicode Trick Resistance | **23.26%** | Critical weakness |
| Multilingual Accuracy | **85.71%** | Strong |
| Generalization Gap | **85.96%** | High variance across sources |

### Per-Difficulty Accuracy

| Difficulty | Samples | Accuracy |
|------------|---------|----------|
| Easy (0–0.3) | 4,477 | **74.22%** |
| Medium (0.3–0.6) | 277 | 45.49% |
| Hard (0.6–0.8) | 326 | 46.01% |
| Adversarial (0.8–1.0) | 80 | **23.75%** |

### Latency

| Percentile | Latency |
|------------|---------|
| Average | 359.9 ms |
| P50 | 226.1 ms |
| P95 | 1,240.9 ms |
| P99 | 2,509.9 ms |

### Layer Distribution

| Layer | Count | Description |
|-------|-------|-------------|
| deep_path | 5,011 | LLM evaluation required |
| fast_block | 42 | Instant block (composite ≥ 0.85) |
| unknown | 107 | Errors / timeouts |

---

## Key Findings

1. **Near-perfect benign detection** (97.87%) — minimal false positives
2. **Strong roleplay detection** (93.35%) — the guardrail excels at DAN/persona attacks
3. **Critical obfuscation gap** — Base64 (48.4%), ROT13 (0%), Unicode (30%), homoglyphs (20%) all bypass the guardrail
4. **Jailbreak category is weakest** (38.33%) — novel jailbreak templates evade detection
5. **Encoding layering defeats detection** (12.12%) — multi-layer encoding is nearly undetectable
6. **High precision, low recall** (99.08% / 61.94%) — when the system flags an attack, it's almost always correct, but it misses ~38% of attacks

### Priority Improvements for Phase 2

- Decode obfuscated payloads before scoring (Base64, ROT13, hex, Unicode escape)
- Expand jailbreak training data with novel template variants
- Add homoglyph normalization preprocessing
- Retrain embeddings on obfuscated attack samples
- Implement multi-language attack detection pipeline

---

## Dataset Composition

### Sources

| Source | Samples | Type |
|--------|---------|------|
| PromptInject (public HF datasets) | 4,421 | Research |
| Synthetic Novel Variants | 271 | Generated |
| Synthetic Obfuscated | 171 | Generated |
| Synthetic Paraphrases | 130 | Generated |
| Lakera PINT | 44 | Research |
| Edge Case Unicode | 86 | Generated |
| Synthetic Multi-turn | 12 | Generated |
| Edge Case Context-dependent | 11 | Generated |
| Edge Case Multi-language | 7 | Generated |
| Edge Case Ambiguous | 7 | Generated |

### Categories (Balanced)

| Category | Samples |
|----------|---------|
| Jailbreak | 1,174 |
| Direct Injection | 1,174 |
| Indirect Injection | 1,174 |
| Benign Control | 1,174 |
| System Prompt Extraction | 266 |
| Refusal Bypass | 198 |

### Techniques Covered

| Technique | Samples |
|-----------|---------|
| None (template-based) | 2,959 |
| Virtualization Roleplay | 1,067 |
| Base64 Encoding | 188 |
| Instruction Hierarchy Abuse | 161 |
| Unicode Obfuscation | 130 |
| Semantic Camouflage | 148 |
| Few-shot Override | 127 |
| Payload Splitting | 114 |
| Hypothetical Scenario | 85 |
| Prompt Smuggling | 70 |
| ROT13 Encoding | 40 |
| Encoding Layering | 33 |
| Homoglyph Substitution | 25 |
| Multi-turn Escalation | 12 |
| Language Switching | 1 |

---

## Architecture

Guardrailer uses a 3-layer cascaded detection system:

```
User Prompt
    │
    ▼
BGE-large-en-v1.5 Dense Encoding (1024-dim)
    │
    ▼
Qdrant Hybrid Search (12 candidates, 0.65×dense + 0.35×idf_sparse)
    │
    ▼
Multi-Signal Re-Ranking (top 3, 6-signal composite scoring)
    │
    ├─ composite ≥ 0.85 + malicious + critical/high → BLOCK (Layer 1, <15ms)
    ├─ 0.60 ≤ composite < 0.85 → LLM evaluation (Layer 2, ~220ms)
    └─ composite < 0.60 → ALLOW (Safe, <15ms)
```

### 6-Signal Composite Scoring

```
composite = 0.40 × dense_score           (BAAI/bge-large-en-v1.5 cosine similarity)
          + 0.20 × sparse_idf_score      (IDF-weighted BM25 keyword matching)
          + 0.15 × centroid_score        (category centroid distance)
          + 0.15 × cross_encoder_score   (ms-marco-MiniLM-L-6-v2)
          + 0.05 × uniqueness_score      (inverse mean distance to k-NN)
          + 0.05 × length_norm_score     (log1p text length normalization)
```

---

## Project Structure

```
guardrailer_security/
├── phase1_expansion/                # Phase 1: Dataset Expansion
│   ├── __init__.py
│   ├── __main__.py                  # CLI entry point
│   ├── schema.py                    # Expanded dataset schema (10 categories, 23 techniques)
│   ├── research_collectors.py       # Lakera PINT, WildJailbreak, Anthropic, OpenAI, PromptInject
│   ├── synthetic_generator.py       # Novel variants, paraphrases, multi-turn, obfuscated
│   ├── edge_cases.py                # Ambiguous, multi-language, Unicode, context-dependent
│   ├── benchmark_framework.py       # Multi-dimensional evaluation framework
│   ├── orchestrator.py              # Pipeline orchestrator
│   ├── phase1_expanded_dataset.parquet
│   ├── phase1_benchmark.yaml
│   ├── phase1_benchmark_report.json
│   └── phase1_pipeline_summary.json
│
├── security_engine.py               # Main FastAPI engine (v2.0.0)
├── scoring.py                       # Multi-signal composite scoring module
├── app.py                           # Streamlit dashboard
├── ingest_precomputed.py            # Qdrant ingestion (checkpoints, backups)
├── ingest_enhanced.py               # Full ingestion pipeline
├── download_and_merge.py            # Dataset download and merge
├── feedback_logger.py               # JSONL feedback persistence
├── feedback_processor.py            # Feedback → training samples
├── pint_benchmark_guardrailer.py    # PINT benchmark adapter
├── pint_benchmark_pipeline.py       # Full PINT benchmark pipeline
├── test_suite.py                    # 100-prompt benchmark suite
├── test_fixes.py                    # Fix verification tests
├── requirements.txt
├── .env                             # API keys (not committed)
└── corpus_meta.json                 # Pre-computed IDF, centroids, weights

crypto_agent/                        # Encoded payload decoder + classifier
├── decoder/                         # Multi-layer encoding decoder (20+ types)
├── classifier/                      # Rule-based + LLM threat classification
├── rag/                             # Semantic search retriever
└── app.py                           # Streamlit dashboard

documents/
├── architecture.md                  # Full system architecture (v2.0)
└── guardrailer_enhanced_ingest_kaggle.ipynb  # Kaggle ingestion notebook
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Qdrant running on `localhost:6333`
- GPU recommended for ingestion

### Install Dependencies

```bash
pip install -r guardrailer_security/requirements.txt
```

### Configure Environment

Create `guardrailer_security/.env`:

```env
GUARDRAILER_API_KEY=your-api-key
GUARDRAILER_API_BASE=https://api.groq.com/openai/v1
GUARDRAILER_MODEL=llama-3.3-70b-versatile
QDRANT_URL=http://localhost:6333
```

### Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v ./qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

### Ingest Data

```bash
python3 guardrailer_security/reassemble_embeddings.py ~/Downloads/
python3 guardrailer_security/ingest_precomputed.py --input-dir ~/Downloads --batch-size 128
```

### Start Engine

```bash
python3 guardrailer_security/security_engine.py
```

### Run Phase 1 Benchmark

```bash
# Full pipeline (collect + generate + benchmark)
python3 -m guardrailer_security.phase1_expansion --mode full

# Benchmark existing dataset
python3 -m guardrailer_security.phase1_expansion --mode benchmark \
    --dataset guardrailer_security/phase1_expansion/phase1_expanded_dataset.parquet
```

### Run Original Benchmark

```bash
python3 guardrailer_security/pint_benchmark_guardrailer.py \
    --url http://localhost:8090 \
    --dataset guardrailer_security/guardrailer_benchmark_final.yaml
```

---

## API

### `POST /v1/evaluate-prompt`

```json
{
  "query": "string"
}
```

Response:

```json
{
  "query": "string",
  "is_blocked": true,
  "layer": "deep_path",
  "composite_score": 0.93,
  "similarity_score": 0.91,
  "attack_category": "system_prompt_extraction",
  "attack_technique": "virtualization_roleplay",
  "risk_level": "critical",
  "reasoning": "...",
  "latency_ms": 45.2,
  "retrieved_context": [...],
  "llm_verdict": {...}
}
```

### `POST /v1/feedback`

Submit false positive/negative corrections.

### `GET /v1/stats`

Engine stats, collection info, feedback statistics.

### `GET /health`

Returns `{"status": "ok"}`.

---

## License

MIT
