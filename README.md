# Trademarkia AI/ML Engineer Task — Semantic Search System

A lightweight semantic search system over the 20 Newsgroups dataset with fuzzy clustering and a from-scratch semantic cache.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                          │
│  POST /query  ─► SemanticCache.lookup()                         │
│                       │                                         │
│                  HIT ─┤─ MISS                                   │
│                   │   └──► ChromaDB.query() ──► cache.store()   │
│                   │                                             │
│              return cached result          return new result     │
└─────────────────────────────────────────────────────────────────┘

SemanticCache
  ├── _entries: {query → CacheEntry(embedding, result, cluster)}
  └── _cluster_index: {cluster_id → [query, query, ...]}
              ↑
              └─ Built from Fuzzy C-Means clustering (Part 2)
                 Narrows lookup from O(N) → O(N/k)
```

---

## Quick Start

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline in order

```bash
# Part 1: Download dataset, clean, embed, persist to ChromaDB
python part1_ingest.py

# Part 2: Fuzzy clustering + analysis plots
python part2_clustering.py

# Part 3: Threshold analysis (optional standalone demo)
python part3_cache.py

# Part 4: Start the API
uvicorn part4_api:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

```bash
# Semantic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "what are the health risks of smoking?"}'

# Same meaning, different words — should be a cache HIT
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how does cigarette smoking affect your health?"}'

# Cache stats
curl http://localhost:8000/cache/stats

# Flush cache
curl -X DELETE http://localhost:8000/cache
```

---

## Docker (Bonus)

```bash
# Build and start
docker-compose up --build

# Or with plain Docker
docker build -t semantic-search .
docker run -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/embeddings:/app/embeddings \
  semantic-search
```

> **Note**: Run the ingestion and clustering scripts **before** building the Docker image, or mount the data volumes as shown above so the container can access pre-computed artefacts.

---

## Design Decisions

### Part 1 — Embedding & Vector Store

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, strong semantic recall, fast on CPU, designed for sentence-level similarity |
| Vector store | ChromaDB (persistent) | File-backed, no separate server, native cosine distance, metadata filtering |
| Cleaning | Strip headers, quoted lines, PGP blocks, URLs, emails | These are metadata/formatting noise, not content signal |
| Min doc length | 50 characters | Sub-50-char posts carry no meaningful semantic content after cleaning |

### Part 2 — Fuzzy Clustering

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | Fuzzy C-Means (FCM) | Produces membership distributions, not hard labels — correct model for overlapping topics |
| Dim reduction | PCA → 50 dims | Retains ~85% variance, mitigates curse of dimensionality, speeds up FCM |
| k selection | FPC elbow | Objective, corpus-driven evidence for cluster count |
| Fuzziness m | 2.0 | Standard FCM default; explored via FPC; higher m → fuzzier (more overlap) |

### Part 3 — Semantic Cache

| Decision | Choice | Rationale |
|---|---|---|
| Similarity metric | Cosine similarity (dot product on unit-norm vectors) | Angle-based, invariant to vector magnitude, standard for sentence embeddings |
| Default threshold | 0.85 | Catches true paraphrases (avg sim ≈ 0.88–0.92) while rejecting topically-related but distinct queries |
| Lookup strategy | Cluster-narrowed: compare only within dominant cluster | Reduces comparisons from O(N) to O(N/k); cluster structure from Part 2 does real work |
| Data structures | Dict for entries, Dict[int→List] for cluster index | O(1) entry access, O(N/k) lookup, O(1) store |

### Threshold Analysis (key insight)

The threshold is the single most consequential tunable in the cache:

- **0.50–0.70**: Over-aggressive. Queries sharing a topic cluster match even when they ask different questions. High throughput, low result accuracy.
- **0.75–0.85**: Sweet spot. True paraphrases (same meaning, different words) hit; semantically adjacent but distinct queries miss and recompute correctly.  
- **0.90–1.00**: Conservative. Even slight lexical variation causes misses. Safe when result correctness is critical and computation is cheap.

Run `python part3_cache.py` to see the full threshold analysis plot.

---

## Output Files

After running all scripts:

```
chroma_db/                  # Persistent ChromaDB vector store
data/
  processed.pkl             # Cleaned corpus + metadata
  clustering_results.pkl    # FCM model, membership matrix, PCA transform
embeddings/
  embeddings.npz            # Cached document embeddings (384-dim)
clustering_results/
  fpc_elbow.png             # FPC vs k — cluster count selection evidence
  category_cluster_heatmap.png  # Ground-truth category vs fuzzy cluster
  membership_entropy.png    # Per-document membership entropy distribution
  threshold_analysis.png    # Cache threshold hit/FP rate analysis
```

---

## API Reference

### `POST /query`

```json
// Request
{ "query": "what causes global warming?" }

// Response
{
  "query": "what causes global warming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.72,
  "result": "Top 5 results for: ...",
  "dominant_cluster": 4
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

```json
{ "message": "Cache flushed successfully.", "stats": { ... } }
```

### `GET /health`

```json
{
  "status": "ok",
  "cache_entries": 12,
  "chroma_available": true,
  "cluster_model_available": true
}
```
