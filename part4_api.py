"""
Part 4: FastAPI Service (Enhanced)
====================================
New additions vs baseline:
  1. GET  /           — serves the live demo UI (static/index.html)
  2. POST /explain    — explains WHY a result was returned (transparency)
  3. PATCH /cache/threshold — change similarity threshold at runtime
  4. Query logging    — every query written to query_log.jsonl

Endpoints:
  POST  /query             — semantic search with cache
  POST  /explain           — explain cache decision for a query
  GET   /cache/stats       — cache statistics
  DELETE /cache            — flush cache
  PATCH /cache/threshold   — update threshold without restart
  GET   /health            — health check
  GET   /                  — demo UI

Start with:
  uvicorn part4_api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import pickle
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb

from part3_cache import SemanticCache

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHROMA_DIR     = "./chroma_db"
CLUSTER_CACHE  = "./data/clustering_results.pkl"
DEFAULT_THRESH = 0.85
TOP_K_RESULTS  = 5
LOG_FILE       = "./query_log.jsonl"
STATIC_DIR     = "./static"


def log_query(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({**entry, "ts": time.time()}) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection("newsgroups")
        print(f"ChromaDB: {collection.count():,} documents")
    except Exception:
        collection = None
        print("ChromaDB collection not found - run part1_ingest.py first.")

    print("Loading cluster model...")
    cluster_model = None
    if os.path.exists(CLUSTER_CACHE):
        with open(CLUSTER_CACHE, "rb") as f:
            cluster_model = pickle.load(f)
        print(f"Cluster model: k={cluster_model['k']}")
    else:
        print("No cluster model - run part2_clustering.py first.")

    cache = SemanticCache(
        threshold=DEFAULT_THRESH,
        embed_model=embed_model,
        cluster_model=cluster_model,
    )

    app.state.embed_model   = embed_model
    app.state.collection    = collection
    app.state.cluster_model = cluster_model
    app.state.cache         = cache

    print("Service ready - visit http://localhost:8000")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Trademarkia Semantic Search",
    description="Semantic search over 20 Newsgroups with fuzzy cluster cache",
    version="2.0.0",
    lifespan=lifespan,
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: float
    result: str
    dominant_cluster: int

class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

class ExplainResponse(BaseModel):
    query: str
    dominant_cluster: int
    cluster_distribution: list
    candidates_checked: int
    threshold: float
    best_similarity: Optional[float]
    decision: str
    reasoning: str

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


def _retrieve_from_chroma(query, collection, embed_model):
    if collection is None:
        return "ChromaDB not available - run part1_ingest.py first."
    vec = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[vec],
        n_results=TOP_K_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    lines = [f"Top {len(docs)} results for: \"{query}\"\n"]
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        sim = 1 - dist
        lines.append(
            f"[{rank}] Category: {meta.get('category','unknown')} | Similarity: {sim:.3f}\n"
            f"    {doc[:300].replace(chr(10),' ')}...\n"
        )
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = Path(STATIC_DIR) / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>UI not found. Place static/index.html in project root.</h2>")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    cache: SemanticCache = app.state.cache
    cached_entry, sim_score = cache.lookup(body.query)

    if cached_entry is not None:
        log_query({"query": body.query, "cache_hit": True,
                   "matched_query": cached_entry.query,
                   "similarity_score": sim_score,
                   "dominant_cluster": cached_entry.dominant_cluster})
        return QueryResponse(
            query=body.query, cache_hit=True,
            matched_query=cached_entry.query,
            similarity_score=round(sim_score, 4),
            result=cached_entry.result,
            dominant_cluster=cached_entry.dominant_cluster,
        )

    result = _retrieve_from_chroma(body.query, app.state.collection, app.state.embed_model)
    entry  = cache.store(body.query, result)
    log_query({"query": body.query, "cache_hit": False,
               "similarity_score": sim_score,
               "dominant_cluster": entry.dominant_cluster})
    return QueryResponse(
        query=body.query, cache_hit=False,
        matched_query=None,
        similarity_score=round(sim_score, 4),
        result=result,
        dominant_cluster=entry.dominant_cluster,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(body: QueryRequest):
    """
    Explains WHY the cache made its decision.
    Shows cluster assignment, how many candidates were checked,
    best similarity found, and a plain-English reasoning string.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    cache: SemanticCache = app.state.cache
    embedding   = cache._embed_query(body.query)
    membership  = cache._cluster_membership(embedding)
    dom_cluster = int(np.argmax(membership))

    candidates   = cache._cluster_index.get(dom_cluster, [])
    n_candidates = len(candidates)

    best_sim = None
    if candidates:
        best_sim = max(
            float(np.dot(embedding, cache._entries[q].embedding))
            for q in candidates
        )

    threshold = cache.threshold
    if best_sim is None:
        decision  = "MISS"
        reasoning = (
            f"No cached queries in cluster {dom_cluster} yet. "
            f"This is the first query mapped to this cluster."
        )
    elif best_sim >= threshold:
        decision  = "HIT"
        reasoning = (
            f"Best match ({best_sim:.3f}) >= threshold ({threshold}). "
            f"Checked {n_candidates} candidate(s) in cluster {dom_cluster}, "
            f"skipped {len(cache._entries) - n_candidates} entries in other clusters. "
            f"Result served from cache."
        )
    else:
        decision  = "MISS"
        reasoning = (
            f"Best match ({best_sim:.3f}) < threshold ({threshold}). "
            f"Checked {n_candidates} candidate(s) in cluster {dom_cluster}. "
            f"Not similar enough — will recompute and store."
        )

    return ExplainResponse(
        query=body.query,
        dominant_cluster=dom_cluster,
        cluster_distribution=[round(float(v), 4) for v in membership],
        candidates_checked=n_candidates,
        threshold=threshold,
        best_similarity=round(best_sim, 4) if best_sim is not None else None,
        decision=decision,
        reasoning=reasoning,
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    return CacheStats(**app.state.cache.stats)


@app.delete("/cache")
async def flush_cache():
    app.state.cache.flush()
    return {"message": "Cache flushed.", "stats": app.state.cache.stats}


@app.patch("/cache/threshold")
async def update_threshold(body: ThresholdUpdate):
    """Change the similarity threshold at runtime without restarting."""
    old = app.state.cache.threshold
    app.state.cache.threshold = body.threshold
    return {"message": f"Threshold updated: {old} -> {body.threshold}",
            "old_threshold": old, "new_threshold": body.threshold}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cache_entries": len(app.state.cache),
        "threshold": app.state.cache.threshold,
        "chroma_available": app.state.collection is not None,
        "cluster_model_available": app.state.cluster_model is not None,
    }