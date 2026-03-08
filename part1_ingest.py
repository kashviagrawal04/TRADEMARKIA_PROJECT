"""
Part 1: Embedding & Vector Database Setup
==========================================
Loads the 20 Newsgroups dataset, cleans it deliberately, embeds it using
a sentence-transformer model, and persists it in ChromaDB for filtered retrieval.

Design decisions (justified here as instructed):
- Model: 'all-MiniLM-L6-v2'  — 384-dim embeddings, fast, strong on short-to-medium
  English text. Better recall/speed tradeoff than larger models for a newsgroup corpus
  where posts average ~200 tokens.
- Cleaning: We strip email headers (From:, Subject:, etc.), quoted reply blocks (lines
  starting with ">"), PGP blocks, and excess whitespace. Headers are metadata noise;
  quoted lines duplicate content already embedded elsewhere in the corpus.
- Vector store: ChromaDB — file-backed, no separate server, supports metadata
  filtering and cosine distance natively. Perfect fit for a self-contained submission.
- We cap document text at 512 tokens (truncated by the tokenizer) since MiniLM's
  max sequence length is 512; anything beyond that is silently truncated anyway.
- We skip documents shorter than 50 characters after cleaning — they carry no signal.
"""

import re
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR  = "./chroma_db"
EMBED_CACHE = "./embeddings/embeddings.npz"
MIN_DOC_LEN = 50   # characters; shorter docs are noise


# ── Text cleaning ──────────────────────────────────────────────────────────────
_HEADER_RE     = re.compile(r"^(From|Subject|Organization|Lines|Reply-To|"
                             r"Nntp-Posting-Host|X-Newsreader|Message-ID|"
                             r"References|Date|Path|Newsgroups|Xref):.*$",
                             re.MULTILINE | re.IGNORECASE)
_QUOTE_RE      = re.compile(r"^>.*$", re.MULTILINE)
_PGP_RE        = re.compile(r"-----BEGIN PGP.*?-----END PGP[^-]*-----",
                             re.DOTALL)
_URL_RE        = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE      = re.compile(r"\S+@\S+")
_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_text(text: str) -> str:
    """
    Strip metadata noise from a raw newsgroup post.
    Order matters: remove PGP blocks first (they span multiple lines),
    then per-line patterns, then normalise whitespace.
    """
    text = _PGP_RE.sub("", text)
    text = _HEADER_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ── Load & clean dataset ───────────────────────────────────────────────────────
def load_dataset():
    """
    Fetch all 20 categories (train + test merged) so we have a richer corpus
    for clustering. remove_=['headers','footers','quotes'] is sklearn's own
    cleaner — we layer our regex on top for finer control.
    """
    print("Loading 20 Newsgroups dataset …")
    raw = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),  # sklearn baseline strip
        random_state=42,
    )

    docs, labels, label_names, doc_ids = [], [], [], []
    skipped = 0
    for i, (text, label) in enumerate(zip(raw.data, raw.target)):
        cleaned = clean_text(text)
        if len(cleaned) < MIN_DOC_LEN:
            skipped += 1
            continue
        docs.append(cleaned)
        labels.append(int(label))
        label_names.append(raw.target_names[label])
        doc_ids.append(f"doc_{i}")

    print(f"Loaded {len(docs):,} documents  |  skipped {skipped:,} (too short)")
    return docs, labels, label_names, doc_ids, raw.target_names


# ── Embed documents ────────────────────────────────────────────────────────────
def embed_documents(docs, cache_path=EMBED_CACHE):
    """
    Embed with SentenceTransformer.  Results are cached to disk so re-runs
    don't re-embed ~18k documents unnecessarily.
    Batch size 64 fits comfortably in CPU RAM; increase if GPU available.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path} …")
        data = np.load(cache_path)
        return data["embeddings"]

    print(f"Embedding {len(docs):,} documents with '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-norm → cosine sim = dot product
    )
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Embeddings saved → {cache_path}")
    return embeddings


# ── Persist to ChromaDB ────────────────────────────────────────────────────────
def build_vector_store(docs, embeddings, labels, label_names, doc_ids):
    """
    Upsert all documents into ChromaDB.
    Metadata stored per document:
      - label      : integer category id (for filtered retrieval)
      - category   : human-readable category name
    We use cosine distance (via normalised embeddings + dot product space).
    ChromaDB collection is persistent; subsequent runs skip re-insertion.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop & recreate so this script is idempotent
    try:
        client.delete_collection("newsgroups")
    except Exception:
        pass

    collection = client.create_collection(
        name="newsgroups",
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB upsert limit: batch to avoid memory spikes
    BATCH = 500
    for start in tqdm(range(0, len(docs), BATCH), desc="Upserting to ChromaDB"):
        end = min(start + BATCH, len(docs))
        collection.add(
            ids=doc_ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=docs[start:end],
            metadatas=[
                {"label": labels[i], "category": label_names[i]}
                for i in range(start, end)
            ],
        )

    print(f"Vector store built  |  {collection.count():,} documents in ChromaDB")
    return collection


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    docs, labels, label_names, doc_ids, target_names = load_dataset()
    embeddings = embed_documents(docs)

    # Save processed data for Part 2 (clustering)
    os.makedirs("./data", exist_ok=True)
    with open("./data/processed.pkl", "wb") as f:
        pickle.dump({
            "docs": docs,
            "labels": labels,
            "label_names": label_names,
            "doc_ids": doc_ids,
            "target_names": list(target_names),
        }, f)
    print("Processed corpus saved → ./data/processed.pkl")

    build_vector_store(docs, embeddings, labels, label_names, doc_ids)
    print("\n✅  Part 1 complete.")


if __name__ == "__main__":
    main()
