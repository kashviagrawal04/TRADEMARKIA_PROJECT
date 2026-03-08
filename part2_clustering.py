"""
Part 2: Fuzzy Clustering
=========================
Discovers the latent semantic structure of the corpus using Fuzzy C-Means (FCM).

Why fuzzy over hard clustering?
  A document about "gun legislation" genuinely belongs to both
  talk.politics.guns and talk.politics.misc. FCM produces a membership
  distribution over clusters for every document — that's the right model
  for a corpus where topics bleed into each other.

Why not GMM / LDA?
  - GMM on raw 384-dim embeddings is numerically unstable (covariance
    explosion) without heavy regularisation.
  - LDA operates on term counts, not semantic embeddings — it's the wrong
    fit here since we want to leverage the geometry already encoded by the
    transformer.
  FCM operates directly in embedding space and has one tunable parameter
  (fuzziness m) that we explore explicitly.

Cluster count decision:
  We run the elbow / FPC method over k ∈ [5, 30] and pick the k where
  the Fuzzy Partition Coefficient (FPC) starts to plateau.  The 20
  ground-truth categories have significant overlap (rec.* groups, sci.*
  groups, talk.* groups), so we expect the optimal semantic k to be
  notably lower than 20.  Evidence from the FPC curve is shown.

Dimensionality reduction:
  FCM is O(n·k·d) per iteration.  At 384 dims and ~18k docs it's slow
  but tractable.  We project to 50 dims via PCA (retaining ~85% variance)
  before clustering — this also mitigates the curse of dimensionality
  that makes high-dim distance metrics less discriminative.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import skfuzzy as fuzz

# ── Config ─────────────────────────────────────────────────────────────────────
PCA_DIMS    = 100
K_RANGE     = range(5, 26)   # candidate cluster counts
FUZZINESS_M = 2.0            # standard FCM fuzziness (explored in Part 3)
MAX_ITER    = 150
ERROR       = 1e-5
RANDOM_SEED = 42

RESULTS_DIR = "./clustering_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data():
    with open("./data/processed.pkl", "rb") as f:
        corpus = pickle.load(f)
    embeddings = np.load("./embeddings/embeddings.npz")["embeddings"]
    return corpus, embeddings


# ── PCA reduction ──────────────────────────────────────────────────────────────
def reduce_dims(embeddings, n_components=PCA_DIMS):
    """
    Reduce 384-dim embeddings to 50 dims.
    - Embeddings are already L2-normalised (from Part 1).
    - We fit PCA on the full corpus (no train/test split needed here).
    - 50 components chosen to retain ≥85% explained variance empirically.
    """
    print(f"Running PCA: {embeddings.shape[1]}→{n_components} dims …")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance retained: {var_explained:.1%}")
    return reduced, pca


# ── FPC elbow analysis ─────────────────────────────────────────────────────────
def find_optimal_k(data_T, k_range=K_RANGE):
    """
    Fuzzy Partition Coefficient (FPC) ranges [1/k, 1].
    FPC = 1  → crisp partition (bad for fuzzy, means clusters are too separated)
    FPC = 1/k → maximally fuzzy (no structure)
    We look for the 'elbow': where increasing k stops improving FPC,
    signalling we've found the natural granularity of the corpus.

    data_T: features × samples (FCM convention, transposed)
    """
    fpcs = []
    print("Scanning cluster counts for optimal k …")
    for k in tqdm(k_range):
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            data_T, c=k, m=FUZZINESS_M,
            error=ERROR, maxiter=MAX_ITER, seed=RANDOM_SEED,
        )
        fpcs.append(fpc)

    # Plot
    ks = list(k_range)
    plt.figure(figsize=(9, 4))
    plt.plot(ks, fpcs, "o-", color="#4C72B0", linewidth=2, markersize=6)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Fuzzy Partition Coefficient")
    plt.title("FPC Elbow — Choosing Optimal k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fpc_elbow.png", dpi=150)
    plt.close()
    print(f"FPC elbow plot saved → {RESULTS_DIR}/fpc_elbow.png")

    best_k = ks[np.argmax(fpcs)]
    print(f"Best k by FPC: {best_k}  (FPC={max(fpcs):.4f})")
    return best_k, fpcs


# ── Run FCM ────────────────────────────────────────────────────────────────────
def run_fcm(data_T, k, m=FUZZINESS_M):
    """
    Returns:
      centers   : k × d  cluster centroids
      membership: n × k  membership matrix (rows sum to 1)
    """
    print(f"Running Fuzzy C-Means: k={k}, m={m} …")
    centers, u, _, _, _, n_iter, fpc = fuzz.cluster.cmeans(
        data_T, c=k, m=m,
        error=ERROR, maxiter=MAX_ITER, seed=RANDOM_SEED,
    )
    membership = u.T   # n × k
    print(f"  Converged in {n_iter} iterations  |  FPC={fpc:.4f}")
    return centers, membership, fpc


# ── Analysis helpers ───────────────────────────────────────────────────────────
def hard_labels(membership):
    """Argmax cluster for each document (for analysis only)."""
    return np.argmax(membership, axis=1)


def cluster_top_docs(membership, docs, cluster_id, n=5):
    """Documents most strongly belonging to a cluster."""
    scores = membership[:, cluster_id]
    top_idx = np.argsort(scores)[::-1][:n]
    return [(scores[i], docs[i][:200]) for i in top_idx]


def boundary_docs(membership, docs, n=10):
    """
    Documents where the top-2 memberships are close — these sit on cluster
    boundaries and are semantically the most interesting/ambiguous cases.
    """
    sorted_mem = np.sort(membership, axis=1)[:, ::-1]
    margin = sorted_mem[:, 0] - sorted_mem[:, 1]   # smaller = more uncertain
    boundary_idx = np.argsort(margin)[:n]
    results = []
    for i in boundary_idx:
        top2 = np.argsort(membership[i])[::-1][:2]
        results.append({
            "doc_snippet": docs[i][:200],
            "cluster_a": int(top2[0]),
            "membership_a": float(membership[i, top2[0]]),
            "cluster_b": int(top2[1]),
            "membership_b": float(membership[i, top2[1]]),
            "margin": float(margin[i]),
        })
    return results


def category_cluster_heatmap(membership, label_names, target_names, k):
    """
    Heatmap of average membership per (ground-truth category, cluster) pair.
    This validates semantic coherence: if our clusters are meaningful,
    each row (GT category) should peak in 1–2 clusters.
    """
    n_cats = len(target_names)
    heat = np.zeros((n_cats, k))
    for cat_id in range(n_cats):
        mask = np.array(label_names) == target_names[cat_id]
        if mask.sum() > 0:
            heat[cat_id] = membership[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(k, 12), 10))
    sns.heatmap(
        heat, ax=ax,
        xticklabels=[f"C{i}" for i in range(k)],
        yticklabels=[t.replace("talk.politics.", "tp.")
                      .replace("talk.religion.", "tr.")
                      .replace("sci.", "sci.")
                      .replace("rec.", "rec.")
                      for t in target_names],
        cmap="YlOrRd", linewidths=0.3,
    )
    ax.set_title("Average Cluster Membership per Ground-Truth Category")
    ax.set_xlabel("Fuzzy Cluster")
    ax.set_ylabel("Ground-Truth Category")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/category_cluster_heatmap.png", dpi=150)
    plt.close()
    print(f"Heatmap saved → {RESULTS_DIR}/category_cluster_heatmap.png")


def membership_entropy_histogram(membership):
    """
    Entropy of membership distribution per document.
    Low entropy → document strongly belongs to one cluster (crisp).
    High entropy → document is genuinely cross-topic (fuzzy).
    """
    eps = 1e-10
    entropy = -(membership * np.log(membership + eps)).sum(axis=1)
    plt.figure(figsize=(8, 4))
    n_bins = max(5, min(60, len(np.unique(np.round(entropy, 3)))))
    plt.hist(entropy, bins=n_bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    plt.xlabel("Membership Entropy")
    plt.ylabel("Document count")
    plt.title("Per-document Membership Entropy\n"
              "(high = genuinely cross-topic, low = clearly categorised)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/membership_entropy.png", dpi=150)
    plt.close()
    print(f"Entropy histogram saved → {RESULTS_DIR}/membership_entropy.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    corpus, embeddings = load_data()
    docs        = corpus["docs"]
    label_names = corpus["label_names"]
    target_names= corpus["target_names"]

    # Reduce dimensions for FCM
    reduced, pca_model = reduce_dims(embeddings)

    # Transpose for skfuzzy convention (features × samples)
    data_T = reduced.T

    # --- Find optimal k ---
    best_k, fpcs = find_optimal_k(data_T)

    # Use best_k (or override if you want a specific value)
    K = best_k
    print(f"\nUsing K = {K} clusters for final FCM run")

    # --- Run final FCM ---
    centers, membership, fpc = run_fcm(data_T, K)

    # --- Analysis ---
    print("\n── Cluster Summary ──────────────────────────────────────────────")
    hard = hard_labels(membership)
    for cid in range(K):
        size = (hard == cid).sum()
        avg_mem = membership[hard == cid, cid].mean() if size > 0 else 0
        print(f"  Cluster {cid:2d}: {size:5,} docs  |  avg dominant membership: {avg_mem:.3f}")

    print("\n── Boundary Documents (most uncertain) ──────────────────────────")
    boundaries = boundary_docs(membership, docs, n=5)
    for b in boundaries:
        print(f"  Clusters {b['cluster_a']}({b['membership_a']:.2f}) vs "
              f"{b['cluster_b']}({b['membership_b']:.2f})  margin={b['margin']:.3f}")
        print(f"  \"{b['doc_snippet'][:120]}…\"\n")

    # Visualisations
    category_cluster_heatmap(membership, label_names, target_names, K)
    membership_entropy_histogram(membership)

    # --- Persist results for Parts 3 & 4 ---
    results = {
        "k": K,
        "centers": centers,          # k × 50  (PCA space)
        "membership": membership,    # n × k
        "fpc": fpc,
        "pca_model": pca_model,
        "hard_labels": hard,
        "doc_ids": corpus["doc_ids"],
    }
    with open("./data/clustering_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nClustering results saved → ./data/clustering_results.pkl")
    print("✅  Part 2 complete.")


if __name__ == "__main__":
    main()