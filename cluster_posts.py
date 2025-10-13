import pyarrow.dataset as ds
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import re


BASE_DIR = "output/raw_posts_embeddings_gemma"

# Load the dataset
dataset = ds.dataset(BASE_DIR, format="parquet")
table = dataset.to_table(columns=["post_id", "text", "embedding"])

# Pull columns
post_id = table.column("post_id").to_pylist()
texts   = table.column("text").to_pylist()

# Efficiently turn list<float32> → (N, D) float32 NumPy
embeddings = np.vstack(table.column("embedding").to_pylist()).astype(np.float32)

# --- Sanity check & clean ---
finite_mask = np.isfinite(embeddings).all(axis=1)
if not finite_mask.all():
    n_bad = (~finite_mask).sum()
    print(f"Found {n_bad} rows with NaN/Inf in embeddings; dropping them.")
    # keep only good rows across all aligned arrays
    embeddings = embeddings[finite_mask]
    post_id = [pid for i, pid in enumerate(post_id) if finite_mask[i]]
    texts   = [tx  for i, tx  in enumerate(texts)   if finite_mask[i]]

# Optional: if you'd rather replace instead of drop:
# embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

print("After cleaning:", embeddings.shape, embeddings.dtype)

# --- Safe L2 normalize (cosine-ready) ---
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
nonzero = norms.squeeze() > 0
# avoid division-by-zero
embeddings[nonzero] = embeddings[nonzero] / norms[nonzero]
# rows with zero norm stay zero; they will cluster as noise/outliers

X = embeddings  # already L2-normalized for nonzero rows
k = 20  # choose/tune
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X)

# ---------------- Label clusters from texts ----------------
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# 1) Build a TF-IDF model on ALL texts
# If your texts are mostly English, set stop_words="english".
# If they're mixed/Swedish, leave None (or provide your own list).
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),        # unigrams + bigrams label better
    max_features=50000,        # tune as needed
    min_df=2                   # ignore ultra-rare terms
)
tfidf = vectorizer.fit_transform(texts)  # shape: (N_docs, Vocab)

# 2) Group doc indices by cluster
clusters = defaultdict(list)
for i, c in enumerate(labels):
    clusters[c].append(i)

terms = vectorizer.get_feature_names_out()

def top_terms_for_cluster(doc_indices, top_k=6):
    # average TF-IDF vector for docs in the cluster, then take top terms
    sub = tfidf[doc_indices]                # (n_i, V)
    mean_vec = sub.mean(axis=0)             # (1, V) sparse
    mean_vec = mean_vec.A1                  # to 1D np array
    top_idx = mean_vec.argsort()[::-1][:top_k]
    return [terms[j] for j in top_idx if mean_vec[j] > 0]

def representative_docs(doc_indices, center_embedding, k=3):
    # closest texts to the centroid in *embedding* space (better semantic reps)
    E = X[doc_indices]                      # normalized embeddings
    d = cosine_distances(E, center_embedding[None, :]).ravel()
    order = np.argsort(d)[:k]
    return [(doc_indices[idx], float(d[order[i]])) for i, idx in enumerate(order)]

# 3) Print labels + sample representative posts for each cluster
for c in sorted(clusters.keys()):
    doc_idx = clusters[c]
    if len(doc_idx) == 0:
        continue

    # label from TF-IDF
    label_terms = top_terms_for_cluster(doc_idx, top_k=6)
    label_str = ", ".join(label_terms) if label_terms else "(no strong terms)"

    # centroid in embedding space (already L2-normalized)
    centroid = X[doc_idx].mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-12)

    reps = representative_docs(doc_idx, centroid, k=3)

    print(f"\n=== Cluster {c} | {len(doc_idx)} posts ===")
    print(f"Label: {label_str}")
    for ridx, dist in reps:
        raw = texts[ridx][:160]
        snippet = re.sub(r"\s+", " ", raw).strip()
        print(f"  • [{dist:.3f}] {snippet}")

