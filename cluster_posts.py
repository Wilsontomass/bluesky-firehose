import pyarrow.dataset as ds
import numpy as np
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

BASE_DIR = "output/raw_posts_embeddings_gemma"

# ---------------- Load embeddings ----------------
print("Loading Embeddings...")
dataset = ds.dataset(BASE_DIR, format="parquet")
table = dataset.to_table(columns=["post_id", "text", "text_clean", "embedding"])

post_id = table.column("post_id").to_pylist()
texts = table.column("text").to_pylist()
texts_clean = table.column("text_clean").to_pylist()
embeddings = np.vstack(table.column("embedding").to_pylist()).astype(np.float32)

# Normalize embeddings
print("Normalizing Embeddings...")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
nonzero = norms.squeeze() > 0
embeddings[nonzero] /= norms[nonzero]

# ---------------- Dimensionality reduction ----------------
print("Dimensionality reduction...")
print("Embedding Dimension ---> 64D...")
reducer_high = umap.UMAP(n_components=64, metric="cosine")
X_64d = reducer_high.fit_transform(embeddings)

# ---------------- HDBSCAN clustering ----------------
print("HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom"
)
labels = clusterer.fit_predict(X_64d)

# ---------------- Group clusters ----------------
clusters = defaultdict(list)
for i, c in enumerate(labels):
    if c == -1:
        continue
    clusters[c].append(i)

# ---------------- Compute cluster centroids in 64D ----------------
cluster_ids = []
cluster_centroids_64d = []

for c, idxs in clusters.items():
    cluster_ids.append(c)
    cluster_centroids_64d.append(X_64d[idxs].mean(axis=0))

cluster_centroids_64d = np.vstack(cluster_centroids_64d)

# ---------------- Reduce centroids to 2D for visualization ----------------
print("Reducing cluster centroids to 2D...")
reducer_2d = umap.UMAP(
    n_components=2,
    metric="cosine",
    min_dist=0.1,
    spread=1.0,
)
X_2d_centroids = reducer_2d.fit_transform(cluster_centroids_64d)

# Map back to clusters for plotting
cluster_positions = {c: X_2d_centroids[i] for i, c in enumerate(cluster_ids)}

# ---------------- Summarizer pipeline ----------------
print("Summarizing...")
summarizer = pipeline("summarization", model="t5-small")  # lightweight, can switch to flan-t5-base

# ---------------- Prepare cluster summary ----------------
cluster_summary = []

for i, c in enumerate(cluster_ids):
    doc_idx = clusters[c]
    if len(doc_idx) == 0:
        continue

    # Use precomputed 2D centroid positions
    centroid_2d = X_2d_centroids[i]

    # High-dimensional centroid for selecting representative posts
    E_hd = X_64d[doc_idx]
    centroid_hd = E_hd.mean(axis=0)
    sim = cosine_similarity(E_hd, centroid_hd.reshape(1, -1)).ravel()
    top_idx = np.argsort(-sim)[:3]
    context_text = " ".join(texts_clean[doc_idx[j]] for j in top_idx)

    # Summarize context into a short label
    prompt = context_text
    try:
        input_len = len(prompt.split())
        summary_text = summarizer(
            prompt,
            min_length=3,
            max_new_tokens=10,
            do_sample=False,
            clean_up_tokenization_spaces=True
        )[0]["summary_text"]
    except Exception:
        # fallback to short snippets
        summary_text = ", ".join([texts_clean[doc_idx[j]][:15] for j in top_idx])

    cluster_summary.append({
        "cluster": int(c),
        "size": len(doc_idx),
        "x": float(centroid_2d[0]),
        "y": float(centroid_2d[1]),
        "summary": summary_text
    })

# ---------------- Save cluster summary ----------------
with open("clusters.pkl", "wb") as f:
    pickle.dump(cluster_summary, f)

print(f"Saved {len(cluster_summary)} clusters with summaries to clusters.pkl")




