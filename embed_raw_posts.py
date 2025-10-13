import os, shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sentence_transformers import SentenceTransformer
import torch

# ---------------- Settings ----------------
INPUT_PARQUET = "output/raw_posts_kafka"
BASE_DIR = "output/raw_posts_embeddings_gemma"
MODEL_ID = "google/embeddinggemma-300m"
BATCH_SIZE = 256          # safer than 512; raise if stable
MAX_SEQ_LEN = 128
NORMALIZE = False         # set True if you want unit vectors on disk

# Let tokenizers parallelize (helps feed GPU)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------- Load posts ----------------
df = pd.read_parquet(INPUT_PARQUET, columns=["did", "rkey", "text"])
df["post_id"] = df["did"].astype(str) + "/" + df["rkey"].astype(str)
df = df[df["text"].notna()]
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]

# ---------------- Load model (GPU if available) ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, device=device).eval()
try:
    model.max_seq_length = MAX_SEQ_LEN
except Exception:
    pass

# IMPORTANT: keep Float32; enable TF32 on Ampere for speed + stability
if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
# DO NOT call model.half() — FP16 caused NaNs for you

# ---------------- Encode (float32) ----------------
texts = df["text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True
).astype(np.float32)  # (N, D) float32

# ---------------- Validate & (optional) fix bad rows ----------------
finite_mask = np.isfinite(embeddings).all(axis=1)
if not finite_mask.all():
    n_bad = (~finite_mask).sum()
    print(f"Found {n_bad} rows with NaN/Inf; re-encoding just those rows in float32…")
    bad_idx = np.where(~finite_mask)[0]
    bad_texts = [texts[i] for i in bad_idx]

    # Re-encode bad rows again (still float32)
    fixed = model.encode(
        bad_texts,
        batch_size=max(64, BATCH_SIZE // 2),
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True
    ).astype(np.float32)

    embeddings[bad_idx] = fixed
    assert np.isfinite(embeddings).all(), "Still found NaN/Inf after repair"

# Optional: L2 normalize before saving (if you want cosine-ready vectors on disk)
if NORMALIZE:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nz = norms.squeeze() > 0
    embeddings[nz] = embeddings[nz] / norms[nz]

# ---------------- Save as a Parquet *directory* dataset ----------------
shutil.rmtree(BASE_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)

emb_list_arrays = [pa.array(row, type=pa.float32()) for row in embeddings]
table = pa.table({
    "post_id":  pa.array(df["post_id"].tolist()),
    "text":     pa.array(df["text"].tolist()),
    "embedding": pa.array(emb_list_arrays, type=pa.list_(pa.float32()))
})

fmt = ds.ParquetFileFormat()
opts = fmt.make_write_options(compression=os.environ.get("PARQUET_COMPRESSION", "zstd"))

ds.write_dataset(
    data=table,
    base_dir=BASE_DIR,
    format=fmt,
    file_options=opts,
    existing_data_behavior="overwrite_or_ignore"
)

print(f"✓ Wrote dataset to directory: {BASE_DIR}")
