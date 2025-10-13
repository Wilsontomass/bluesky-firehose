import os, shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sentence_transformers import SentenceTransformer
import torch

# ---------------- Load posts ----------------
INPUT_PARQUET = "output/raw_posts_kafka"
BASE_DIR = "output/raw_posts_embeddings_gemma"

df = pd.read_parquet(INPUT_PARQUET, columns=["did", "rkey", "text"])
df["post_id"] = df["did"].astype(str) + "/" + df["rkey"].astype(str)
# optional hygiene
df = df[df["text"].notna()]
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]

# ---------------- Load model (GPU if available) ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300m", device=device).eval()
try:
    model.max_seq_length = 128
except Exception:
    pass
if device == "cuda":
    try:
        model = model.half()           # FP16 on 30xx GPUs
    except Exception:
        pass

# ---------------- Encode ----------------
texts = df["text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True
).astype(np.float32)  # now embeddings is np.ndarray [n, d] float32

# ---------------- Save as a Parquet *directory* dataset ----------------
# Clean output dir
shutil.rmtree(BASE_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)

# Build a list< float32 > column explicitly with PyArrow
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
