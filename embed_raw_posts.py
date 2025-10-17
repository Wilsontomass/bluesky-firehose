import os
import shutil
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from sentence_transformers import SentenceTransformer

# ---------------- Settings ----------------
INPUT_PARQUET = "output/raw_posts_kafka"
BASE_DIR = "output/raw_posts_embeddings_gemma"
MODEL_ID = "google/embeddinggemma-300m"
BATCH_SIZE = 256
MAX_SEQ_LEN = 128
NORMALIZE = False

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------- NSFW / filtering ----------------
NSFW_KEYWORDS = set([
    "porn","sex","nude","sexy","fuck","cock","cum","blowjob",
    "dick","tits","ass","horny","slut","nsfw","onlyfans"
])

MULTILINGUAL_STOPWORDS = set([
    "the","and","a","of","in","to","is","it","for","on",
    "que","de","le","la","el","en","und","der","die"
])

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text_words = [w for w in text.split() if w not in MULTILINGUAL_STOPWORDS]
    return " ".join(text_words)

def is_meaningful(text):
    if not re.search(r"\w", text):
        return False
    tokens = text.split()
    if len(tokens) == 0:
        return False
    url_count = sum(bool(URL_PATTERN.match(tok)) for tok in tokens)
    if url_count / len(tokens) > 0.5:
        return False
    return True

def is_nsfw(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in NSFW_KEYWORDS)

# ---------------- Load posts ----------------
df = pd.read_parquet(INPUT_PARQUET, columns=["did", "rkey", "text"])
print(f"Before filtering: {len(df)} posts.")

df["post_id"] = df["did"].astype(str) + "/" + df["rkey"].astype(str)
df = df[df["text"].notna()]
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df["text_clean"] = df["text"].apply(preprocess)

# ---------------- Preprocessing ----------------
df = df[df["text_clean"].apply(is_meaningful)]
df = df[~df["text_clean"].apply(is_nsfw)]
df = df[df["text_clean"].str.len() > 0]

print(f"After filtering: {len(df)} posts remain.")

# ---------------- Load model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, device=device,truncate_dim=256).eval()
try:
    model.max_seq_length = MAX_SEQ_LEN
except Exception:
    pass

if device == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ---------------- Encode embeddings ----------------
texts = df["text_clean"].tolist()
embeddings = model.encode(
    texts,
    prompt_name="Clustering",
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True
).astype(np.float32)

# ---------------- Validate embeddings ----------------
finite_mask = np.isfinite(embeddings).all(axis=1)
if not finite_mask.all():
    n_bad = (~finite_mask).sum()
    print(f"Found {n_bad} rows with NaN/Inf; re-encoding just those rows…")
    bad_idx = np.where(~finite_mask)[0]
    bad_texts = [texts[i] for i in bad_idx]

    fixed = model.encode(
        bad_texts,
        batch_size=max(64, BATCH_SIZE // 2),
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True
    ).astype(np.float32)

    embeddings[bad_idx] = fixed
    assert np.isfinite(embeddings).all(), "Still found NaN/Inf after repair"

if NORMALIZE:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nz = norms.squeeze() > 0
    embeddings[nz] = embeddings[nz] / norms[nz]

# ---------------- Save embeddings ----------------
shutil.rmtree(BASE_DIR, ignore_errors=True)
os.makedirs(BASE_DIR, exist_ok=True)

emb_list_arrays = [pa.array(row, type=pa.float32()) for row in embeddings]
table = pa.table({
    "post_id": pa.array(df["post_id"].tolist()),
    "text": pa.array(df["text"].tolist()),
    "text_clean": pa.array(df["text_clean"].tolist()),
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
