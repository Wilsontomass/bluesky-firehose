"""
Offline embedding job for Bluesky posts produced by the Kafka/Spark pipeline.

Reads:   output/raw_posts_kafka (parquet)
Writes:  output/raw_posts_embeddings_gemma (parquet)

Each output row contains the original post fields + an embedding vector.
"""

import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


import math
from typing import Iterator

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType,
    TimestampType, IntegerType
)

# ---- Configuration ---------------------------------------------------------

INPUT_PARQUET = os.environ.get("RAW_POSTS_PARQUET", "output/raw_posts_kafka")
OUTPUT_PARQUET = os.environ.get("EMBEDDINGS_PARQUET", "output/raw_posts_embeddings_gemma")

# Hugging Face model
MODEL_ID = os.environ.get("EMBED_MODEL_ID", "google/embeddinggemma-300m")

# Set to 256 to use MRL-truncated vectors (smaller storage) or 768 for full size
EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))  # 768 | 512 | 256 | 128

# Batch size per worker encode() call
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))

# Prompting per Google’s guidance (document embeddings)
# Ref: title: {title|"none"} | text: {content}
DOC_PROMPT = os.environ.get("EMBED_DOC_PROMPT", 'title: none | text: ')

# Optional: limit processed rows for dry runs
ROW_LIMIT = int(os.environ.get("ROW_LIMIT", "0"))

# ---- Spark -----------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("OfflineBlueskyEmbeddings_Gemma")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "100")
    .config("spark.python.worker.reuse", "false")  # easier debugging on Windows
    .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    .config("spark.python.worker.faulthandler.enabled", "true")
    # optional if you have RAM:
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ---- Input schema guard (matches your raw_posts_df) ------------------------

# We only need these columns from the upstream parquet
NEEDED_COLS = [
    "did", "rkey", "user_id", "text", "langs", "post_timestamp", "createdAt"
]

# ---- Pandas mapper with local model singleton ------------------------------

# We import transformers/sentence-transformers lazily inside the worker to keep driver lean
_model_singleton = {"model": None, "dim": EMBED_DIM}

def _load_model():
    """
    Load SentenceTransformer model once per worker process.
    Uses MRL truncation by slicing to EMBED_DIM and re-normalizing.
    """
    if _model_singleton["model"] is None:
        # NOTE: You need sentence-transformers >= 3.0 and transformers that support EmbeddingGemma.
        # See Google's doc for install guidance.
        # pip install -U sentence-transformers
        # (Older docs used a preview transformers branch; current HF release supports the model.)
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_ID, device=device)
        _model_singleton["model"] = model
    return _model_singleton["model"]

def _truncate_mrl(embeds, target_dim: int):
    """
    Matryoshka truncation: keep leading target_dim, then L2-normalize.
    embeds: np.ndarray [N, 768]
    """
    import numpy as np
    if target_dim <= 0 or target_dim >= embeds.shape[1]:
        return embeds
    E = embeds[:, :target_dim]
    # L2 normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return E / norms

def _encode_batch(texts):
    """
    Encode a list of strings with prompt prefix and return a list of lists[float].
    """
    import numpy as np
    model = _load_model()

    # Prepend recommended document prompt
    prompted = [DOC_PROMPT + (t or "") for t in texts]

    # sentence-transformers handles batching internally but we can pass batch_size for better control
    embeds = model.encode(
        prompted,
        batch_size=min(BATCH_SIZE, max(1, len(texts))),
        show_progress_bar=False,
        normalize_embeddings=True  # cosine-ready
    )

    # Optional Matryoshka truncation
    if EMBED_DIM and EMBED_DIM < embeds.shape[1]:
        embeds = _truncate_mrl(embeds, EMBED_DIM)

    # Ensure python lists of float32 (smaller parquet footprint)
    return [list(np.asarray(vec, dtype=np.float32)) for vec in embeds]


def map_partition(pdf_iter: Iterator["pandas.DataFrame"]) -> Iterator["pandas.DataFrame"]:
    """
    mapInPandas iterator. Receives Pandas DataFrames, returns a DataFrame with added embedding column.
    """
    import pandas as pd

    for pdf in pdf_iter:
        if pdf.empty:
            yield pdf.assign(embedding=[[]]*0)
            continue

        # Clean and keep index mapping
        texts = pdf["text"].fillna("").astype(str).tolist()

        # Encode in mini-batches to cap memory if needed
        out_vectors = []
        if len(texts) == 0:
            out_vectors = []
        else:
            for start in range(0, len(texts), BATCH_SIZE):
                chunk = texts[start:start+BATCH_SIZE]
                out_vectors.extend(_encode_batch(chunk))

        # Safety: lengths must match
        assert len(out_vectors) == len(texts), "Embedding/text length mismatch"

        pdf = pdf.copy()
        pdf["embedding"] = out_vectors
        pdf["embedding_dim"] = EMBED_DIM
        pdf["embedding_model"] = MODEL_ID
        pdf["embedding_prompt"] = DOC_PROMPT
        yield pdf

# ---- Read input ------------------------------------------------------------

df = spark.read.parquet(INPUT_PARQUET).select(*NEEDED_COLS)

if ROW_LIMIT > 0:
    df = df.limit(ROW_LIMIT)

# We’ll ensure types expected by the mapper
df = (df
      .withColumn("did", col("did").cast(StringType()))
      .withColumn("rkey", col("rkey").cast(StringType()))
      .withColumn("user_id", col("user_id").cast(StringType()))
      .withColumn("text", col("text").cast(StringType()))
      .withColumn("createdAt", col("createdAt").cast(StringType()))
      .withColumn("post_timestamp", col("post_timestamp").cast(TimestampType()))
)

# ---- Apply embeddings ------------------------------------------------------

# Define the output schema for mapInPandas to avoid schema inference surprises
from pyspark.sql.types import StructType, StructField

output_schema = StructType([
    StructField("did", StringType(), True),
    StructField("rkey", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("langs", ArrayType(StringType()), True),
    StructField("post_timestamp", TimestampType(), True),
    StructField("createdAt", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True),
    StructField("embedding_dim", IntegerType(), True), 
    StructField("embedding_model",    # provenance
                StringType(), True),
    StructField("embedding_prompt",   # provenance
                StringType(), True),
])

# Note: store embedding_dim as string for easier filtering in Spark SQL; change to IntegerType() if preferred
df = df.mapInPandas(map_partition, schema=output_schema)

# ---- Write output ----------------------------------------------------------

(
    df.write
      .mode("overwrite")
      .option("compression", "zstd")      # smaller parquet
      .parquet(OUTPUT_PARQUET)
)

print(f"✓ Wrote embeddings to: {OUTPUT_PARQUET}")
spark.stop()
