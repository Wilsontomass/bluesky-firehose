import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from typing import Iterator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType, FloatType
)

# --- Config ---
INPUT_PARQUET  = os.environ.get("RAW_POSTS_PARQUET", "output/raw_posts_kafka")
OUTPUT_PARQUET = os.environ.get("EMBEDDINGS_PARQUET", "output/raw_posts_embeddings_gemma")
MODEL_ID       = os.environ.get("EMBED_MODEL_ID", "google/embeddinggemma-300m")
BATCH_SIZE     = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
ROW_LIMIT      = int(os.environ.get("ROW_LIMIT", "0"))

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

# --- Input: keep only the columns you need ---
# If you ALREADY have a single 'post_id' column in parquet, just select it.
# Otherwise, create one (common: 'at://{did}/app.bsky.feed.post/{rkey}')
df = spark.read.parquet(INPUT_PARQUET).select("did", "rkey", "text")

# Make a simple post_id. Adjust to your canonical format if needed.
df = df.withColumn("post_id", concat_ws("/", col("did"), col("rkey"))).select("post_id", "text")

if ROW_LIMIT > 0:
    df = df.limit(ROW_LIMIT)

# --- Pandas partition mapper with per-worker model singleton ---
_model_singleton = {"model": None}

def _load_model():
    if _model_singleton["model"] is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_singleton["model"] = SentenceTransformer(MODEL_ID, device=device)
    return _model_singleton["model"]

def map_partition(pdf_iter: Iterator["pandas.DataFrame"]) -> Iterator["pandas.DataFrame"]:
    import pandas as pd
    import numpy as np

    model = _load_model()  # load once per worker

    for pdf in pdf_iter:
        if pdf.empty:
            yield pd.DataFrame({"post_id": [], "text": [], "embedding": []})
            continue

        texts = pdf["text"].fillna("").astype(str).tolist()

        # Encode the whole partition; sentence-transformers handles mini-batching.
        embeds = model.encode(
            texts,
            batch_size=min(BATCH_SIZE, max(1, len(texts))),
            show_progress_bar=False,
            normalize_embeddings=True  # cosine-ready
        )

        # Ensure Python list[float32] for compact parquet
        embeds = [list(np.asarray(v, dtype=np.float32)) for v in embeds]

        out = pdf[["post_id", "text"]].copy()
        out["embedding"] = embeds
        yield out

output_schema = StructType([
    StructField("post_id",  StringType(), True),
    StructField("text",     StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True),
])

out_df = df.mapInPandas(map_partition, schema=output_schema)

(out_df.write
      .mode("overwrite")
      .option("compression", "zstd")
      .parquet(OUTPUT_PARQUET))

print(f"✓ Wrote (post_id, text, embedding) to: {OUTPUT_PARQUET}")
spark.stop()
