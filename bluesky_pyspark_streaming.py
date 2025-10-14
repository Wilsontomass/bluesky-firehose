import json
import asyncio
import websockets
import threading
import time
import logging
from datetime import datetime
from typing import Optional
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlueskyToKafkaProducer:
    """
    Connects to Bluesky WebSocket firehose and produces messages to Kafka.
    This component can run on a separate machine from Spark.
    """
    def __init__(self, websocket_uri: str, kafka_bootstrap_servers: str, kafka_topic: str):
        self.websocket_uri = websocket_uri
        self.kafka_topic = kafka_topic
        self.running = False
        self.websocket_thread = None
        
        # Initialize Kafka producer with optimal settings for high throughput
        self.producer = KafkaProducer(
            api_version=(3, 9),
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            # Performance optimizations
            batch_size=16384,  # Batch messages for efficiency
            linger_ms=10,      # Small delay to allow batching
            compression_type='snappy',  # Compress messages
            acks=1,          # Balance between performance and durability
            retries=3,
            max_in_flight_requests_per_connection=5
        )
        
    def start(self):
        """Start the WebSocket-to-Kafka producer."""
        if not self.running:
            self.running = True
            self.websocket_thread = threading.Thread(target=self._run_websocket_listener)
            self.websocket_thread.daemon = True
            self.websocket_thread.start()
            logger.info(f"Bluesky-to-Kafka producer started for topic: {self.kafka_topic}")
    
    def stop(self):
        """Stop the producer and flush remaining messages."""
        self.running = False
        if self.websocket_thread:
            self.websocket_thread.join(timeout=10)
        
        # Flush any remaining messages
        self.producer.flush()
        self.producer.close()
        logger.info("Bluesky-to-Kafka producer stopped")
    
    def _run_websocket_listener(self):
        """Run WebSocket listener in async context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._websocket_listener())
    
    async def _websocket_listener(self):
        """Listen to WebSocket and produce to Kafka."""
        message_count = 0
        last_report = time.time()
        
        while self.running:
            try:
                async with websockets.connect(self.websocket_uri) as websocket:
                    logger.info(f"Connected to Bluesky firehose: {self.websocket_uri}")
                    while self.running:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=1.0
                            )
                            
                            try:
                                data = json.loads(message)
                                
                                # Use DID as partition key for even distribution
                                partition_key = data.get('did', 'unknown')
                                
                                # Send to Kafka (non-blocking)
                                future = self.producer.send(
                                    self.kafka_topic, 
                                    value=data, 
                                    key=partition_key
                                )
                                
                                # Optional: Add callback for error handling
                                future.add_callback(self._on_send_success)
                                future.add_errback(self._on_send_error)
                                
                                message_count += 1
                                
                                # Periodic reporting
                                if time.time() - last_report > 30:
                                    logger.info(f"Produced {message_count} messages to Kafka")
                                    last_report = time.time()
                                    message_count = 0
                                    
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {message[:100]}")
                                
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed as e:
                            logger.warning(f"WebSocket connection closed: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message send."""
        pass  # Could add metrics here
        
    def _on_send_error(self, exception):
        """Callback for failed message send."""
        logger.error(f"Failed to send message to Kafka: {exception}")


def create_bluesky_kafka_stream_df(spark: SparkSession, 
                                  kafka_bootstrap_servers: str, 
                                  kafka_topic: str,
                                  starting_offsets: str = "latest"):
    """
    Create a Spark Structured Streaming DataFrame from Kafka.
    This scales horizontally - each Spark executor reads from different Kafka partitions.
    """
    
    # Define schema for Bluesky post data
    schema = StructType([
        StructField("did", StringType(), True),
        StructField("time_us", LongType(), True),
        StructField("kind", StringType(), True),
        StructField("commit", StructType([
            StructField("rev", StringType(), True),
            StructField("operation", StringType(), True),
            StructField("collection", StringType(), True),
            StructField("rkey", StringType(), True),
            StructField("record", StructType([
                StructField("$type", StringType(), True),
                StructField("text", StringType(), True),
                StructField("langs", ArrayType(StringType()), True),
                StructField("createdAt", StringType(), True)
            ]), True)
        ]), True)
    ])
    
    # Create streaming DataFrame from Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", starting_offsets) \
        .option("maxOffsetsPerTrigger", 10000) \
        .option("kafka.consumer.group.id", "bluesky-analytics") \
        .load()
    
    # Parse the JSON value from Kafka
    parsed_df = kafka_df.select(
        col("partition"),
        col("offset"),
        col("timestamp").alias("kafka_timestamp"),
        col("key").cast("string").alias("partition_key"),
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("partition", "offset", "kafka_timestamp", "partition_key", "data.*")
    
    return parsed_df


def process_language_counts_kafka(df):
    """
    Process language counts with Kafka metadata for better observability.
    """
    # Filter for post commits
    posts_df = df.filter(
        (col("kind") == "commit") & 
        (col("commit.operation") == "create") &
        (col("commit.collection") == "app.bsky.feed.post")
    )
    
    # Extract languages with partition info for debugging
    languages_df = posts_df \
        .select(
            col("partition"),
            col("offset"), 
            col("kafka_timestamp"),
            col("time_us"),
            col("partition_key"),
            col("commit.record.text").alias("text"),
            explode(coalesce(col("commit.record.langs"), array())).alias("language")
        ) \
        .filter(col("language").isNotNull() & (col("language") != ""))
    
    # Use post timestamp for windowing, but keep Kafka metadata
    languages_df = languages_df \
        .withColumn("post_timestamp", (col("time_us") / 1000000).cast("timestamp")) \
        .withColumn("processing_delay_sec", 
                   (col("kafka_timestamp").cast("long") - col("post_timestamp").cast("long")))
    
    # Aggregate with windowing
    language_counts = languages_df \
        .withWatermark("post_timestamp", "1 minute") \
        .groupBy(
            window(col("post_timestamp"), "2 minutes", "1 minute"),
            col("language")
        ) \
        .agg(
            count("*").alias("post_count"),
            approx_count_distinct("partition_key").alias("unique_users"),
            avg("processing_delay_sec").alias("avg_processing_delay"),
            approx_count_distinct("partition").alias("partitions_processed"),
            min("offset").alias("min_offset"),
            max("offset").alias("max_offset")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("language"),
            col("post_count"),
            col("unique_users"),
            round(col("avg_processing_delay"), 2).alias("avg_delay_sec"),
            col("partitions_processed"),
            col("min_offset"),
            col("max_offset")
        )
    
    return language_counts


def process_posts_for_embeddings(df):
    posts_df = df.filter(
        (col("kind") == "commit") &
        (col("commit.operation") == "create") &
        (col("commit.collection") == "app.bsky.feed.post")
    )
    
    raw_posts_df = posts_df.select(
        col("partition_key").alias("user_id"),
        col("commit.record.text").alias("text"),
        coalesce(col("commit.record.langs"), array()).alias("languages"),
        (col("time_us") / 1000000).cast("timestamp").alias("post_timestamp")
    ).filter(col("text").isNotNull() & (col("text") != ""))

    raw_posts_df = raw_posts_df.withWatermark("post_timestamp", "3 minutes") \
                           .dropDuplicates(["user_id", "post_timestamp"])


    raw_posts_df = raw_posts_df.withColumn(
        "window_start", window(col("post_timestamp"), "2 minutes").start
    ).withColumn(
        "window_end", window(col("post_timestamp"), "2 minutes").end
    )

    return raw_posts_df



def process_platform_health_metrics(df):
    """
    Monitor overall platform health and processing performance.
    """
    # General metrics across all message types
    health_df = df.select(
        col("partition"),
        col("offset"),
        col("kafka_timestamp"),
        col("time_us"),
        col("kind"),
        col("did")
    ).withColumn("post_timestamp", (col("time_us") / 1000000).cast("timestamp")) \
     .withColumn("processing_delay_sec", 
                (col("kafka_timestamp").cast("long") - col("post_timestamp").cast("long")))
    
    # Aggregate health metrics
    health_metrics = health_df \
        .withWatermark("post_timestamp", "1 minute") \
        .groupBy(
            window(col("post_timestamp"), "1 minute", "30 seconds")
        ) \
        .agg(
            count("*").alias("total_messages"),
            approx_count_distinct("did").alias("unique_users"),
            approx_count_distinct("partition").alias("kafka_partitions_active"),
            avg("processing_delay_sec").alias("avg_processing_delay"),
            sum(when(col("kind") == "commit", 1).otherwise(0)).alias("commit_messages"),
            sum(when(col("kind") == "identity", 1).otherwise(0)).alias("identity_messages"),
            sum(when(col("kind") == "account", 1).otherwise(0)).alias("account_messages")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("total_messages"),
            col("unique_users"),
            col("kafka_partitions_active"),
            round(col("avg_processing_delay"), 2).alias("avg_delay_sec"),
            col("commit_messages"),
            col("identity_messages"),
            col("account_messages"),
            round((col("commit_messages") * 100.0 / col("total_messages")), 2).alias("commit_pct")
        )
    
    return health_metrics


def main():
    """
    Main entry point for the Kafka-based streaming application.
    """
    # Configuration
    kafka_bootstrap_servers = "localhost:9092"  # Adjust for your Kafka cluster
    kafka_topic = "bluesky-firehose"
    websocket_uri = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
    
    # Initialize Spark with Kafka package
    spark = SparkSession.builder \
        .appName("BlueskyKafkaAnalysis") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.streaming.checkpointLocation.timeout", "10s") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Start Kafka producer (this could run on a separate machine)
    producer = BlueskyToKafkaProducer(websocket_uri, kafka_bootstrap_servers, kafka_topic)
    producer.start()
    
    # Wait for some data to flow
    logger.info("Waiting for Kafka producer to establish connection and produce data...")
    time.sleep(15)
    
    try:
        # Create streaming DataFrame from Kafka
        logger.info("Starting Spark streaming from Kafka...")
        stream_df = create_bluesky_kafka_stream_df(
            spark, 
            kafka_bootstrap_servers, 
            kafka_topic,
            starting_offsets="earliest"  # Process all available data
        )
        
        # Process different analytics
        language_counts = process_language_counts_kafka(stream_df)
        health_metrics = process_platform_health_metrics(stream_df)
        raw_posts_df = process_posts_for_embeddings(stream_df)
        
        # Console outputs for monitoring
        language_console_query = language_counts.writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 20) \
            .trigger(processingTime="1 minute") \
            .queryName("language_console") \
            .start()
        
        raw_posts_console_query = raw_posts_df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 10) \
            .trigger(processingTime="1 minute") \
            .queryName("raw_posts_console") \
            .start()
        
        health_console_query = health_metrics.writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime="30 seconds") \
            .queryName("health_console") \
            .start()
        
        # Persistent storage with partitioning
        language_parquet_query = language_counts.writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", "output/language_counts_kafka") \
            .option("checkpointLocation", "checkpoint/language_kafka") \
            .partitionBy("window_start") \
            .trigger(processingTime="2 minutes") \
            .queryName("language_parquet") \
            .start()
        
        raw_posts_parquet_query = raw_posts_df.writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", "output/raw_posts_kafka") \
            .option("checkpointLocation", "checkpoint/raw_posts_kafka") \
            .partitionBy("window_start") \
            .trigger(processingTime="1 minute") \
            .queryName("raw_posts_parquet") \
            .start()


        
        health_parquet_query = health_metrics.writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", "output/health_metrics_kafka") \
            .option("checkpointLocation", "checkpoint/health_kafka") \
            .partitionBy("window_start") \
            .trigger(processingTime="1 minute") \
            .queryName("health_parquet") \
            .start()
        
        # In-memory tables for real-time dashboards
        language_memory_query = language_counts.writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("language_counts_kafka_table") \
            .trigger(processingTime="1 minute") \
            .start()
        
        posts_memory_query = raw_posts_df.writeStream \
            .outputMode("append") \
            .format("memory") \
            .queryName("raw_posts_kafka_table") \
            .trigger(processingTime="1 minute") \
            .start()
        
        health_memory_query = health_metrics.writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("health_metrics_kafka_table") \
            .trigger(processingTime="30 seconds") \
            .start()
        
        logger.info("=== Kafka-based Bluesky Analytics Pipeline Running ===")
        logger.info("✓ Horizontal scaling via Kafka partitions")  
        logger.info("✓ No driver bottleneck - each executor reads different partitions")
        logger.info("✓ Fault tolerant with Kafka's replication")
        logger.info("✓ Backpressure handling via maxOffsetsPerTrigger")
        logger.info("Press Ctrl+C to stop...")
        
        # Monitor and report processing health
        last_health_check = time.time()
        while True:
            time.sleep(15)
            
            if time.time() - last_health_check > 60:
                # Check streaming query health
                active_queries = [q for q in spark.streams.active if q.isActive]
                logger.info(f"Health: {len(active_queries)} active queries")
                
                # Log any query exceptions
                for query in active_queries:
                    if query.exception():
                        logger.error(f"Query {query.name} has exception: {query.exception()}")
                
                last_health_check = time.time()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        # Graceful shutdown
        logger.info("Shutting down pipeline...")
        
        # Stop Kafka producer first
        producer.stop()
        
        # Stop all streaming queries
        for query in spark.streams.active:
            try:
                query.stop()
                logger.info(f"Stopped query: {query.name}")
            except Exception as e:
                logger.error(f"Error stopping {query.name}: {e}")
        
        spark.stop()
        logger.info("=== Pipeline shutdown complete ===")


# Advanced analysis utilities for Kafka-based data

def analyze_processing_delays(spark: SparkSession):
    """Analyze processing delays across the pipeline."""
    try:
        return spark.sql("""
            SELECT 
                window_start,
                avg_delay_sec,
                partitions_processed,
                CASE 
                    WHEN avg_delay_sec < 1 THEN 'Excellent'
                    WHEN avg_delay_sec < 5 THEN 'Good' 
                    WHEN avg_delay_sec < 15 THEN 'Fair'
                    ELSE 'Poor'
                END as latency_grade
            FROM language_counts_kafka_table
            ORDER BY window_start DESC
            LIMIT 10
        """)
    except Exception as e:
        logger.error(f"Error analyzing delays: {e}")
        return None

def check_kafka_partition_distribution(spark: SparkSession):
    """Check how evenly data is distributed across Kafka partitions.""" 
    try:
        return spark.sql("""
            SELECT 
                window_start,
                partitions_processed,
                (max_offset - min_offset) as offset_range,
                post_count,
                round(post_count / partitions_processed, 2) as avg_posts_per_partition
            FROM language_counts_kafka_table
            WHERE partitions_processed > 0
            ORDER BY window_start DESC
            LIMIT 10
        """)
    except Exception as e:
        logger.error(f"Error checking partition distribution: {e}")
        return None

def get_top_languages_with_kafka_metrics(spark: SparkSession, limit: int = 15):
    """Get top languages with Kafka processing metrics."""
    try:
        return spark.sql(f"""
            SELECT 
                language,
                SUM(post_count) as total_posts,
                SUM(unique_users) as total_unique_users,
                AVG(avg_delay_sec) as avg_processing_delay,
                COUNT(*) as time_windows,
                AVG(partitions_processed) as avg_partitions_used
            FROM language_counts_kafka_table
            GROUP BY language
            ORDER BY total_posts DESC
            LIMIT {limit}
        """)
    except Exception as e:
        logger.error(f"Error querying top languages: {e}")
        return None


if __name__ == "__main__":
    main()