"""
Standalone Bluesky to Kafka Producer
Connects to Bluesky WebSocket firehose and streams to Kafka
"""
import json
import asyncio
import websockets
import logging
import os
import signal
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlueskyKafkaProducer:
    """Connects to Bluesky WebSocket firehose and produces messages to Kafka."""
    
    def __init__(self, websocket_uri: str, kafka_bootstrap_servers: str, 
                 kafka_topic: str, sasl_username: str = None, sasl_password: str = None):
        self.websocket_uri = websocket_uri
        self.kafka_topic = kafka_topic
        self.running = False
        self.message_count = 0
        self.error_count = 0
        
        # Build Kafka config
        kafka_config = {
            'bootstrap_servers': kafka_bootstrap_servers,
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'key_serializer': lambda x: x.encode('utf-8') if x else None,
            'batch_size': 16384,
            'linger_ms': 10,
            'compression_type': 'snappy',
            'acks': 1,
            'retries': 3,
            'max_in_flight_requests_per_connection': 5
        }
        
        # Add SASL authentication if credentials provided
        if sasl_username and sasl_password:
            kafka_config.update({
                'security_protocol': 'SASL_PLAINTEXT',
                'sasl_mechanism': 'PLAIN',
                'sasl_plain_username': sasl_username,
                'sasl_plain_password': sasl_password
            })
            logger.info(f"Kafka authentication enabled for user: {sasl_username}")
        
        try:
            self.producer = KafkaProducer(**kafka_config)
            logger.info(f"Connected to Kafka at {kafka_bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def start(self):
        """Start the WebSocket listener and Kafka producer."""
        self.running = True
        logger.info(f"Starting producer for topic: {self.kafka_topic}")
        
        while self.running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def _connect_and_stream(self):
        """Connect to WebSocket and stream messages."""
        try:
            async with websockets.connect(
                self.websocket_uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                logger.info("Connected to Bluesky firehose")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        await self._process_message(message)
                        
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket timeout, checking connection...")
                        continue
                    except websockets.ConnectionClosed as e:
                        logger.warning(f"WebSocket closed: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def _process_message(self, message: str):
        """Process and send a single message to Kafka."""
        try:
            data = json.loads(message)
            
            # Use DID as partition key for better distribution
            partition_key = data.get('did', 'unknown')
            
            # Send to Kafka
            future = self.producer.send(
                self.kafka_topic,
                value=data,
                key=partition_key
            )
            
            # Optional: add callback for debugging
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            self.message_count += 1
            
            # Log progress
            if self.message_count % 1000 == 0:
                logger.info(f"Processed {self.message_count} messages (errors: {self.error_count})")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self.error_count += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
    
    def _on_send_success(self, record_metadata):
        """Callback for successful Kafka send."""
        pass  # Could log detailed info if needed
    
    def _on_send_error(self, exception):
        """Callback for failed Kafka send."""
        logger.error(f"Failed to send message to Kafka: {exception}")
        self.error_count += 1
    
    def stop(self):
        """Stop the producer gracefully."""
        logger.info("Stopping producer...")
        self.running = False
        
        # Flush remaining messages
        try:
            self.producer.flush(timeout=10)
            logger.info(f"Flushed remaining messages. Total processed: {self.message_count}")
        except Exception as e:
            logger.error(f"Error flushing producer: {e}")
        
        self.producer.close()
        logger.info("Producer stopped")


async def main():
    """Main entry point."""
    # Load configuration from environment variables
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'bluesky-firehose')
    WEBSOCKET_URI = os.getenv('BLUESKY_WEBSOCKET_URI', 
                              'wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post')
    SASL_USERNAME = os.getenv('KAFKA_SASL_USERNAME')
    SASL_PASSWORD = os.getenv('KAFKA_SASL_PASSWORD')
    
    # Log configuration (without sensitive data)
    logger.info("Configuration:")
    logger.info(f"  Kafka Servers: {KAFKA_SERVERS}")
    logger.info(f"  Kafka Topic: {KAFKA_TOPIC}")
    logger.info(f"  WebSocket URI: {WEBSOCKET_URI}")
    logger.info(f"  Auth Enabled: {bool(SASL_USERNAME and SASL_PASSWORD)}")
    
    # Initialize producer
    producer = BlueskyKafkaProducer(
        websocket_uri=WEBSOCKET_URI,
        kafka_bootstrap_servers=KAFKA_SERVERS,
        kafka_topic=KAFKA_TOPIC,
        sasl_username=SASL_USERNAME,
        sasl_password=SASL_PASSWORD
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        producer.stop()
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Start streaming
        await producer.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        producer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)