"""
Stream Processor for Real-time Data Processing

Handles real-time processing of data streams from Kafka topics.
Integrates with NLP pipeline and data structuring components.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
from .nlp_pipeline import NLPPipeline
from .data_structurer import DataStructurer
from config import CONFIG

logger = logging.getLogger(__name__)

class StreamProcessor:
    """
    Real-time stream processor for unstructured data.
    
    Features:
    - Multi-topic Kafka consumption
    - Parallel processing with thread pools
    - NLP pipeline integration
    - Data structuring and normalization
    - Error handling and retry logic
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize stream processor."""
        self.config = CONFIG
        self.is_running = False
        
        # Initialize components
        self.nlp_pipeline = NLPPipeline()
        self.data_structurer = DataStructurer()
        
        # Initialize Kafka consumer and producer
        self.consumer = self._init_kafka_consumer()
        self.producer = self._init_kafka_producer()
        
        # Initialize Redis for caching
        self.redis_client = self._init_redis_client()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=CONFIG.processing.max_workers)
        
        # Performance metrics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        logger.info("Stream processor initialized")
    
    def _init_kafka_consumer(self) -> KafkaConsumer:
        """Initialize Kafka consumer."""
        try:
            consumer = KafkaConsumer(
                *CONFIG.kafka.topics.values(),
                bootstrap_servers=CONFIG.kafka.bootstrap_servers,
                group_id=CONFIG.kafka.consumer_group,
                auto_offset_reset=CONFIG.kafka.auto_offset_reset,
                enable_auto_commit=CONFIG.kafka.enable_auto_commit,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000  # 1 second timeout
            )
            logger.info("Kafka consumer initialized")
            return consumer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for processed data."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=CONFIG.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            logger.info("Kafka producer initialized")
            return producer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _init_redis_client(self) -> redis.Redis:
        """Initialize Redis client."""
        try:
            client = redis.Redis(
                host=CONFIG.redis.host,
                port=CONFIG.redis.port,
                db=CONFIG.redis.db,
                password=CONFIG.redis.password,
                decode_responses=CONFIG.redis.decode_responses
            )
            client.ping()
            logger.info("Redis client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    def _extract_text_content(self, message: Dict[str, Any]) -> str:
        """
        Extract text content from message based on source type.
        
        Args:
            message: Message data
            
        Returns:
            Extracted text content
        """
        data = message.get('data', {})
        connector = message.get('connector', '')
        
        # Handle different data sources
        if connector == 'rss_connector':
            content_parts = []
            if data.get('title'):
                content_parts.append(data['title'])
            if data.get('content'):
                content_parts.append(data['content'])
            elif data.get('summary'):
                content_parts.append(data['summary'])
            return ' '.join(content_parts)
        
        elif connector == 'news_connector':
            content_parts = []
            if data.get('title'):
                content_parts.append(data['title'])
            if data.get('description'):
                content_parts.append(data['description'])
            if data.get('content'):
                content_parts.append(data['content'])
            elif data.get('full_content'):
                content_parts.append(data['full_content'])
            return ' '.join(content_parts)
        
        elif connector == 'file_connector':
            content_parts = []
            metadata = data.get('metadata', {})
            if metadata.get('filename'):
                content_parts.append(f"File: {metadata['filename']}")
            if data.get('content'):
                content_parts.append(data['content'])
            return ' '.join(content_parts)
        
        else:
            # Generic content extraction
            if isinstance(data, dict):
                content_fields = ['content', 'text', 'description', 'summary', 'title']
                for field in content_fields:
                    if data.get(field):
                        return str(data[field])
            return str(data)
    
    def _process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single message through the NLP pipeline.
        
        Args:
            message: Message to process
            
        Returns:
            Processed and structured data
        """
        try:
            # Extract text content
            text_content = self._extract_text_content(message)
            
            if not text_content or len(text_content.strip()) < 10:
                logger.debug("Skipping message with insufficient content")
                return None
            
            # Process through NLP pipeline
            nlp_results = self.nlp_pipeline.process_text(text_content)
            
            if 'error' in nlp_results:
                logger.error(f"NLP processing error: {nlp_results['error']}")
                return None
            
            # Structure the data
            structured_data = self.data_structurer.structure_data(message, nlp_results)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            return None
    
    def _publish_processed_data(self, processed_data: Dict[str, Any]) -> bool:
        """
        Publish processed data to output topic.
        
        Args:
            processed_data: Processed data to publish
            
        Returns:
            True if published successfully
        """
        try:
            topic = CONFIG.kafka.topics["processed_data"]
            key = processed_data.get('id', '')
            
            future = self.producer.send(topic, value=processed_data, key=key)
            future.get(timeout=10)
            
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error publishing processed data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error publishing processed data: {e}")
            return False
    
    def _cache_processed_data(self, processed_data: Dict[str, Any]) -> bool:
        """
        Cache processed data in Redis.
        
        Args:
            processed_data: Data to cache
            
        Returns:
            True if cached successfully
        """
        try:
            key = f"processed:{processed_data.get('id', 'unknown')}"
            self.redis_client.setex(key, 3600, json.dumps(processed_data))
            return True
        except Exception as e:
            logger.error(f"Error caching processed data: {e}")
            return False
    
    def _process_batch(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of messages in parallel.
        
        Args:
            messages: List of messages to process
            
        Returns:
            List of processed data
        """
        processed_results = []
        
        # Submit processing tasks
        future_to_message = {
            self.executor.submit(self._process_message, msg): msg 
            for msg in messages
        }
        
        # Collect results
        for future in as_completed(future_to_message):
            try:
                result = future.result(timeout=CONFIG.processing.processing_timeout)
                if result:
                    processed_results.append(result)
                    self.processed_count += 1
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                self.error_count += 1
        
        return processed_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the stream processor.
        
        Returns:
            Dictionary containing performance metrics
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'processing_rate': self.processed_count / max(elapsed_time, 1),
            'elapsed_time': elapsed_time,
            'is_running': self.is_running
        }
    
    def start(self):
        """Start the stream processor."""
        self.is_running = True
        self.start_time = time.time()
        logger.info("Stream processor started")
    
    def stop(self):
        """Stop the stream processor and cleanup resources."""
        self.is_running = False
        
        try:
            if self.consumer:
                self.consumer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka consumer: {e}")
        
        try:
            if self.producer:
                self.producer.flush()
                self.producer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")
        
        try:
            if self.redis_client:
                self.redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
        
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")
        
        logger.info("Stream processor stopped")
    
    def run(self):
        """
        Main execution loop for stream processor.
        
        Continuously processes messages from Kafka topics.
        """
        self.start()
        
        logger.info("Starting stream processor main loop")
        
        try:
            batch = []
            batch_size = CONFIG.processing.batch_size
            
            while self.is_running:
                try:
                    # Poll for messages
                    message_pack = self.consumer.poll(timeout_ms=1000)
                    
                    if not message_pack:
                        # Process any remaining messages in batch
                        if batch:
                            processed_data = self._process_batch(batch)
                            
                            # Publish processed data
                            for data in processed_data:
                                self._publish_processed_data(data)
                                self._cache_processed_data(data)
                            
                            batch = []
                        continue
                    
                    # Collect messages from all partitions
                    for topic_partition, messages in message_pack.items():
                        for message in messages:
                            try:
                                batch.append(message.value)
                                
                                # Process batch when it reaches target size
                                if len(batch) >= batch_size:
                                    processed_data = self._process_batch(batch)
                                    
                                    # Publish processed data
                                    for data in processed_data:
                                        self._publish_processed_data(data)
                                        self._cache_processed_data(data)
                                    
                                    batch = []
                                    
                            except Exception as e:
                                logger.error(f"Error processing individual message: {e}")
                                self.error_count += 1
                
                except Exception as e:
                    logger.error(f"Error in main processing loop: {e}")
                    time.sleep(1)  # Brief pause before retrying
                    
        except KeyboardInterrupt:
            logger.info("Stream processor interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in stream processor: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run stream processor
    processor = StreamProcessor()
    processor.run()

