"""
Base Connector Class for Data Sources

Provides common interface and functionality for all data source connectors.
"""

import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import redis
from config import CONFIG

logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.
    
    Provides common functionality for:
    - Kafka message publishing
    - Redis caching
    - Error handling and retry logic
    - Health monitoring
    - Configuration management
    """
    
    def __init__(self, connector_name: str, kafka_topic: str):
        """
        Initialize the base connector.
        
        Args:
            connector_name: Unique name for this connector
            kafka_topic: Kafka topic to publish messages to
        """
        self.connector_name = connector_name
        self.kafka_topic = kafka_topic
        self.is_running = False
        self.error_count = 0
        self.message_count = 0
        self.last_activity = None
        
        # Initialize Kafka producer
        self.kafka_producer = self._init_kafka_producer()
        
        # Initialize Redis client
        self.redis_client = self._init_redis_client()
        
        logger.info(f"Initialized {connector_name} connector")
    
    def _init_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer with appropriate configuration."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=CONFIG.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logger.info(f"Kafka producer initialized for {self.connector_name}")
            return producer
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def _init_redis_client(self) -> redis.Redis:
        """Initialize Redis client for caching and state management."""
        try:
            client = redis.Redis(
                host=CONFIG.redis.host,
                port=CONFIG.redis.port,
                db=CONFIG.redis.db,
                password=CONFIG.redis.password,
                decode_responses=CONFIG.redis.decode_responses
            )
            # Test connection
            client.ping()
            logger.info(f"Redis client initialized for {self.connector_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise
    
    def publish_message(self, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Publish a message to Kafka topic.
        
        Args:
            data: Message data to publish
            key: Optional message key for partitioning
            
        Returns:
            bool: True if message was published successfully
        """
        try:
            # Add metadata
            message = {
                'connector': self.connector_name,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            # Publish to Kafka
            future = self.kafka_producer.send(
                self.kafka_topic,
                value=message,
                key=key
            )
            
            # Wait for acknowledgment (with timeout)
            record_metadata = future.get(timeout=10)
            
            self.message_count += 1
            self.last_activity = datetime.utcnow()
            
            logger.debug(f"Published message to {record_metadata.topic}:{record_metadata.partition}")
            return True
            
        except KafkaError as e:
            self.error_count += 1
            logger.error(f"Kafka error publishing message: {e}")
            return False
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error publishing message: {e}")
            return False
    
    def cache_data(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """
        Cache data in Redis.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            bool: True if data was cached successfully
        """
        try:
            cache_key = f"{self.connector_name}:{key}"
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(data) if not isinstance(data, str) else data
            )
            return True
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieve cached data from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        try:
            cache_key = f"{self.connector_name}:{key}"
            data = self.redis_client.get(cache_key)
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the connector.
        
        Returns:
            Dictionary containing health metrics
        """
        return {
            'connector_name': self.connector_name,
            'is_running': self.is_running,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'error_rate': self.error_count / max(self.message_count, 1)
        }
    
    def start(self):
        """Start the connector."""
        self.is_running = True
        logger.info(f"Started {self.connector_name} connector")
    
    def stop(self):
        """Stop the connector and cleanup resources."""
        self.is_running = False
        
        try:
            if self.kafka_producer:
                self.kafka_producer.flush()
                self.kafka_producer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")
        
        try:
            if self.redis_client:
                self.redis_client.close()
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
        
        logger.info(f"Stopped {self.connector_name} connector")
    
    @abstractmethod
    def collect_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Abstract method to collect data from the source.
        
        Yields:
            Dictionary containing collected data
        """
        pass
    
    @abstractmethod
    def run(self):
        """
        Abstract method to run the connector.
        
        This method should implement the main loop for data collection
        and publishing.
        """
        pass

