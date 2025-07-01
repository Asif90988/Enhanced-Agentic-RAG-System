"""
RSS/Atom Feed Connector

Collects data from RSS and Atom feeds in real-time.
Implements intelligent polling with conditional GET requests.
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Generator, Optional
from datetime import datetime, timedelta
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .base_connector import BaseConnector
from config import CONFIG

logger = logging.getLogger(__name__)

class RSSConnector(BaseConnector):
    """
    RSS/Atom feed connector for real-time data ingestion.
    
    Features:
    - Intelligent polling with configurable intervals
    - Conditional GET requests to minimize bandwidth
    - Duplicate detection and filtering
    - Feed health monitoring
    - Automatic retry with exponential backoff
    """
    
    def __init__(self):
        super().__init__("rss_connector", CONFIG.kafka.topics["web_feeds"])
        
        self.feeds = CONFIG.data_sources.rss_feeds
        self.poll_interval = CONFIG.data_sources.rss_poll_interval
        
        # Initialize HTTP session with retry strategy
        self.session = self._init_http_session()
        
        # Track feed states for conditional GET
        self.feed_states = {}
        
        # Track processed items to avoid duplicates
        self.processed_items = set()
        
        logger.info(f"RSS connector initialized with {len(self.feeds)} feeds")
    
    def _init_http_session(self) -> requests.Session:
        """Initialize HTTP session with retry strategy and appropriate headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set user agent
        session.headers.update({
            'User-Agent': 'Enhanced-Agentic-RAG/1.0 (RSS Connector)'
        })
        
        return session
    
    def _get_item_id(self, item: Dict[str, Any]) -> str:
        """
        Generate unique ID for feed item.
        
        Args:
            item: Feed item dictionary
            
        Returns:
            Unique identifier for the item
        """
        # Use link if available, otherwise use title + published date
        if 'link' in item:
            return hashlib.md5(item['link'].encode()).hexdigest()
        else:
            content = f"{item.get('title', '')}{item.get('published', '')}"
            return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_feed_item(self, item, feed_url: str) -> Dict[str, Any]:
        """
        Parse and normalize feed item data.
        
        Args:
            item: Raw feed item from feedparser
            feed_url: URL of the source feed
            
        Returns:
            Normalized item data
        """
        # Extract publication date
        published = None
        if hasattr(item, 'published_parsed') and item.published_parsed:
            published = datetime(*item.published_parsed[:6]).isoformat()
        elif hasattr(item, 'updated_parsed') and item.updated_parsed:
            published = datetime(*item.updated_parsed[:6]).isoformat()
        
        # Extract content
        content = ""
        if hasattr(item, 'content') and item.content:
            content = item.content[0].value
        elif hasattr(item, 'summary'):
            content = item.summary
        elif hasattr(item, 'description'):
            content = item.description
        
        # Extract tags/categories
        tags = []
        if hasattr(item, 'tags'):
            tags = [tag.term for tag in item.tags]
        
        return {
            'id': self._get_item_id(item.__dict__),
            'title': getattr(item, 'title', ''),
            'link': getattr(item, 'link', ''),
            'content': content,
            'summary': getattr(item, 'summary', ''),
            'published': published,
            'author': getattr(item, 'author', ''),
            'tags': tags,
            'source_feed': feed_url,
            'source_type': 'rss_feed',
            'collected_at': datetime.utcnow().isoformat()
        }
    
    def _fetch_feed(self, feed_url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch and parse RSS/Atom feed.
        
        Args:
            feed_url: URL of the feed to fetch
            
        Returns:
            List of new feed items or None if error
        """
        try:
            # Get cached state for conditional GET
            feed_state = self.feed_states.get(feed_url, {})
            headers = {}
            
            if 'etag' in feed_state:
                headers['If-None-Match'] = feed_state['etag']
            if 'last_modified' in feed_state:
                headers['If-Modified-Since'] = feed_state['last_modified']
            
            # Fetch feed
            response = self.session.get(feed_url, headers=headers, timeout=30)
            
            # Handle 304 Not Modified
            if response.status_code == 304:
                logger.debug(f"Feed not modified: {feed_url}")
                return []
            
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
            
            # Update feed state
            self.feed_states[feed_url] = {
                'etag': response.headers.get('ETag'),
                'last_modified': response.headers.get('Last-Modified'),
                'last_fetched': datetime.utcnow().isoformat()
            }
            
            # Cache feed state
            self.cache_data(f"feed_state:{feed_url}", self.feed_states[feed_url])
            
            # Process feed items
            new_items = []
            for item in feed.entries:
                item_data = self._parse_feed_item(item, feed_url)
                item_id = item_data['id']
                
                # Check if item is new
                if item_id not in self.processed_items:
                    self.processed_items.add(item_id)
                    new_items.append(item_data)
                    
                    # Cache processed item ID
                    self.cache_data(f"processed_item:{item_id}", True, ttl=86400)  # 24 hours
            
            logger.info(f"Fetched {len(new_items)} new items from {feed_url}")
            return new_items
            
        except requests.RequestException as e:
            logger.error(f"HTTP error fetching feed {feed_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing feed {feed_url}: {e}")
            return None
    
    def collect_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Collect data from all configured RSS feeds.
        
        Yields:
            Dictionary containing feed item data
        """
        for feed_url in self.feeds:
            if not self.is_running:
                break
                
            try:
                items = self._fetch_feed(feed_url)
                if items:
                    for item in items:
                        yield item
                        
            except Exception as e:
                logger.error(f"Error collecting data from {feed_url}: {e}")
                self.error_count += 1
    
    def _load_processed_items(self):
        """Load previously processed item IDs from cache."""
        try:
            # This is a simplified version - in production, you might want to
            # load from a persistent store or use a more sophisticated approach
            logger.info("Loading processed items from cache")
            # For now, we'll start fresh each time
            self.processed_items = set()
        except Exception as e:
            logger.error(f"Error loading processed items: {e}")
    
    def run(self):
        """
        Main execution loop for RSS connector.
        
        Continuously polls RSS feeds at configured intervals.
        """
        self.start()
        self._load_processed_items()
        
        logger.info(f"Starting RSS connector with {len(self.feeds)} feeds")
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Collect and publish data
                for item_data in self.collect_data():
                    if not self.is_running:
                        break
                    
                    # Publish to Kafka
                    success = self.publish_message(item_data, key=item_data['id'])
                    if success:
                        logger.debug(f"Published RSS item: {item_data['title'][:50]}...")
                
                # Calculate sleep time to maintain polling interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed_time)
                
                if sleep_time > 0 and self.is_running:
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("RSS connector interrupted by user")
        except Exception as e:
            logger.error(f"Error in RSS connector main loop: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run RSS connector
    connector = RSSConnector()
    connector.run()

