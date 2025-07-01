"""
News API Connector

Collects real-time news data from various news APIs.
Supports multiple news sources and intelligent deduplication.
"""

import logging
import time
import hashlib
from typing import Dict, Any, List, Generator, Optional
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .base_connector import BaseConnector
from config import CONFIG

logger = logging.getLogger(__name__)

class NewsConnector(BaseConnector):
    """
    News API connector for real-time news data ingestion.
    
    Features:
    - Multiple news API support (NewsAPI, Guardian, etc.)
    - Intelligent deduplication across sources
    - Category and keyword filtering
    - Rate limiting compliance
    - Content extraction and enrichment
    """
    
    def __init__(self):
        super().__init__("news_connector", CONFIG.kafka.topics["news_articles"])
        
        self.api_key = CONFIG.data_sources.news_api_key
        self.sources = CONFIG.data_sources.news_sources
        self.poll_interval = 300  # 5 minutes for news
        
        # Initialize HTTP session
        self.session = self._init_http_session()
        
        # Track processed articles
        self.processed_articles = set()
        
        # News API endpoints
        self.news_api_base = "https://newsapi.org/v2"
        
        logger.info(f"News connector initialized with {len(self.sources)} sources")
    
    def _init_http_session(self) -> requests.Session:
        """Initialize HTTP session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            'User-Agent': 'Enhanced-Agentic-RAG/1.0 (News Connector)',
            'X-API-Key': self.api_key if self.api_key else ''
        })
        
        return session
    
    def _get_article_id(self, article: Dict[str, Any]) -> str:
        """
        Generate unique ID for news article.
        
        Args:
            article: Article data
            
        Returns:
            Unique identifier for the article
        """
        # Use URL if available, otherwise use title + published date
        if article.get('url'):
            return hashlib.md5(article['url'].encode()).hexdigest()
        else:
            content = f"{article.get('title', '')}{article.get('publishedAt', '')}"
            return hashlib.md5(content.encode()).hexdigest()
    
    def _normalize_article(self, article: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalize article data from different APIs.
        
        Args:
            article: Raw article data
            source: Source identifier
            
        Returns:
            Normalized article data
        """
        return {
            'id': self._get_article_id(article),
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'content': article.get('content', ''),
            'url': article.get('url', ''),
            'url_to_image': article.get('urlToImage', ''),
            'published_at': article.get('publishedAt', ''),
            'source_name': article.get('source', {}).get('name', source),
            'source_id': article.get('source', {}).get('id', source),
            'author': article.get('author', ''),
            'source_type': 'news_api',
            'collected_at': datetime.utcnow().isoformat()
        }
    
    def _fetch_newsapi_headlines(self, sources: str = None, category: str = None) -> List[Dict[str, Any]]:
        """
        Fetch headlines from NewsAPI.
        
        Args:
            sources: Comma-separated list of source IDs
            category: News category
            
        Returns:
            List of article data
        """
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        try:
            params = {
                'country': 'us',
                'pageSize': 100,
                'sortBy': 'publishedAt'
            }
            
            if sources:
                params['sources'] = sources
                # Remove country when using sources (NewsAPI restriction)
                del params['country']
            
            if category:
                params['category'] = category
            
            response = self.session.get(
                f"{self.news_api_base}/top-headlines",
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = []
            for article in data.get('articles', []):
                normalized = self._normalize_article(article, 'newsapi')
                article_id = normalized['id']
                
                if article_id not in self.processed_articles:
                    self.processed_articles.add(article_id)
                    articles.append(normalized)
            
            logger.info(f"Fetched {len(articles)} new articles from NewsAPI")
            return articles
            
        except requests.RequestException as e:
            logger.error(f"HTTP error fetching NewsAPI headlines: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing NewsAPI headlines: {e}")
            return []
    
    def _fetch_everything_newsapi(self, query: str = None, sources: str = None) -> List[Dict[str, Any]]:
        """
        Fetch articles from NewsAPI everything endpoint.
        
        Args:
            query: Search query
            sources: Comma-separated list of source IDs
            
        Returns:
            List of article data
        """
        if not self.api_key:
            return []
        
        try:
            # Get articles from last hour
            from_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            
            params = {
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'from': from_time
            }
            
            if query:
                params['q'] = query
            
            if sources:
                params['sources'] = sources
            
            response = self.session.get(
                f"{self.news_api_base}/everything",
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI everything error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = []
            for article in data.get('articles', []):
                normalized = self._normalize_article(article, 'newsapi')
                article_id = normalized['id']
                
                if article_id not in self.processed_articles:
                    self.processed_articles.add(article_id)
                    articles.append(normalized)
            
            logger.info(f"Fetched {len(articles)} new articles from NewsAPI everything")
            return articles
            
        except requests.RequestException as e:
            logger.error(f"HTTP error fetching NewsAPI everything: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing NewsAPI everything: {e}")
            return []
    
    def _extract_full_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract full article content from URL.
        
        Args:
            article: Article data with URL
            
        Returns:
            Article data with extracted content
        """
        if not article.get('url'):
            return article
        
        try:
            # Simple content extraction - in production, you might want to use
            # more sophisticated tools like newspaper3k or readability
            response = self.session.get(article['url'], timeout=15)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Update article with extracted content
            article['full_content'] = text[:5000]  # Limit content length
            article['content_extracted'] = True
            
        except Exception as e:
            logger.debug(f"Could not extract content from {article['url']}: {e}")
            article['content_extracted'] = False
        
        return article
    
    def collect_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        Collect data from news APIs.
        
        Yields:
            Dictionary containing article data
        """
        try:
            # Fetch headlines
            articles = self._fetch_newsapi_headlines(sources=','.join(self.sources))
            
            for article in articles:
                if not self.is_running:
                    break
                
                # Extract full content if needed
                if article.get('content') and len(article['content']) < 200:
                    article = self._extract_full_content(article)
                
                yield article
            
            # Fetch recent articles with general queries
            queries = ['technology', 'artificial intelligence', 'business', 'science']
            for query in queries:
                if not self.is_running:
                    break
                
                articles = self._fetch_everything_newsapi(query=query)
                for article in articles:
                    if not self.is_running:
                        break
                    
                    # Extract full content if needed
                    if article.get('content') and len(article['content']) < 200:
                        article = self._extract_full_content(article)
                    
                    yield article
                    
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            self.error_count += 1
    
    def run(self):
        """
        Main execution loop for news connector.
        
        Continuously fetches news articles at configured intervals.
        """
        self.start()
        
        logger.info(f"Starting news connector with {len(self.sources)} sources")
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Collect and publish data
                for article_data in self.collect_data():
                    if not self.is_running:
                        break
                    
                    # Publish to Kafka
                    success = self.publish_message(article_data, key=article_data['id'])
                    if success:
                        logger.debug(f"Published news article: {article_data['title'][:50]}...")
                
                # Calculate sleep time
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.poll_interval - elapsed_time)
                
                if sleep_time > 0 and self.is_running:
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("News connector interrupted by user")
        except Exception as e:
            logger.error(f"Error in news connector main loop: {e}")
        finally:
            self.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run news connector
    connector = NewsConnector()
    connector.run()

