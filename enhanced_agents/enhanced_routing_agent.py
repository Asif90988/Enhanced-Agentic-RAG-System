"""
Enhanced Routing Agent with Real-time Data Integration

Routes queries to appropriate data sources and search strategies based on
query analysis and real-time data availability.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from storage.document_repository import DocumentRepository
from processing.nlp_pipeline import NLPPipeline

logger = logging.getLogger(__name__)

class EnhancedRoutingAgent:
    """
    Enhanced routing agent that determines optimal data sources and search strategies.
    
    Features:
    - Query intent analysis
    - Data source selection based on recency and relevance
    - Search strategy optimization
    - Real-time data prioritization
    - Performance-based routing decisions
    """
    
    def __init__(self, document_repository: DocumentRepository):
        """Initialize enhanced routing agent."""
        self.document_repository = document_repository
        self.nlp_pipeline = NLPPipeline()
        
        # Routing configuration
        self.source_priorities = {
            'news_connector': 0.9,
            'rss_connector': 0.8,
            'file_connector': 0.7
        }
        
        self.search_strategies = {
            'factual': 'hybrid',
            'recent': 'semantic',
            'analytical': 'text',
            'exploratory': 'semantic'
        }
        
        logger.info("Enhanced routing agent initialized")
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query to appropriate data sources and search strategies.
        
        Args:
            query: User query
            context: Additional context information
            
        Returns:
            Routing decision with recommended sources and strategies
        """
        try:
            # Analyze query intent and characteristics
            query_analysis = self._analyze_query(query)
            
            # Determine data source preferences
            source_preferences = self._determine_source_preferences(query_analysis, context)
            
            # Select search strategy
            search_strategy = self._select_search_strategy(query_analysis)
            
            # Check real-time data availability
            real_time_availability = self._check_real_time_data(query_analysis)
            
            # Generate routing decision
            routing_decision = {
                'query': query,
                'query_analysis': query_analysis,
                'recommended_sources': source_preferences,
                'search_strategy': search_strategy,
                'real_time_data': real_time_availability,
                'routing_confidence': self._calculate_routing_confidence(query_analysis),
                'estimated_response_time': self._estimate_response_time(search_strategy, source_preferences),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.debug(f"Query routed with strategy: {search_strategy}")
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return self._get_fallback_routing(query)
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to understand intent and characteristics.
        
        Args:
            query: User query
            
        Returns:
            Query analysis results
        """
        try:
            # Process query through NLP pipeline
            nlp_results = self.nlp_pipeline.process_text(query)
            
            # Extract temporal indicators
            temporal_indicators = self._extract_temporal_indicators(query, nlp_results.get('entities', []))
            
            # Determine query type
            query_type = self._classify_query_type(query, nlp_results)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(nlp_results.get('entities', []), nlp_results.get('key_phrases', []))
            
            # Analyze complexity
            complexity = self._analyze_query_complexity(query, nlp_results)
            
            return {
                'query_type': query_type,
                'temporal_indicators': temporal_indicators,
                'key_concepts': key_concepts,
                'complexity': complexity,
                'entities': nlp_results.get('entities', []),
                'sentiment': nlp_results.get('sentiment', {}),
                'language': nlp_results.get('language', 'en'),
                'word_count': nlp_results.get('word_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {'query_type': 'general', 'complexity': 'medium'}
    
    def _extract_temporal_indicators(self, query: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract temporal information from query."""
        temporal_keywords = {
            'recent': ['recent', 'latest', 'new', 'current', 'today', 'now'],
            'past': ['yesterday', 'last week', 'last month', 'ago', 'previous'],
            'future': ['tomorrow', 'next', 'upcoming', 'future', 'will'],
            'specific': ['date', 'time', 'when', 'during']
        }
        
        query_lower = query.lower()
        temporal_info = {
            'has_temporal': False,
            'temporal_type': None,
            'recency_required': False,
            'date_entities': []
        }
        
        # Check for temporal keywords
        for temp_type, keywords in temporal_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                temporal_info['has_temporal'] = True
                temporal_info['temporal_type'] = temp_type
                if temp_type == 'recent':
                    temporal_info['recency_required'] = True
                break
        
        # Extract date entities
        for entity in entities:
            if entity.get('entity_type') in ['date', 'time']:
                temporal_info['date_entities'].append(entity)
                temporal_info['has_temporal'] = True
        
        return temporal_info
    
    def _classify_query_type(self, query: str, nlp_results: Dict[str, Any]) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Question words
        question_patterns = {
            'factual': ['what', 'who', 'where', 'when', 'which'],
            'analytical': ['why', 'how', 'analyze', 'compare', 'explain'],
            'exploratory': ['find', 'search', 'discover', 'explore', 'show me'],
            'recent': ['latest', 'recent', 'current', 'new', 'update']
        }
        
        for query_type, patterns in question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        return 'general'
    
    def _extract_key_concepts(self, entities: List[Dict[str, Any]], key_phrases: List[Dict[str, Any]]) -> List[str]:
        """Extract key concepts from entities and phrases."""
        concepts = []
        
        # Add important entities
        for entity in entities:
            if entity.get('entity_type') in ['person', 'organization', 'location']:
                concepts.append(entity.get('text', ''))
        
        # Add key phrases
        for phrase in key_phrases[:5]:  # Top 5 phrases
            if phrase.get('importance', 0) > 2:
                concepts.append(phrase.get('text', ''))
        
        return list(set(concepts))  # Remove duplicates
    
    def _analyze_query_complexity(self, query: str, nlp_results: Dict[str, Any]) -> str:
        """Analyze query complexity."""
        word_count = nlp_results.get('word_count', 0)
        entity_count = len(nlp_results.get('entities', []))
        
        if word_count < 5 and entity_count < 2:
            return 'simple'
        elif word_count < 15 and entity_count < 5:
            return 'medium'
        else:
            return 'complex'
    
    def _determine_source_preferences(self, query_analysis: Dict[str, Any], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Determine preferred data sources based on query analysis."""
        preferences = []
        
        # Base preferences on query type and temporal requirements
        query_type = query_analysis.get('query_type', 'general')
        temporal_info = query_analysis.get('temporal_indicators', {})
        
        if temporal_info.get('recency_required', False):
            # Prioritize real-time sources for recent queries
            preferences.extend([
                {'source_type': 'news_connector', 'priority': 0.9, 'reason': 'recent_news'},
                {'source_type': 'rss_connector', 'priority': 0.8, 'reason': 'live_feeds'},
                {'source_type': 'file_connector', 'priority': 0.3, 'reason': 'static_files'}
            ])
        elif query_type == 'analytical':
            # Prioritize comprehensive sources for analytical queries
            preferences.extend([
                {'source_type': 'file_connector', 'priority': 0.9, 'reason': 'detailed_documents'},
                {'source_type': 'news_connector', 'priority': 0.7, 'reason': 'news_analysis'},
                {'source_type': 'rss_connector', 'priority': 0.6, 'reason': 'feed_content'}
            ])
        else:
            # Balanced approach for general queries
            preferences.extend([
                {'source_type': 'news_connector', 'priority': 0.8, 'reason': 'current_information'},
                {'source_type': 'file_connector', 'priority': 0.7, 'reason': 'comprehensive_content'},
                {'source_type': 'rss_connector', 'priority': 0.6, 'reason': 'diverse_sources'}
            ])
        
        # Sort by priority
        preferences.sort(key=lambda x: x['priority'], reverse=True)
        
        return preferences
    
    def _select_search_strategy(self, query_analysis: Dict[str, Any]) -> str:
        """Select optimal search strategy based on query analysis."""
        query_type = query_analysis.get('query_type', 'general')
        complexity = query_analysis.get('complexity', 'medium')
        
        # Strategy selection logic
        if query_type in self.search_strategies:
            base_strategy = self.search_strategies[query_type]
        else:
            base_strategy = 'hybrid'
        
        # Adjust based on complexity
        if complexity == 'simple' and base_strategy == 'hybrid':
            return 'semantic'  # Simple queries work well with semantic search
        elif complexity == 'complex' and base_strategy == 'semantic':
            return 'hybrid'  # Complex queries benefit from hybrid approach
        
        return base_strategy
    
    def _check_real_time_data(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check availability of real-time data relevant to the query."""
        try:
            # Get recent documents from preferred sources
            recent_docs = self.document_repository.get_recent_documents(limit=50)
            
            # Analyze relevance to query
            key_concepts = query_analysis.get('key_concepts', [])
            relevant_docs = []
            
            for doc in recent_docs:
                relevance_score = self._calculate_document_relevance(doc, key_concepts)
                if relevance_score > 0.3:
                    relevant_docs.append({
                        'document_id': doc['id'],
                        'source_type': doc['source_type'],
                        'relevance_score': relevance_score,
                        'created_at': doc['created_at']
                    })
            
            # Sort by relevance and recency
            relevant_docs.sort(key=lambda x: (x['relevance_score'], x['created_at']), reverse=True)
            
            return {
                'available': len(relevant_docs) > 0,
                'document_count': len(relevant_docs),
                'top_documents': relevant_docs[:10],
                'freshness_score': self._calculate_freshness_score(relevant_docs),
                'coverage_score': self._calculate_coverage_score(relevant_docs, key_concepts)
            }
            
        except Exception as e:
            logger.error(f"Error checking real-time data: {e}")
            return {'available': False, 'document_count': 0}
    
    def _calculate_document_relevance(self, document: Dict[str, Any], key_concepts: List[str]) -> float:
        """Calculate relevance of document to key concepts."""
        if not key_concepts:
            return 0.0
        
        doc_text = (
            document.get('title', '') + ' ' +
            document.get('content_summary', '') + ' ' +
            document.get('content_text', '')[:500]  # First 500 chars
        ).lower()
        
        matches = 0
        for concept in key_concepts:
            if concept.lower() in doc_text:
                matches += 1
        
        return matches / len(key_concepts)
    
    def _calculate_freshness_score(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate freshness score based on document recency."""
        if not documents:
            return 0.0
        
        now = datetime.utcnow()
        total_score = 0.0
        
        for doc in documents:
            created_at = doc.get('created_at')
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            # Calculate age in hours
            age_hours = (now - created_at).total_seconds() / 3600
            
            # Fresher documents get higher scores
            if age_hours < 1:
                score = 1.0
            elif age_hours < 24:
                score = 0.8
            elif age_hours < 168:  # 1 week
                score = 0.5
            else:
                score = 0.2
            
            total_score += score
        
        return total_score / len(documents)
    
    def _calculate_coverage_score(self, documents: List[Dict[str, Any]], key_concepts: List[str]) -> float:
        """Calculate how well documents cover the key concepts."""
        if not key_concepts or not documents:
            return 0.0
        
        covered_concepts = set()
        
        for doc in documents:
            for concept in key_concepts:
                if doc.get('relevance_score', 0) > 0.3:
                    covered_concepts.add(concept)
        
        return len(covered_concepts) / len(key_concepts)
    
    def _calculate_routing_confidence(self, query_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in routing decision."""
        factors = []
        
        # Query clarity
        word_count = query_analysis.get('word_count', 0)
        if word_count >= 3:
            factors.append(0.8)
        else:
            factors.append(0.4)
        
        # Entity presence
        entity_count = len(query_analysis.get('entities', []))
        if entity_count > 0:
            factors.append(0.9)
        else:
            factors.append(0.6)
        
        # Temporal clarity
        temporal_info = query_analysis.get('temporal_indicators', {})
        if temporal_info.get('has_temporal', False):
            factors.append(0.9)
        else:
            factors.append(0.7)
        
        return sum(factors) / len(factors)
    
    def _estimate_response_time(self, search_strategy: str, source_preferences: List[Dict[str, Any]]) -> float:
        """Estimate response time based on strategy and sources."""
        base_times = {
            'text': 0.5,
            'semantic': 1.0,
            'hybrid': 1.5
        }
        
        base_time = base_times.get(search_strategy, 1.0)
        
        # Adjust based on number of sources
        source_count = len(source_preferences)
        source_multiplier = 1.0 + (source_count - 1) * 0.2
        
        return base_time * source_multiplier
    
    def _get_fallback_routing(self, query: str) -> Dict[str, Any]:
        """Get fallback routing decision when analysis fails."""
        return {
            'query': query,
            'query_analysis': {'query_type': 'general', 'complexity': 'medium'},
            'recommended_sources': [
                {'source_type': 'news_connector', 'priority': 0.8, 'reason': 'fallback'},
                {'source_type': 'file_connector', 'priority': 0.7, 'reason': 'fallback'}
            ],
            'search_strategy': 'hybrid',
            'real_time_data': {'available': False, 'document_count': 0},
            'routing_confidence': 0.5,
            'estimated_response_time': 2.0,
            'timestamp': datetime.utcnow().isoformat(),
            'fallback': True
        }
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        try:
            # Get repository statistics
            repo_stats = self.document_repository.get_statistics()
            
            return {
                'total_documents': repo_stats.get('database', {}).get('total_documents', 0),
                'documents_by_source': repo_stats.get('database', {}).get('documents_by_source', {}),
                'vector_store_status': repo_stats.get('vector_store', {}),
                'supported_strategies': list(self.search_strategies.values()),
                'source_priorities': self.source_priorities
            }
            
        except Exception as e:
            logger.error(f"Error getting routing statistics: {e}")
            return {}

