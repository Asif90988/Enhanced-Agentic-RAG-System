"""
Data Structurer for Organizing Processed Data

Converts processed NLP results into structured formats suitable for storage
and retrieval by the RAG system.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class DataStructurer:
    """
    Data structurer for converting processed NLP results into structured formats.
    
    Features:
    - Schema normalization across different data sources
    - Metadata enrichment and standardization
    - Data quality validation
    - Relationship mapping
    - Vector embedding organization
    """
    
    def __init__(self):
        """Initialize data structurer."""
        self.schema_version = "1.0"
        logger.info("Data structurer initialized")
    
    def _generate_document_id(self, source_data: Dict[str, Any]) -> str:
        """
        Generate unique document ID.
        
        Args:
            source_data: Original source data
            
        Returns:
            Unique document identifier
        """
        # Use existing ID if available
        if 'id' in source_data.get('data', {}):
            return source_data['data']['id']
        
        # Generate ID based on content and timestamp
        content = str(source_data.get('data', {}))
        timestamp = source_data.get('timestamp', datetime.utcnow().isoformat())
        
        id_string = f"{content}:{timestamp}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def _extract_source_metadata(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize source metadata.
        
        Args:
            source_data: Original source data
            
        Returns:
            Normalized metadata
        """
        data = source_data.get('data', {})
        connector = source_data.get('connector', 'unknown')
        
        metadata = {
            'source_type': connector,
            'collected_at': source_data.get('timestamp'),
            'schema_version': self.schema_version
        }
        
        # Extract connector-specific metadata
        if connector == 'rss_connector':
            metadata.update({
                'source_feed': data.get('source_feed'),
                'published': data.get('published'),
                'author': data.get('author'),
                'tags': data.get('tags', []),
                'link': data.get('link')
            })
        
        elif connector == 'news_connector':
            metadata.update({
                'source_name': data.get('source_name'),
                'source_id': data.get('source_id'),
                'published_at': data.get('published_at'),
                'author': data.get('author'),
                'url': data.get('url'),
                'url_to_image': data.get('url_to_image')
            })
        
        elif connector == 'file_connector':
            file_metadata = data.get('metadata', {})
            metadata.update({
                'filename': file_metadata.get('filename'),
                'file_path': file_metadata.get('file_path'),
                'file_size': file_metadata.get('file_size'),
                'file_extension': file_metadata.get('file_extension'),
                'mime_type': file_metadata.get('mime_type'),
                'created_time': file_metadata.get('created_time'),
                'modified_time': file_metadata.get('modified_time'),
                'source_directory': data.get('source_directory')
            })
        
        return metadata
    
    def _structure_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Structure entity data for storage.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Structured entity data
        """
        structured_entities = []
        
        for entity in entities:
            structured_entity = {
                'id': str(uuid.uuid4()),
                'text': entity.get('text', ''),
                'label': entity.get('label', ''),
                'description': entity.get('description', ''),
                'start_position': entity.get('start', 0),
                'end_position': entity.get('end', 0),
                'confidence': entity.get('confidence', 1.0),
                'entity_type': self._normalize_entity_type(entity.get('label', ''))
            }
            structured_entities.append(structured_entity)
        
        return structured_entities
    
    def _normalize_entity_type(self, label: str) -> str:
        """
        Normalize entity type labels.
        
        Args:
            label: Original entity label
            
        Returns:
            Normalized entity type
        """
        label_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'monetary',
            'PERCENT': 'percentage',
            'EMAIL': 'email',
            'URL': 'url',
            'PHONE': 'phone'
        }
        
        return label_mapping.get(label.upper(), label.lower())
    
    def _structure_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Structure relationship data for storage.
        
        Args:
            relationships: List of extracted relationships
            
        Returns:
            Structured relationship data
        """
        structured_relationships = []
        
        for relationship in relationships:
            structured_relationship = {
                'id': str(uuid.uuid4()),
                'subject': relationship.get('subject', ''),
                'predicate': relationship.get('predicate', ''),
                'object': relationship.get('object', ''),
                'confidence': relationship.get('confidence', 1.0),
                'relationship_type': 'semantic'
            }
            structured_relationships.append(structured_relationship)
        
        return structured_relationships
    
    def _structure_content(self, nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure content data for storage.
        
        Args:
            nlp_results: NLP processing results
            
        Returns:
            Structured content data
        """
        return {
            'original_text': nlp_results.get('original_text', ''),
            'cleaned_text': nlp_results.get('cleaned_text', ''),
            'summary': nlp_results.get('summary', ''),
            'language': nlp_results.get('language', 'en'),
            'word_count': nlp_results.get('word_count', 0),
            'character_count': nlp_results.get('character_count', 0),
            'chunks': nlp_results.get('chunks', [])
        }
    
    def _structure_analysis(self, nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure analysis results for storage.
        
        Args:
            nlp_results: NLP processing results
            
        Returns:
            Structured analysis data
        """
        sentiment = nlp_results.get('sentiment', {})
        key_phrases = nlp_results.get('key_phrases', [])
        
        return {
            'sentiment': {
                'label': sentiment.get('sentiment', 'neutral'),
                'confidence': sentiment.get('confidence', 0.0),
                'scores': sentiment.get('scores', {})
            },
            'key_phrases': [
                {
                    'text': phrase.get('text', ''),
                    'type': phrase.get('type', ''),
                    'importance': phrase.get('importance', 0)
                }
                for phrase in key_phrases
            ],
            'topics': [],  # Could be added later
            'categories': []  # Could be added later
        }
    
    def _structure_embeddings(self, nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure embedding data for storage.
        
        Args:
            nlp_results: NLP processing results
            
        Returns:
            Structured embedding data
        """
        embeddings = nlp_results.get('embeddings', [])
        chunks = nlp_results.get('chunks', [])
        
        embedding_data = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # From config
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'chunk_embeddings': []
        }
        
        # Pair chunks with their embeddings
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                embedding_data['chunk_embeddings'].append({
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_text': chunk,
                    'chunk_index': i,
                    'embedding_vector': embeddings[i]
                })
        
        return embedding_data
    
    def _validate_structured_data(self, structured_data: Dict[str, Any]) -> bool:
        """
        Validate structured data quality.
        
        Args:
            structured_data: Structured data to validate
            
        Returns:
            True if data is valid
        """
        required_fields = ['id', 'metadata', 'content', 'entities', 'analysis', 'embeddings']
        
        for field in required_fields:
            if field not in structured_data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate content
        content = structured_data.get('content', {})
        if not content.get('cleaned_text'):
            logger.warning("No cleaned text content available")
            return False
        
        # Validate embeddings
        embeddings = structured_data.get('embeddings', {})
        if not embeddings.get('chunk_embeddings'):
            logger.warning("No embeddings available")
            return False
        
        return True
    
    def structure_data(self, source_data: Dict[str, Any], nlp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure data from source and NLP results into standardized format.
        
        Args:
            source_data: Original source data
            nlp_results: NLP processing results
            
        Returns:
            Structured data ready for storage
        """
        try:
            # Generate document ID
            document_id = self._generate_document_id(source_data)
            
            # Extract and structure components
            metadata = self._extract_source_metadata(source_data)
            entities = self._structure_entities(nlp_results.get('entities', []))
            relationships = self._structure_relationships(nlp_results.get('relationships', []))
            content = self._structure_content(nlp_results)
            analysis = self._structure_analysis(nlp_results)
            embeddings = self._structure_embeddings(nlp_results)
            
            # Create structured document
            structured_data = {
                'id': document_id,
                'metadata': metadata,
                'content': content,
                'entities': entities,
                'relationships': relationships,
                'analysis': analysis,
                'embeddings': embeddings,
                'processing_info': {
                    'processed_at': nlp_results.get('processed_at'),
                    'schema_version': self.schema_version,
                    'processing_pipeline': 'enhanced_agentic_rag'
                }
            }
            
            # Validate structured data
            if not self._validate_structured_data(structured_data):
                logger.error(f"Data validation failed for document {document_id}")
                return None
            
            logger.debug(f"Successfully structured document {document_id}")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error structuring data: {e}")
            return None
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the data schema.
        
        Returns:
            Schema information
        """
        return {
            'schema_version': self.schema_version,
            'supported_sources': ['rss_connector', 'news_connector', 'file_connector'],
            'entity_types': ['person', 'organization', 'location', 'date', 'time', 'monetary', 'percentage', 'email', 'url', 'phone'],
            'analysis_features': ['sentiment', 'key_phrases', 'topics', 'categories'],
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }

