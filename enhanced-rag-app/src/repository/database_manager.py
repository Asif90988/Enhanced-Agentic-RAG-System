"""
Database Manager for Structured Data Storage

Manages PostgreSQL database operations for storing structured document metadata,
entities, relationships, and analysis results.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, Integer, Float, DateTime, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import uuid
from config import CONFIG

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for structured data storage using PostgreSQL.
    
    Features:
    - Document metadata storage
    - Entity and relationship management
    - Analysis results storage
    - Query optimization and indexing
    - Transaction management
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.config = CONFIG
        self.engine = None
        self.metadata = MetaData()
        self.Session = None
        
        # Initialize database connection
        self._init_database()
        
        # Define database schema
        self._define_schema()
        
        # Create tables
        self._create_tables()
        
        logger.info("Database manager initialized")
    
    def _init_database(self):
        """Initialize database connection."""
        try:
            self.engine = create_engine(
                self.config.database.postgres_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _define_schema(self):
        """Define database schema."""
        
        # Documents table
        self.documents_table = Table(
            'documents',
            self.metadata,
            Column('id', String(64), primary_key=True),
            Column('source_type', String(50), nullable=False),
            Column('source_metadata', JSON),
            Column('title', Text),
            Column('content_text', Text),
            Column('content_summary', Text),
            Column('language', String(10), default='en'),
            Column('word_count', Integer, default=0),
            Column('character_count', Integer, default=0),
            Column('processed_at', DateTime, default=datetime.utcnow),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
            Column('is_active', Boolean, default=True)
        )
        
        # Entities table
        self.entities_table = Table(
            'entities',
            self.metadata,
            Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column('document_id', String(64), sa.ForeignKey('documents.id'), nullable=False),
            Column('text', Text, nullable=False),
            Column('label', String(50), nullable=False),
            Column('entity_type', String(50), nullable=False),
            Column('description', Text),
            Column('start_position', Integer),
            Column('end_position', Integer),
            Column('confidence', Float, default=1.0),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Relationships table
        self.relationships_table = Table(
            'relationships',
            self.metadata,
            Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column('document_id', String(64), sa.ForeignKey('documents.id'), nullable=False),
            Column('subject', Text, nullable=False),
            Column('predicate', Text, nullable=False),
            Column('object', Text),
            Column('relationship_type', String(50), default='semantic'),
            Column('confidence', Float, default=1.0),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Analysis results table
        self.analysis_table = Table(
            'analysis_results',
            self.metadata,
            Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column('document_id', String(64), sa.ForeignKey('documents.id'), nullable=False),
            Column('sentiment_label', String(20)),
            Column('sentiment_confidence', Float),
            Column('sentiment_scores', JSON),
            Column('key_phrases', JSON),
            Column('topics', JSON),
            Column('categories', JSON),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Document chunks table (for vector embeddings reference)
        self.chunks_table = Table(
            'document_chunks',
            self.metadata,
            Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
            Column('document_id', String(64), sa.ForeignKey('documents.id'), nullable=False),
            Column('chunk_index', Integer, nullable=False),
            Column('chunk_text', Text, nullable=False),
            Column('embedding_id', String(64)),  # Reference to vector store
            Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    def _create_tables(self):
        """Create database tables."""
        try:
            self.metadata.create_all(self.engine)
            
            # Create indexes for better query performance
            self._create_indexes()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for query optimization."""
        try:
            with self.engine.connect() as conn:
                # Index on document source_type and created_at
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_documents_source_type 
                    ON documents(source_type)
                """))
                
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_documents_created_at 
                    ON documents(created_at DESC)
                """))
                
                # Index on entities for fast lookup
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_entities_document_id 
                    ON entities(document_id)
                """))
                
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_entities_type 
                    ON entities(entity_type)
                """))
                
                # Index on relationships
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_relationships_document_id 
                    ON relationships(document_id)
                """))
                
                # Index on chunks
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                    ON document_chunks(document_id)
                """))
                
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def store_document(self, structured_data: Dict[str, Any]) -> bool:
        """
        Store structured document data in the database.
        
        Args:
            structured_data: Structured document data
            
        Returns:
            True if stored successfully
        """
        session = self.Session()
        try:
            # Extract data components
            doc_id = structured_data['id']
            metadata = structured_data['metadata']
            content = structured_data['content']
            entities = structured_data['entities']
            relationships = structured_data['relationships']
            analysis = structured_data['analysis']
            embeddings = structured_data['embeddings']
            
            # Store document
            doc_data = {
                'id': doc_id,
                'source_type': metadata.get('source_type', 'unknown'),
                'source_metadata': metadata,
                'title': self._extract_title(content, metadata),
                'content_text': content.get('cleaned_text', ''),
                'content_summary': content.get('summary', ''),
                'language': content.get('language', 'en'),
                'word_count': content.get('word_count', 0),
                'character_count': content.get('character_count', 0),
                'processed_at': datetime.fromisoformat(structured_data['processing_info']['processed_at'].replace('Z', '+00:00'))
            }
            
            # Insert or update document
            stmt = sa.dialects.postgresql.insert(self.documents_table).values(**doc_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_=dict(
                    source_metadata=stmt.excluded.source_metadata,
                    content_text=stmt.excluded.content_text,
                    content_summary=stmt.excluded.content_summary,
                    updated_at=datetime.utcnow()
                )
            )
            session.execute(stmt)
            
            # Store entities
            for entity in entities:
                entity_data = {
                    'document_id': doc_id,
                    'text': entity['text'],
                    'label': entity['label'],
                    'entity_type': entity['entity_type'],
                    'description': entity['description'],
                    'start_position': entity['start_position'],
                    'end_position': entity['end_position'],
                    'confidence': entity['confidence']
                }
                session.execute(self.entities_table.insert().values(**entity_data))
            
            # Store relationships
            for relationship in relationships:
                rel_data = {
                    'document_id': doc_id,
                    'subject': relationship['subject'],
                    'predicate': relationship['predicate'],
                    'object': relationship['object'],
                    'relationship_type': relationship['relationship_type'],
                    'confidence': relationship['confidence']
                }
                session.execute(self.relationships_table.insert().values(**rel_data))
            
            # Store analysis results
            analysis_data = {
                'document_id': doc_id,
                'sentiment_label': analysis['sentiment']['label'],
                'sentiment_confidence': analysis['sentiment']['confidence'],
                'sentiment_scores': analysis['sentiment']['scores'],
                'key_phrases': analysis['key_phrases'],
                'topics': analysis['topics'],
                'categories': analysis['categories']
            }
            session.execute(self.analysis_table.insert().values(**analysis_data))
            
            # Store document chunks
            for chunk_data in embeddings['chunk_embeddings']:
                chunk_info = {
                    'document_id': doc_id,
                    'chunk_index': chunk_data['chunk_index'],
                    'chunk_text': chunk_data['chunk_text'],
                    'embedding_id': chunk_data['chunk_id']
                }
                session.execute(self.chunks_table.insert().values(**chunk_info))
            
            session.commit()
            logger.debug(f"Stored document {doc_id} in database")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error storing document: {e}")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing document: {e}")
            return False
        finally:
            session.close()
    
    def _extract_title(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Extract title from content or metadata."""
        # Try to get title from metadata first
        if 'title' in metadata:
            return metadata['title']
        
        # Try to extract from content
        text = content.get('cleaned_text', '')
        if text:
            # Take first line or first 100 characters as title
            lines = text.split('\n')
            if lines:
                return lines[0][:100]
        
        return f"Document {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        session = self.Session()
        try:
            # Get document
            doc_result = session.execute(
                sa.select(self.documents_table).where(self.documents_table.c.id == document_id)
            ).fetchone()
            
            if not doc_result:
                return None
            
            # Get entities
            entities_result = session.execute(
                sa.select(self.entities_table).where(self.entities_table.c.document_id == document_id)
            ).fetchall()
            
            # Get relationships
            relationships_result = session.execute(
                sa.select(self.relationships_table).where(self.relationships_table.c.document_id == document_id)
            ).fetchall()
            
            # Get analysis
            analysis_result = session.execute(
                sa.select(self.analysis_table).where(self.analysis_table.c.document_id == document_id)
            ).fetchone()
            
            # Get chunks
            chunks_result = session.execute(
                sa.select(self.chunks_table).where(self.chunks_table.c.document_id == document_id)
            ).fetchall()
            
            # Assemble document data
            document = {
                'id': doc_result.id,
                'source_type': doc_result.source_type,
                'source_metadata': doc_result.source_metadata,
                'title': doc_result.title,
                'content_text': doc_result.content_text,
                'content_summary': doc_result.content_summary,
                'language': doc_result.language,
                'word_count': doc_result.word_count,
                'character_count': doc_result.character_count,
                'processed_at': doc_result.processed_at,
                'created_at': doc_result.created_at,
                'entities': [dict(row._mapping) for row in entities_result],
                'relationships': [dict(row._mapping) for row in relationships_result],
                'analysis': dict(analysis_result._mapping) if analysis_result else {},
                'chunks': [dict(row._mapping) for row in chunks_result]
            }
            
            return document
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None
        finally:
            session.close()
    
    def search_documents(self, query: str, source_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by text query.
        
        Args:
            query: Search query
            source_type: Filter by source type
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        session = self.Session()
        try:
            # Build query
            stmt = sa.select(self.documents_table).where(
                sa.or_(
                    self.documents_table.c.title.ilike(f'%{query}%'),
                    self.documents_table.c.content_text.ilike(f'%{query}%'),
                    self.documents_table.c.content_summary.ilike(f'%{query}%')
                )
            )
            
            if source_type:
                stmt = stmt.where(self.documents_table.c.source_type == source_type)
            
            stmt = stmt.order_by(self.documents_table.c.created_at.desc()).limit(limit)
            
            results = session.execute(stmt).fetchall()
            
            return [dict(row._mapping) for row in results]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
        finally:
            session.close()
    
    def get_recent_documents(self, limit: int = 10, source_type: str = None) -> List[Dict[str, Any]]:
        """
        Get recently processed documents.
        
        Args:
            limit: Maximum number of documents
            source_type: Filter by source type
            
        Returns:
            List of recent documents
        """
        session = self.Session()
        try:
            stmt = sa.select(self.documents_table).where(self.documents_table.c.is_active == True)
            
            if source_type:
                stmt = stmt.where(self.documents_table.c.source_type == source_type)
            
            stmt = stmt.order_by(self.documents_table.c.created_at.desc()).limit(limit)
            
            results = session.execute(stmt).fetchall()
            
            return [dict(row._mapping) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting recent documents: {e}")
            return []
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing statistics
        """
        session = self.Session()
        try:
            # Document counts
            total_docs = session.execute(
                sa.select(sa.func.count(self.documents_table.c.id))
            ).scalar()
            
            # Documents by source type
            source_counts = session.execute(
                sa.select(
                    self.documents_table.c.source_type,
                    sa.func.count(self.documents_table.c.id)
                ).group_by(self.documents_table.c.source_type)
            ).fetchall()
            
            # Entity counts
            total_entities = session.execute(
                sa.select(sa.func.count(self.entities_table.c.id))
            ).scalar()
            
            # Relationship counts
            total_relationships = session.execute(
                sa.select(sa.func.count(self.relationships_table.c.id))
            ).scalar()
            
            return {
                'total_documents': total_docs,
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'documents_by_source': {row[0]: row[1] for row in source_counts}
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        try:
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

