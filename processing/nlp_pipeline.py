"""
NLP Pipeline for Text Processing and Structuring

Provides comprehensive natural language processing capabilities including:
- Text extraction and preprocessing
- Named entity recognition
- Relationship extraction
- Sentiment analysis
- Summarization
- Vector embedding generation
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from config import CONFIG

logger = logging.getLogger(__name__)

class NLPPipeline:
    """
    Comprehensive NLP pipeline for processing unstructured text data.
    
    Features:
    - Multi-language support
    - Named entity recognition and classification
    - Relationship extraction
    - Sentiment analysis and emotion detection
    - Text summarization
    - Key phrase extraction
    - Vector embedding generation
    - Content categorization
    """
    
    def __init__(self):
        """Initialize NLP pipeline with required models."""
        self.config = CONFIG.nlp
        
        # Initialize spaCy model
        self.nlp = self._load_spacy_model()
        
        # Initialize embedding model
        self.embedding_model = self._load_embedding_model()
        
        # Initialize sentiment analysis
        self.sentiment_analyzer = self._load_sentiment_analyzer()
        
        # Initialize summarization model
        self.summarizer = self._load_summarizer()
        
        # Text preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        
        logger.info("NLP pipeline initialized successfully")
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing."""
        try:
            nlp = spacy.load(self.config.spacy_model)
            
            # Add custom components if needed
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            
            logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
            return nlp
        except OSError:
            logger.error(f"spaCy model {self.config.spacy_model} not found. Please install it.")
            raise
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        try:
            model = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analysis pipeline."""
        try:
            analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Loaded sentiment analysis model")
            return analyzer
        except Exception as e:
            logger.warning(f"Failed to load sentiment analyzer: {e}")
            return None
    
    def _load_summarizer(self):
        """Load text summarization pipeline."""
        try:
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=30,
                do_sample=False
            )
            logger.info("Loaded summarization model")
            return summarizer
        except Exception as e:
            logger.warning(f"Failed to load summarizer: {e}")
            return None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entity dictionaries
        """
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': getattr(ent, 'confidence', 1.0)
                }
                entities.append(entity)
            
            # Extract additional patterns
            entities.extend(self._extract_pattern_entities(text))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_pattern_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        # URLs
        for match in self.url_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'URL',
                'description': 'Web URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        # Email addresses
        for match in self.email_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'EMAIL',
                'description': 'Email address',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        # Phone numbers
        for match in self.phone_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'label': 'PHONE',
                'description': 'Phone number',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relationship dictionaries
        """
        try:
            doc = self.nlp(text)
            relationships = []
            
            # Simple dependency-based relationship extraction
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj'] and token.head.pos_ == 'VERB':
                    relationship = {
                        'subject': token.text,
                        'predicate': token.head.text,
                        'object': None,
                        'confidence': 0.7
                    }
                    
                    # Find object
                    for child in token.head.children:
                        if child.dep_ in ['dobj', 'pobj'] and child != token:
                            relationship['object'] = child.text
                            break
                    
                    if relationship['object']:
                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        if not self.sentiment_analyzer or not text:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {}}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            results = self.sentiment_analyzer(text)
            
            # Process results
            scores = {}
            max_score = 0
            predicted_sentiment = 'neutral'
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                scores[label] = score
                
                if score > max_score:
                    max_score = score
                    predicted_sentiment = label
            
            return {
                'sentiment': predicted_sentiment,
                'confidence': max_score,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {}}
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key phrases from text.
        
        Args:
            text: Text to analyze
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrase dictionaries
        """
        try:
            doc = self.nlp(text)
            phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    phrase = {
                        'text': chunk.text,
                        'type': 'noun_phrase',
                        'start': chunk.start_char,
                        'end': chunk.end_char,
                        'importance': len(chunk.text.split())  # Simple importance score
                    }
                    phrases.append(phrase)
            
            # Extract named entities as key phrases
            for ent in doc.ents:
                phrase = {
                    'text': ent.text,
                    'type': 'named_entity',
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'importance': 5  # Higher importance for named entities
                }
                phrases.append(phrase)
            
            # Sort by importance and return top phrases
            phrases.sort(key=lambda x: x['importance'], reverse=True)
            return phrases[:max_phrases]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Generate summary of text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Text summary
        """
        if not self.summarizer or not text or len(text) < 100:
            return text[:max_length] if text else ""
        
        try:
            # Ensure text is not too long for the model
            if len(text) > 1024:
                text = text[:1024]
            
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] if text else ""
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate vector embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if not texts:
                return np.array([])
            
            # Clean texts
            cleaned_texts = [self.preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(cleaned_texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        # Split by sentences first
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text processing pipeline.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary containing all analysis results
        """
        if not text:
            return {}
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Extract entities
            entities = self.extract_entities(cleaned_text)
            
            # Extract relationships
            relationships = self.extract_relationships(cleaned_text)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(cleaned_text)
            
            # Extract key phrases
            key_phrases = self.extract_key_phrases(cleaned_text)
            
            # Generate summary
            summary = self.summarize_text(cleaned_text)
            
            # Chunk text
            chunks = self.chunk_text(cleaned_text)
            
            # Generate embeddings for chunks
            embeddings = self.generate_embeddings(chunks) if chunks else np.array([])
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'entities': entities,
                'relationships': relationships,
                'sentiment': sentiment,
                'key_phrases': key_phrases,
                'summary': summary,
                'chunks': chunks,
                'embeddings': embeddings.tolist() if embeddings.size > 0 else [],
                'processed_at': datetime.utcnow().isoformat(),
                'language': 'en',  # Could be detected
                'word_count': len(cleaned_text.split()),
                'character_count': len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Error in text processing pipeline: {e}")
            return {
                'original_text': text,
                'error': str(e),
                'processed_at': datetime.utcnow().isoformat()
            }

