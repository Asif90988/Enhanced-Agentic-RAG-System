"""
Data Processing Package for Enhanced Agentic RAG Application

This package contains components for processing and structuring unstructured data:
- Stream processors for real-time data transformation
- NLP pipelines for text analysis and entity extraction
- Data structuring and normalization
- Vector embedding generation
"""

from .stream_processor import StreamProcessor
from .nlp_pipeline import NLPPipeline
from .data_structurer import DataStructurer

__all__ = [
    'StreamProcessor',
    'NLPPipeline', 
    'DataStructurer'
]

