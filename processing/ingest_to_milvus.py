import sys
import os

# ✅ Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from storage.milvus_vector_store import MilvusVectorStore
from processing.data_structurer import DataStructurer
from processing.nlp_pipeline import NLPProcessor

def ingest_documents_to_milvus():
    print("🔄 Initializing NLP pipeline...")
    processor = NLPProcessor()
    structurer = DataStructurer()
    vector_store = MilvusVectorStore()

    # Example document for testing ingestion
    documents = [
        {
            "source": "test.txt",
            "text": "Enhanced Agentic RAG systems help route queries intelligently to the right reasoning strategies."
        }
    ]

    print("📚 Processing documents...")
    processed_docs = [processor.process(doc["text"]) for doc in documents]
    structured_docs = [structurer.structure(doc) for doc in processed_docs]

    print("💾 Ingesting into Milvus vector store...")
    vector_store.ingest_documents(structured_docs)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_documents_to_milvus()
