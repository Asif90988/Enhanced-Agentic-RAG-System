import logging
from config import CONFIG

class VectorStore:
    def __init__(self):
        self.config = CONFIG  # ✅ Fix: Use the top-level config

        try:
            if self.config.database.vector_db_type == "milvus":
                from storage.milvus_vector_store import MilvusVectorStore
                self.store = MilvusVectorStore()
            else:
                from storage.faiss_vector_store import FAISSVectorStore
                self.store = FAISSVectorStore()
        except Exception as e:
            logging.error(f"Failed to initialize {self.config.vector_db_type} vector store: {e}")
            from storage.faiss_vector_store import FAISSVectorStore
            self.store = FAISSVectorStore()

    def add(self, documents):
        return self.store.add(documents)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(query, k)

    def delete(self, doc_id):
        return self.store.delete(doc_id)
