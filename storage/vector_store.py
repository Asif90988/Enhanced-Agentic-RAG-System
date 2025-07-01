# storage/vector_store.py

from storage.faiss_vector_store import FAISSVectorStore


class VectorStore:
    def __init__(self):
        self.store = FAISSVectorStore()

    def add_documents(self, documents):
        return self.store.add_documents(documents)

    def search(self, query, k=5):
        return self.store.search(query, k)

