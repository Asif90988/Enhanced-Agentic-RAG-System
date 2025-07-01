# storage/faiss_vector_store.py

class FAISSVectorStore:
    def __init__(self):
        print("⚠️ FAISS Vector Store is being used as a fallback.")

    def add(self, documents):
        print("FAISS add() called (no real action).")
        return

    def similarity_search(self, query, k=5):
        print("FAISS similarity_search() called (no real action).")
        return []

    def delete(self, doc_id):
        print("FAISS delete() called (no real action).")
        return