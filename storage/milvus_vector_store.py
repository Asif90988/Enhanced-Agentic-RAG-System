from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus import utility
import uuid

class MilvusVectorStore:
    def __init__(self, collection_name="rag_documents", dim=384, host="localhost", port="19530"):
        self.collection_name = collection_name
        self.dim = dim
        self.connect(host, port)
        self._create_schema()

    def connect(self, host, port):
        connections.connect(alias="default", host=host, port=port)

    def _create_schema(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="RAG document vector store")
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
            self.collection.load()

    def insert(self, documents):
        # documents: list of dicts with 'id', 'content', 'embedding'
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        self.collection.insert([ids, contents, embeddings])
        self.collection.flush()

    def search(self, query_embedding, top_k=5):
        self.collection.load()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["id", "content"]
        )
        hits = results[0]
        return [{"id": hit.id, "content": hit.entity.get("content")} for hit in hits]

