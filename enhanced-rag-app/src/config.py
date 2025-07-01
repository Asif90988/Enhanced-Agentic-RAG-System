import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class AppConfig(BaseModel):
    debug: bool = True
    secret_key: str = "defaultsecret"

class DatabaseConfig(BaseModel):
    postgres_url: str = os.getenv("POSTGRES_URL", "sqlite:///default.db")
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "faiss")  # Options: faiss, milvus
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")

class NLPConfig(BaseModel):
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

class Config(BaseModel):
    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()
    nlp: NLPConfig = NLPConfig()
    vector_db_type: str = "faiss"  # or "weaviate", "qdrant", etc.

CONFIG = Config()
