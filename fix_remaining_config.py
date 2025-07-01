# fix_remaining_config.py

import re
from pathlib import Path

config_path = Path("config.py")
original = config_path.read_text()

updated = original

# Step 1: Insert milvus_host and milvus_port into DatabaseConfig if not present
if "milvus_host" not in original:
    updated = re.sub(
        r"(class DatabaseConfig\(BaseModel\):\n)",
        r"\1    milvus_host: str = os.getenv(\"MILVUS_HOST\", \"localhost\")\n"
        r"    milvus_port: str = os.getenv(\"MILVUS_PORT\", \"19530\")\n",
        updated
    )
    print("✅ Inserted milvus_host and milvus_port into DatabaseConfig.")

# Step 2: Create NLPConfig class if missing
if "class NLPConfig(BaseModel):" not in original:
    nlp_config_class = '''
class NLPConfig(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    language: str = os.getenv("NLP_LANGUAGE", "en")
'''
    updated += "\n" + nlp_config_class
    print("✅ Added NLPConfig class.")

# Step 3: Add CONFIG.nlp = NLPConfig() if missing
if "nlp = NLPConfig()" not in original:
    updated = re.sub(
        r"(# Assign top-level CONFIG values.*?)(\nCONFIG = Config)",
        r"\1\n    nlp = NLPConfig()\2",
        updated,
        flags=re.DOTALL
    )
    print("✅ Assigned nlp = NLPConfig() in CONFIG.")

# Save updated file
config_path.write_text(updated)
print("💾 config.py updated successfully.")
