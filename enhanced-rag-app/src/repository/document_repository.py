# src/repository/document_repository.py

import os
import sqlite3
from typing import List, Dict, Optional


class DocumentRepository:
    def __init__(self, db_path: str = "default.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_document(self, title: str, content: str, metadata: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO documents (title, content, metadata) VALUES (?, ?, ?)',
            (title, content, metadata)
        )
        conn.commit()
        conn.close()


    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        return [
        {"content": f"Dummy content {i+1} for '{query}'", "source": f"source_{i+1}.txt"}
        for i in range(top_k)
        ]




    def get_all_documents(self) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, metadata FROM documents')
        rows = cursor.fetchall()
        conn.close()
        return [
            {"id": row[0], "title": row[1], "content": row[2], "metadata": row[3]}
            for row in rows
        ]
