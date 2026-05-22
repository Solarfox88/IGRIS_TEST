from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


class EmbeddingStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                text_content TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def embed(self, text: str) -> Optional[np.ndarray]:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_EMBED_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return None

        emb = data.get("embedding")
        if not isinstance(emb, list):
            return None

        vec = np.asarray(emb, dtype=np.float32)
        if vec.shape != (EMBED_DIM,):
            return None
        return vec

    def upsert(self, node_id: str, node_type: str, text: str) -> bool:
        vec = self.embed(text)
        if vec is None:
            return False
        self._conn.execute(
            """
            INSERT INTO embeddings (node_id, node_type, text_content, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_type=excluded.node_type,
                text_content=excluded.text_content,
                vector=excluded.vector,
                created_at=excluded.created_at
            """,
            (node_id, node_type, text, vec.tobytes(), time.time()),
        )
        self._conn.commit()
        return True

    def search(self, query: str, top_k: int = 5, node_type: Optional[str] = None) -> List[dict]:
        qvec = self.embed(query)
        if qvec is None:
            return []

        params: tuple = ()
        sql = "SELECT node_id, node_type, text_content, vector FROM embeddings"
        if node_type:
            sql += " WHERE node_type = ?"
            params = (node_type,)
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        results = []
        qnorm = float(np.linalg.norm(qvec))
        for node_id, n_type, text_content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape != (EMBED_DIM,):
                continue
            score = float(np.dot(qvec, vec) / (qnorm * float(np.linalg.norm(vec)) + 1e-9))
            results.append(
                {
                    "node_id": node_id,
                    "node_type": n_type,
                    "text_content": text_content,
                    "score": score,
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: max(0, int(top_k))]

    def delete(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM embeddings WHERE node_id = ?", (node_id,))
        self._conn.commit()
