import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from igris.core.embedding_store import EMBED_DIM, EmbeddingStore
from igris.core.memory_graph import MemoryGraph


def _mock_urlopen_with_vector(vec):
    payload = json.dumps({"embedding": vec}).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return payload

    return _Response()


def test_embed_returns_none_on_ollama_error(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        assert store.embed("hello") is None


def test_upsert_returns_false_when_embed_fails(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    with patch.object(store, "embed", return_value=None):
        assert store.upsert("n1", "lesson", "hello") is False


def test_upsert_and_search_roundtrip(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_with_vector(vec)):
        assert store.upsert("n1", "lesson", "hello world") is True
        results = store.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0]["node_id"] == "n1"


def test_search_empty_db_returns_empty(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    vec = [0.0] * EMBED_DIM
    vec[1] = 1.0
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_with_vector(vec)):
        assert store.search("query") == []


def test_search_top_k_respected(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    vectors = []
    for i in range(5):
        v = [0.0] * EMBED_DIM
        v[i] = 1.0
        vectors.append(v)

    with patch.object(store, "embed", side_effect=[np.array(vectors[0], dtype=np.float32), *[np.array(v, dtype=np.float32) for v in vectors]]):
        for i in range(5):
            assert store.upsert(f"n{i}", "lesson", f"text {i}") is True
        results = store.search("query", top_k=2)
    assert len(results) == 2


def test_search_node_type_filter(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    qv = np.zeros(EMBED_DIM, dtype=np.float32)
    qv[0] = 1.0
    with patch.object(store, "embed", side_effect=[qv, qv, qv]):
        assert store.upsert("a", "lesson", "lesson") is True
        assert store.upsert("b", "goal", "goal") is True
        results = store.search("q", node_type="goal")
    assert len(results) == 1
    assert results[0]["node_id"] == "b"


def test_cosine_similarity_ordering(tmp_path):
    store = EmbeddingStore(str(tmp_path / "embeddings.db"))
    q = np.zeros(EMBED_DIM, dtype=np.float32)
    q[0] = 1.0
    a = q.copy()
    b = np.zeros(EMBED_DIM, dtype=np.float32)
    b[1] = 1.0
    with patch.object(store, "embed", side_effect=[a, b, q]):
        assert store.upsert("A", "lesson", "A") is True
        assert store.upsert("B", "lesson", "B") is True
        results = store.search("query", top_k=2)
    assert results[0]["node_id"] == "A"


def test_memory_graph_semantic_search_delegates(tmp_path):
    mg = MemoryGraph(str(tmp_path / "project"))
    with patch("igris.core.memory_graph.EmbeddingStore") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.search.return_value = [{"node_id": "n1", "node_type": "lesson", "text_content": "x", "score": 0.9}]
        out = mg.semantic_search("hello", top_k=3, node_type="lesson")
    assert out
    mock_store.search.assert_called_once_with(query="hello", top_k=3, node_type="lesson")


def test_memory_graph_index_node(tmp_path):
    mg = MemoryGraph(str(tmp_path / "project"))
    with patch("igris.core.memory_graph.EmbeddingStore") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.upsert.return_value = True
        ok = mg.index_node_for_search("n1", "lesson", "text")
    assert ok is True
    mock_store.upsert.assert_called_once_with(node_id="n1", node_type="lesson", text="text")
